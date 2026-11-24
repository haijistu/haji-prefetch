#include "triage_sms.h"

#include <algorithm>
#include <stdexcept>

#include "cache.h"

void triage_sms::prefetcher_initialize()
{
  triage::prefetcher_initialize();
  if (intern_->OFFSET_BITS != champsim::data::bits{6})
    throw std::invalid_argument{"SMS requires 64-byte cache lines"};
  intern_->sim_stats.sms_filter.enabled = true;
  auto& stats = intern_->sim_stats.triage_prefetch;
  switch (selection) {
  case mode::filter:
    stats.mode = "sms_filter";
    return;
  case mode::spatial:
    stats.mode = "sms_spatial";
    break;
  case mode::lookahead:
    stats.mode = "sms_lookahead";
    transitions.emplace(2048, 4);
    break;
  case mode::lookahead_current_pc:
    stats.mode = "sms_lookahead_current_pc";
    break;
  }
  if (intern_->triage_sms_degree() == 0 || intern_->triage_sms_degree() >= lines_per_region)
    throw std::invalid_argument{"SMS prefetch degree must be between 1 and 63"};
  patterns.emplace(128, 4);
}

bool triage_sms::observe(champsim::block_number line, champsim::address ip)
{
  const auto line_number = line.to<uint64_t>();
  const auto region = line_number / lines_per_region;
  const auto offset = static_cast<unsigned>(line_number % lines_per_region);
  const auto matches = [region](const filter_entry& entry) { return entry.matches(region); };

  auto active = std::find_if(accumulation.begin(), accumulation.end(), [&](const auto& entry) { return matches(entry.generation); });
  if (active != accumulation.end()) {
    active->footprint |= uint64_t{1} << offset;
    active->generation.last_used = ++clock;
    return false;
  }

  auto filtered = std::find_if(filter.begin(), filter.end(), matches);
  if (filtered != filter.end()) {
    filtered->last_used = ++clock;
    if (filtered->trigger_offset != offset) {
      auto victim = std::min_element(accumulation.begin(), accumulation.end(),
                                     [](const auto& left, const auto& right) { return left.generation.last_used < right.generation.last_used; });
      end_generation(*victim);
      *victim = {*filtered, (uint64_t{1} << filtered->trigger_offset) | (uint64_t{1} << offset)};
      *filtered = {};
    }
    return false;
  }

  auto victim = std::min_element(filter.begin(), filter.end(), [](const auto& left, const auto& right) { return left.last_used < right.last_used; });
  *victim = {region, ip, offset, ++clock};
  return true;
}

void triage_sms::end_generation(accumulation_entry& entry)
{
  if (patterns && entry.generation.last_used != 0) {
    pattern_entry learned{entry.generation.trigger_pc, entry.generation.trigger_offset, entry.footprint};
    // Speculative replay uses only offsets observed in both recent generations.
    if (const auto previous = patterns->check_hit(learned); previous)
      learned.stable_footprint = previous->footprint & learned.footprint;
    patterns->fill(learned);
  }
  entry = {};
}

void triage_sms::prefetcher_on_demand_load(champsim::address addr, champsim::address ip, bool hit, bool useful_prefetch)
{
  const champsim::block_number current{addr};
  const bool passed = observe(current, ip);
  auto& stats = intern_->sim_stats.sms_filter;
  if (!hit) {
    ++stats.l2_misses;
    stats.passed_l2_misses += passed;
  } else if (useful_prefetch) {
    ++stats.l2_prefetch_hits;
    stats.passed_l2_prefetch_hits += passed;
  }

  if (passed && (!hit || useful_prefetch)) {
    if (const auto previous = train(current, ip); previous) {
      auto& pairs = hit ? stats.pairs_from_prefetch_hits : stats.pairs_from_misses;
      const bool same_region = previous->to<uint64_t>() / lines_per_region == current.to<uint64_t>() / lines_per_region;
      ++(same_region ? pairs.same_region : pairs.cross_region);
      if (transitions) {
        if (same_region)
          transitions->invalidate({*previous});
        else
          transitions->fill({*previous, current, ip});
      }
    }
  }
  if (passed && patterns)
    queue_pattern(current, ip, source::sms_demand);
}

void triage_sms::prefetcher_on_evict(champsim::address addr)
{
  const auto line_number = champsim::block_number{addr}.to<uint64_t>();
  const auto region = line_number / lines_per_region;
  const auto offset = static_cast<unsigned>(line_number % lines_per_region);
  const auto matches = [region](const filter_entry& entry) { return entry.matches(region); };

  auto filtered = std::find_if(filter.begin(), filter.end(), matches);
  if (filtered != filter.end())
    *filtered = {};

  auto active = std::find_if(accumulation.begin(), accumulation.end(), [&](const auto& entry) { return matches(entry.generation); });
  if (active != accumulation.end() && (active->footprint & (uint64_t{1} << offset)) != 0)
    end_generation(*active);
}

void triage_sms::queue_pattern(champsim::block_number trigger, champsim::address pc, source origin)
{
  const auto line = trigger.to<uint64_t>();
  const auto trigger_offset = static_cast<unsigned>(line % lines_per_region);
  const auto pattern = patterns->check_hit({pc, trigger_offset});
  if (!pattern)
    return;

  const auto footprint = origin == source::sms_lookahead ? pattern->stable_footprint : pattern->footprint;
  const auto region_base = line - trigger_offset;
  auto remaining = intern_->triage_sms_degree();
  for (unsigned offset = 0; offset < lines_per_region && remaining != 0 && spatial_pending.size() < intern_->PQ_SIZE; ++offset) {
    if (offset == trigger_offset || (footprint & (uint64_t{1} << offset)) == 0)
      continue;

    const champsim::block_number target{region_base + offset};
    const bool already_queued =
        std::any_of(spatial_pending.begin(), spatial_pending.end(), [target](const auto& entry) { return entry.target == target; });
    if (already_queued || intern_->has_pending_or_cached_line(target)) {
      ++intern_->sim_stats.triage_prefetch.source(origin).duplicates;
      continue;
    }
    spatial_pending.push_back({target, origin});
    --remaining;
  }
}

void triage_sms::expand_transition(const prediction& temporal)
{
  const auto region = temporal.target.to<uint64_t>() / lines_per_region;
  auto target_pc = temporal.trigger_pc;
  if (transitions) {
    // Match the exact transition alongside the compressed LLC lookup.
    const auto transition = transitions->check_hit({temporal.trigger});
    if (!transition || transition->target != temporal.target)
      return;
    target_pc = transition->target_pc;
  } else if (selection != mode::lookahead_current_pc || temporal.trigger.to<uint64_t>() / lines_per_region == region) {
    return;
  }

  if (std::any_of(filter.begin(), filter.end(), [region](const auto& entry) { return entry.matches(region); })
      || std::any_of(accumulation.begin(), accumulation.end(), [region](const auto& entry) { return entry.generation.matches(region); }))
    return;

  queue_pattern(temporal.target, target_pc, source::sms_lookahead);
}

bool triage_sms::issue(champsim::block_number target, source origin)
{
  if (intern_->has_pending_or_cached_line(target)) {
    ++intern_->sim_stats.triage_prefetch.source(origin).duplicates;
    return true;
  }
  return intern_->prefetch_line(champsim::address{target}, true, 0, origin);
}

void triage_sms::prefetcher_cycle_operate()
{
  if (!patterns) {
    triage::prefetcher_cycle_operate();
    return;
  }
  // Ready Triage candidates claim the shared L2 queue before SMS candidates.
  while (!pending.empty() && pending.front().ready_at <= intern_->current_time) {
    if (!issue(pending.front().target, source::triage))
      return;
    expand_transition(pending.front());
    pending.pop_front();
  }
  while (!spatial_pending.empty()) {
    if (!issue(spatial_pending.front().target, spatial_pending.front().origin))
      return;
    spatial_pending.pop_front();
  }
}
