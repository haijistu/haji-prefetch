#include "triage_sms.h"

#include <algorithm>
#include <stdexcept>

#include "cache.h"

void triage_sms::prefetcher_initialize()
{
  triage::prefetcher_initialize();
  if (intern_->OFFSET_BITS != champsim::data::bits{6})
    throw std::invalid_argument{"The SMS filter requires 64-byte cache lines"};
  intern_->sim_stats.sms_filter.enabled = true;
}

bool triage_sms::observe(champsim::block_number line, champsim::address ip)
{
  const auto line_number = line.to<uint64_t>();
  const auto region = line_number / lines_per_region;
  const auto offset = static_cast<unsigned>(line_number % lines_per_region);
  const auto matches = [region](const filter_entry& entry) { return entry.last_used != 0 && entry.region == region; };

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
      *victim = {*filtered, (uint64_t{1} << filtered->trigger_offset) | (uint64_t{1} << offset)};
      *filtered = {};
    }
    return false;
  }

  auto victim = std::min_element(filter.begin(), filter.end(), [](const auto& left, const auto& right) { return left.last_used < right.last_used; });
  *victim = {region, ip, offset, ++clock};
  return true;
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
    }
  }
}

void triage_sms::prefetcher_on_evict(champsim::address addr)
{
  const auto line_number = champsim::block_number{addr}.to<uint64_t>();
  const auto region = line_number / lines_per_region;
  const auto offset = static_cast<unsigned>(line_number % lines_per_region);
  const auto matches = [region](const filter_entry& entry) { return entry.last_used != 0 && entry.region == region; };

  auto filtered = std::find_if(filter.begin(), filter.end(), matches);
  if (filtered != filter.end())
    *filtered = {};

  auto active = std::find_if(accumulation.begin(), accumulation.end(), [&](const auto& entry) { return matches(entry.generation); });
  if (active != accumulation.end() && (active->footprint & (uint64_t{1} << offset)) != 0)
    *active = {};
}
