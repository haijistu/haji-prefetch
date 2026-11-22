#include "triangel.h"

#include <stdexcept>

#include "cache.h"

namespace
{
constexpr uint64_t second_chance_window = 512;
constexpr unsigned max_degree = 4;

uint64_t training_key(uint64_t pc)
{
  // Fold in the low instruction-address bits for traces with unaligned PCs.
  return pc ^ (pc >> 2);
}
} // namespace

void triangel::prefetcher_initialize()
{
  auto* llc = intern_->get_lower_cache();
  if (llc == nullptr || intern_->virtual_prefetch || intern_->OFFSET_BITS != champsim::data::bits{6} || llc->OFFSET_BITS != intern_->OFFSET_BITS)
    throw std::invalid_argument{"Triangel requires an L2 with physical addresses, 64-byte lines and a lower cache for metadata"};

  markov_capacity = llc->max_metadata_entries();
  llc->enable_metadata_dueller();
  intern_->sim_stats.triangel.enabled = true;
}

triangel::training_entry* triangel::find_owner(owner origin)
{
  auto* prior = training.find(training_key(origin.pc));
  return prior != nullptr && prior->value.generation == origin.generation ? &prior->value : nullptr;
}

bool triangel::should_sample(const training_entry& entry)
{
  const auto scale = uint64_t{1} << entry.sample_rate.value();
  const auto reuse_discount = entry.reuse.value() < 3 ? 16.0 : 1.0;
  const auto probability = decltype(history)::capacity * static_cast<double>(scale) / (static_cast<double>(markov_capacity) * 256.0 * reuse_discount);
  return std::bernoulli_distribution{std::min(1.0, probability)}(random);
}

void triangel::change_reuse(training_entry& entry, int delta)
{
  entry.reuse += delta;
  global_reuse += delta;
}

void triangel::change_pattern(training_entry& entry, bool matched)
{
  const auto pattern_delta = matched ? 1 : -2;
  const auto high_delta = matched ? 1 : -5;
  entry.pattern += pattern_delta;
  entry.high_pattern += high_delta;
  global_pattern += pattern_delta;
  global_high_pattern += high_delta;
}

void triangel::check_second_chance(uint64_t pc, uint64_t line)
{
  auto* sample = second_chance.find(line);
  if (sample == nullptr || sample->value.used)
    return;

  auto& candidate = sample->value;
  if (auto* prior = find_owner(candidate.origin); prior != nullptr) {
    if (timestamp - candidate.timestamp >= second_chance_window) {
      change_pattern(*prior, false);
    } else if (candidate.origin.pc == pc) {
      change_pattern(*prior, true);
      ++intern_->sim_stats.triangel.second_chance_hits;
    } else {
      return;
    }
  }
  candidate.used = true;
}

void triangel::give_second_chance(owner origin, uint64_t expected)
{
  auto& victim = second_chance.select(expected);
  if (victim.stamp != 0 && !victim.value.used) {
    if (auto* prior = find_owner(victim.value.origin); prior != nullptr)
      change_pattern(*prior, false);
  }
  second_chance.write(victim, expected, {origin, timestamp, false});
}

void triangel::sample_history(uint64_t pc, training_entry& entry, uint64_t line)
{
  const owner origin{pc, entry.generation};
  if (auto* sample = history.find(entry.last_line); sample != nullptr && sample->value.origin.generation == entry.generation) {
    auto& record = sample->value;
    const auto distance = entry.sequence - record.sequence;
    if (distance != 0 && distance < markov_capacity)
      change_reuse(entry, 1);
    else if (!record.reused)
      change_reuse(entry, -1);

    ++intern_->sim_stats.triangel.history_matches;
    if (record.next == line)
      change_pattern(entry, true);
    else if (!intern_->has_demanded_line(champsim::block_number{record.next}))
      give_second_chance(origin, record.next);

    record.reused = true;
    record.next = line;
    history.touch(*sample);
    return;
  }

  if (!should_sample(entry))
    return;

  auto& victim = history.select(entry.last_line);
  if (victim.stamp != 0) {
    const auto& old = victim.value;
    if (auto* prior = find_owner(old.origin); prior != nullptr) {
      const auto distance = prior->sequence - old.sequence;
      if (distance >= markov_capacity) {
        if (!old.reused)
          change_reuse(*prior, -1);
        ++entry.sample_rate;
      } else if (distance != 0 && !old.reused) {
        --entry.sample_rate;
      }
    }
  }

  history.write(victim, entry.last_line, {origin, line, entry.sequence, false});
  ++intern_->sim_stats.triangel.history_samples;
}

std::optional<triangel::prediction> triangel::lookup(uint64_t line, time_point earliest)
{
  auto* llc = intern_->get_lower_cache();
  auto& stats = intern_->sim_stats.triangel;
  const champsim::block_number trigger{line};
  const auto revision = llc->metadata_revision(trigger);
  if (auto* cached = reuse_buffer.find(line); cached != nullptr && cached->value.revision == revision) {
    ++stats.reuse_buffer_hits;
    llc->observe_reused_metadata(trigger);
    return prediction{cached->value.target, std::max(cached->value.ready_at, earliest)};
  }

  ++stats.markov_lookups;
  const auto target = llc->lookup_metadata(trigger);
  if (!target)
    return std::nullopt;

  const auto target_line = target->to<uint64_t>();
  const auto ready_at = earliest + llc->HIT_LATENCY;
  reuse_buffer.write(reuse_buffer.select(line), line, {target_line, revision, ready_at});
  return prediction{target_line, ready_at};
}

void triangel::enqueue(prediction next)
{
  const auto duplicate = std::find_if(pending.begin(), pending.end(), [&](const auto& pred) { return pred.target == next.target; });
  if (duplicate != pending.end()) {
    if (duplicate->ready_at <= next.ready_at)
      return;
    pending.erase(duplicate);
  }
  if (pending.size() >= intern_->PQ_SIZE)
    return;

  const auto position = std::upper_bound(pending.begin(), pending.end(), next.ready_at, [](auto ready_at, const auto& pred) { return ready_at < pred.ready_at; });
  pending.insert(position, next);
}

void triangel::predict(uint64_t line, unsigned degree)
{
  std::array<uint64_t, max_degree + 1> seen{line};
  auto ready_at = intern_->current_time;
  for (unsigned step = 0; step < degree; ++step) {
    if (pending.size() >= intern_->PQ_SIZE)
      return;
    auto next = lookup(line, ready_at);
    if (!next || std::find(seen.begin(), seen.begin() + step + 1, next->target) != seen.begin() + step + 1)
      return;

    enqueue(*next);
    seen[step + 1] = line = next->target;
    ready_at = next->ready_at;
  }
}

void triangel::prefetcher_on_demand_load(champsim::address addr, champsim::address ip, bool hit, bool useful_prefetch, uint32_t fill_metadata)
{
  if (markov_capacity == 0 || (hit && (!useful_prefetch || fill_metadata != prefetch_tag)))
    return;

  auto& stats = intern_->sim_stats.triangel;
  ++stats.training_accesses;
  ++timestamp;
  const auto line = champsim::block_number{addr}.to<uint64_t>();
  const auto pc = ip.to<uint64_t>();
  const auto key = training_key(pc);
  check_second_chance(pc, line);
  auto* slot = training.find(key);
  if (slot == nullptr) {
    training_entry fresh{};
    const bool confident = global_reuse.value() > 64 && global_pattern.value() > 64 && global_high_pattern.value() > 64;
    if (confident || should_sample(fresh)) {
      fresh.generation = ++training_generation;
      fresh.last_line = line;
      training.write(training.select(key), key, fresh);
    }
    return;
  }

  training.touch(*slot);
  auto& entry = slot->value;
  if (entry.last_line == line)
    return;

  ++entry.sequence;
  sample_history(pc, entry, line);
  const auto pattern_threshold = global_pattern.value() > 64 ? 7 : 8;
  const auto high_threshold = global_high_pattern.value() > 64 ? 7 : 8;
  const auto reuse_threshold = global_reuse.value() > 64 ? 7 : 8;
  if (entry.high_pattern.value() >= 14)
    entry.two_distance = true;
  if (entry.pattern.value() < pattern_threshold)
    entry.two_distance = false;

  if (entry.reuse.value() > reuse_threshold && entry.pattern.value() > pattern_threshold) {
    const bool lookahead = entry.two_distance && entry.sequence >= 2;
    const auto trigger = lookahead ? entry.previous_line : entry.last_line;
    if (trigger != line) {
      intern_->get_lower_cache()->update_metadata(champsim::block_number{trigger}, champsim::block_number{line});
      ++stats.correlations;
      stats.lookahead2_correlations += lookahead;
    }
    predict(line, entry.high_pattern.value() > high_threshold ? max_degree : 1);
  }

  entry.previous_line = entry.last_line;
  entry.last_line = line;
}

void triangel::prefetcher_cycle_operate()
{
  while (!pending.empty() && pending.front().ready_at <= intern_->current_time) {
    if (!prefetch_line(champsim::address{champsim::block_number{pending.front().target}}, true, prefetch_tag))
      break;
    ++intern_->sim_stats.triangel.predictions_issued;
    pending.pop_front();
  }
}
