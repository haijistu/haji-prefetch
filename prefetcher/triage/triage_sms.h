#ifndef TRIAGE_SMS_H
#define TRIAGE_SMS_H

#include <array>
#include <cstdint>
#include <utility>

#include "block.h"
#include "triage.h"

struct triage_sms : triage {
  explicit triage_sms(CACHE* cache) : triage_sms(cache, mode::filter) {}

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address, champsim::address, bool, bool, access_type, uint32_t metadata_in) { return metadata_in; }
  void prefetcher_cycle_operate();
  void prefetcher_on_demand_load(champsim::address addr, champsim::address ip, bool hit, bool useful_prefetch);
  void prefetcher_on_evict(champsim::address addr);

protected:
  enum class mode { filter, spatial, lookahead, lookahead_current_pc };
  triage_sms(CACHE* cache, mode selected_mode) : triage(cache), selection(selected_mode) {}

private:
  static constexpr uint64_t lines_per_region = 64; // 4 KiB region with 64 B cache lines.
  using source = champsim::triage_prefetch_source;

  struct filter_entry {
    uint64_t region{};
    champsim::address trigger_pc{};
    unsigned trigger_offset{};
    uint64_t last_used{};

    bool matches(uint64_t region_number) const { return last_used != 0 && region == region_number; }
  };

  struct accumulation_entry {
    filter_entry generation{};
    uint64_t footprint{};
  };

  struct pattern_entry {
    champsim::address trigger_pc{};
    unsigned trigger_offset{};
    uint64_t footprint{};
    uint64_t stable_footprint{};

    auto index() const { return (trigger_pc.to<uint64_t>() >> 2) ^ trigger_offset; }
    auto tag() const { return std::pair{trigger_pc, trigger_offset}; }
  };

  struct transition_entry {
    champsim::block_number trigger{};
    champsim::block_number target{};
    champsim::address target_pc{};

    auto index() const
    {
      const auto line = trigger.to<uint64_t>();
      return line ^ (line / lines_per_region);
    }
    auto tag() const { return trigger; }
  };

  struct spatial_prediction {
    champsim::block_number target;
    source origin;
  };

  // A zero timestamp marks an empty entry. Replacing or clearing an entry ends its generation.
  mode selection;
  std::array<filter_entry, 32> filter{};
  std::array<accumulation_entry, 32> accumulation{};
  uint64_t clock{};
  std::optional<champsim::msl::lru_table<pattern_entry>> patterns;
  std::optional<champsim::msl::lru_table<transition_entry>> transitions;
  std::deque<spatial_prediction> spatial_pending;

  bool observe(champsim::block_number line, champsim::address ip);
  void end_generation(accumulation_entry& entry);
  void queue_pattern(champsim::block_number trigger, champsim::address pc, source origin);
  void expand_transition(const prediction& temporal);
  bool issue(champsim::block_number target, source origin);
};

struct triage_sms_spatial : triage_sms {
  explicit triage_sms_spatial(CACHE* cache) : triage_sms(cache, mode::spatial) {}
};

struct triage_sms_lookahead : triage_sms {
  explicit triage_sms_lookahead(CACHE* cache) : triage_sms(cache, mode::lookahead) {}
};

struct triage_sms_lookahead_current_pc : triage_sms {
  explicit triage_sms_lookahead_current_pc(CACHE* cache) : triage_sms(cache, mode::lookahead_current_pc) {}
};

#endif
