#ifndef TRIAGE_SMS_H
#define TRIAGE_SMS_H

#include <array>
#include <cstdint>

#include "triage.h"

struct triage_sms : triage {
  using triage::triage;

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address, champsim::address, bool, bool, access_type, uint32_t metadata_in) { return metadata_in; }
  void prefetcher_on_demand_load(champsim::address addr, champsim::address ip, bool hit, bool useful_prefetch);
  void prefetcher_on_evict(champsim::address addr);

private:
  static constexpr uint64_t lines_per_region = 64; // 4 KiB region with 64 B cache lines.

  struct filter_entry {
    uint64_t region{};
    champsim::address trigger_pc{};
    unsigned trigger_offset{};
    uint64_t last_used{};
  };

  struct accumulation_entry {
    filter_entry generation{};
    uint64_t footprint{};
  };

  // A zero timestamp marks an empty entry. Replacing or clearing an entry ends its generation.
  std::array<filter_entry, 32> filter{};
  std::array<accumulation_entry, 32> accumulation{};
  uint64_t clock{};

  bool observe(champsim::block_number line, champsim::address ip);
};

#endif
