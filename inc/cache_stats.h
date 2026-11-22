#ifndef CACHE_STATS_H
#define CACHE_STATS_H

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "channel.h"
#include "event_counter.h"

struct cache_stats {
  std::string name;
  struct metadata_partition_stats {
    bool enabled{};
    uint32_t ways_at_end{};
    uint64_t changes{};
    uint64_t way_transfers{};
    uint64_t data_evictions{};
    uint64_t dirty_writebacks{};
    uint64_t writeback_stalls{};
    uint64_t metadata_evictions{};
  } partition{};
  struct sms_filter_stats {
    struct training_pairs {
      uint64_t same_region{};
      uint64_t cross_region{};

      uint64_t total() const { return same_region + cross_region; }
    };

    bool enabled{};
    uint64_t l2_misses{};
    uint64_t l2_prefetch_hits{};
    uint64_t passed_l2_misses{};
    uint64_t passed_l2_prefetch_hits{};
    training_pairs pairs_from_misses{};
    training_pairs pairs_from_prefetch_hits{};

    training_pairs all_pairs() const
    {
      return {pairs_from_misses.same_region + pairs_from_prefetch_hits.same_region, pairs_from_misses.cross_region + pairs_from_prefetch_hits.cross_region};
    }
  } sms_filter{};
  struct triangel_stats {
    bool enabled{};
    uint64_t training_accesses{};
    uint64_t history_samples{};
    uint64_t history_matches{};
    uint64_t second_chance_hits{};
    uint64_t correlations{};
    uint64_t lookahead2_correlations{};
    uint64_t markov_lookups{};
    uint64_t reuse_buffer_hits{};
    uint64_t predictions_issued{};
  } triangel{};
  // prefetch stats
  uint64_t pf_requested = 0;
  uint64_t pf_issued = 0;
  uint64_t pf_useful = 0;
  uint64_t pf_useless = 0;
  uint64_t pf_fill = 0;

  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> hits = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> misses = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> miss_merge = {};
  champsim::stats::event_counter<std::pair<access_type, std::remove_cv_t<decltype(NUM_CPUS)>>> fill = {};

  long total_miss_latency_cycles{};
};

cache_stats operator-(cache_stats lhs, cache_stats rhs);

#endif
