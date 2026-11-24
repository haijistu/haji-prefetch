#include "cache_stats.h"

cache_stats operator-(cache_stats lhs, cache_stats rhs)
{
  cache_stats result;
  result.pf_requested = lhs.pf_requested - rhs.pf_requested;
  result.pf_issued = lhs.pf_issued - rhs.pf_issued;
  result.pf_useful = lhs.pf_useful - rhs.pf_useful;
  result.pf_useless = lhs.pf_useless - rhs.pf_useless;
  result.pf_fill = lhs.pf_fill - rhs.pf_fill;
  result.partition = {lhs.partition.enabled,
                      lhs.partition.ways_at_end,
                      lhs.partition.changes - rhs.partition.changes,
                      lhs.partition.way_transfers - rhs.partition.way_transfers,
                      lhs.partition.data_evictions - rhs.partition.data_evictions,
                      lhs.partition.dirty_writebacks - rhs.partition.dirty_writebacks,
                      lhs.partition.writeback_stalls - rhs.partition.writeback_stalls,
                      lhs.partition.metadata_evictions - rhs.partition.metadata_evictions};
  result.sms_filter = {lhs.sms_filter.enabled,
                       lhs.sms_filter.l2_misses - rhs.sms_filter.l2_misses,
                       lhs.sms_filter.l2_prefetch_hits - rhs.sms_filter.l2_prefetch_hits,
                       lhs.sms_filter.passed_l2_misses - rhs.sms_filter.passed_l2_misses,
                       lhs.sms_filter.passed_l2_prefetch_hits - rhs.sms_filter.passed_l2_prefetch_hits,
                       {lhs.sms_filter.pairs_from_misses.same_region - rhs.sms_filter.pairs_from_misses.same_region,
                        lhs.sms_filter.pairs_from_misses.cross_region - rhs.sms_filter.pairs_from_misses.cross_region},
                       {lhs.sms_filter.pairs_from_prefetch_hits.same_region - rhs.sms_filter.pairs_from_prefetch_hits.same_region,
                        lhs.sms_filter.pairs_from_prefetch_hits.cross_region - rhs.sms_filter.pairs_from_prefetch_hits.cross_region}};
  result.triage_prefetch.mode = lhs.triage_prefetch.mode;
  for (std::size_t i = 0; i < result.triage_prefetch.sources.size(); ++i) {
    const auto& left = lhs.triage_prefetch.sources[i];
    const auto& right = rhs.triage_prefetch.sources[i];
    result.triage_prefetch.sources[i] =
        {left.issued - right.issued, left.timely_hits - right.timely_hits, left.late_merges - right.late_merges, left.duplicates - right.duplicates};
  }
  result.triangel = {lhs.triangel.enabled,
                     lhs.triangel.training_accesses - rhs.triangel.training_accesses,
                     lhs.triangel.history_samples - rhs.triangel.history_samples,
                     lhs.triangel.history_matches - rhs.triangel.history_matches,
                     lhs.triangel.second_chance_hits - rhs.triangel.second_chance_hits,
                     lhs.triangel.correlations - rhs.triangel.correlations,
                     lhs.triangel.lookahead2_correlations - rhs.triangel.lookahead2_correlations,
                     lhs.triangel.markov_lookups - rhs.triangel.markov_lookups,
                     lhs.triangel.reuse_buffer_hits - rhs.triangel.reuse_buffer_hits,
                     lhs.triangel.predictions_issued - rhs.triangel.predictions_issued};

  result.hits = lhs.hits - rhs.hits;
  result.misses = lhs.misses - rhs.misses;

  result.total_miss_latency_cycles = lhs.total_miss_latency_cycles - rhs.total_miss_latency_cycles;
  return result;
}
