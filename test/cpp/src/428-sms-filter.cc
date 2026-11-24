#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <catch.hpp>

#include "../../../prefetcher/triage/triage_sms.h"
#include "cache.h"
#include "defaults.hpp"

namespace
{
constexpr champsim::data::bits offset_bits{6};
constexpr uint64_t l2_sets = 128;

champsim::address line_address(uint64_t line) { return champsim::address{champsim::block_number{line}}; }

template <typename Prefetcher = triage_sms>
struct sms_test_cache {
  champsim::channel upstream{8, 8, 8, offset_bits, false};
  champsim::channel downstream{1, 8, 8, offset_bits, false};
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}.sets(128).ways(16).metadata_ways(4).hit_latency(1)};
  CACHE l2;

  explicit sms_test_cache(uint32_t degree = 4)
      : l2(champsim::cache_builder{champsim::defaults::default_l2c}
               .sets(l2_sets)
               .ways(1)
               .hit_latency(1)
               .fill_latency(1)
               .upper_levels({&upstream})
               .lower_level(&downstream)
               .triage_sms_degree(degree)
               .prefetcher<Prefetcher>())
  {
    l2.connect_lower_cache(llc);
    l2.initialize();
    l2.warmup = false;
    l2.begin_phase();
  }

  void tick()
  {
    for (int i = 0; i < 5; ++i)
      l2._operate();
  }

  void seed_hit(uint64_t line, bool prefetched = false)
  {
    auto& slot = l2.block.at(line % l2_sets);
    REQUIRE_FALSE(slot.valid);
    slot.valid = true;
    slot.address = line_address(line);
    slot.prefetch = prefetched;
  }

  void issue(uint64_t line, bool instruction = false, champsim::address ip = champsim::address{0x100})
  {
    champsim::channel::request_type request;
    request.address = line_address(line);
    request.ip = ip;
    request.cpu = 0;
    request.is_instruction = instruction;
    REQUIRE(upstream.add_rq(request));
  }

  void complete_read()
  {
    REQUIRE(downstream.RQ.size() == 1);
    downstream.returned.emplace_back(downstream.RQ.front());
    downstream.RQ.pop_front();
    tick();
  }

  void complete_prefetch(uint64_t line)
  {
    const auto request = std::find_if(downstream.PQ.begin(), downstream.PQ.end(),
                                      [line](const auto& entry) { return entry.address == line_address(line); });
    REQUIRE(request != downstream.PQ.end());
    downstream.returned.emplace_back(*request);
    downstream.PQ.erase(request);
    tick();
  }

  void access(uint64_t line, bool instruction = false, champsim::address ip = champsim::address{0x100})
  {
    issue(line, instruction, ip);
    tick();
    if (!downstream.RQ.empty())
      complete_read();
  }
};
} // namespace

TEST_CASE("SMS filters both classes of Triage training requests at L2", "[sms-filter]")
{
  sms_test_cache test;
  test.seed_hit(65);
  test.access(65); // An ordinary hit opens the generation.
  test.access(66); // A different line promotes FT to AT and is filtered.
  test.seed_hit(67, true);
  test.access(67); // An AT hit filters a useful L2 prefetch hit.
  test.access(67); // The prefetch bit has now been consumed.
  test.access(130); // First Triage training sample.
  test.access(131); // In the same generation, so Triage skips it.
  test.seed_hit(196, true);
  test.access(196); // Second Triage training sample, despite the intervening miss.
  test.access(265, true); // An instruction miss does not open a generation.

  const auto& stats = test.l2.sim_stats.sms_filter;
  CHECK(stats.enabled);
  CHECK(stats.l2_misses == 3);
  CHECK(stats.l2_prefetch_hits == 2);
  CHECK(stats.passed_l2_misses == 1);
  CHECK(stats.passed_l2_prefetch_hits == 1);
  CHECK(test.llc.lookup_metadata(champsim::block_number{130}) == champsim::block_number{196});
  CHECK_FALSE(test.llc.lookup_metadata(champsim::block_number{131}));
  CHECK(test.l2.sim_stats.pf_issued == 0);

  test.l2.invalidate_entry(line_address(130));
  test.l2.invalidate_entry(line_address(196));
  test.access(130);
  REQUIRE(test.downstream.PQ.size() == 1);
  CHECK(test.downstream.PQ.front().address == line_address(196));

  test.access(266);
  CHECK(stats.passed_l2_misses == 3); // The earlier instruction miss did not filter this demand.
  test.l2.end_phase(0);
  CHECK(test.l2.roi_stats.sms_filter.passed_l2_misses == stats.passed_l2_misses);
}

TEST_CASE("SMS counts accepted misses once and carries generations into the ROI", "[sms-filter]")
{
  sms_test_cache test;
  test.l2.warmup = true;
  test.access(64);
  REQUIRE(test.l2.sim_stats.sms_filter.passed_l2_misses == 1);
  test.l2.warmup = false;
  test.l2.begin_phase();
  CHECK(test.l2.sim_stats.sms_filter.l2_misses == 0);
  CHECK(test.l2.sim_stats.sms_filter.l2_prefetch_hits == 0);

  champsim::channel::request_type blocker;
  blocker.address = line_address(512);
  REQUIRE(test.downstream.add_rq(blocker));
  test.issue(65);
  test.tick();
  test.tick();
  CHECK(test.l2.sim_stats.sms_filter.l2_misses == 0);

  test.downstream.RQ.pop_front();
  test.tick();
  test.complete_read();
  CHECK(test.l2.sim_stats.sms_filter.l2_misses == 1);
  CHECK(test.l2.sim_stats.sms_filter.passed_l2_misses == 0);
}

TEST_CASE("SMS classifies only Triage training pairs by physical region", "[sms-filter]")
{
  sms_test_cache test;
  test.l2.warmup = true;
  test.access(64); // The first sample for this PC has no predecessor.
  CHECK(test.l2.sim_stats.sms_filter.all_pairs().total() == 0);
  test.l2.invalidate_entry(line_address(64));
  test.l2.warmup = false;
  test.l2.begin_phase();

  test.access(65); // A new generation in the same region forms 64 -> 65.
  test.access(66); // SMS filters this miss because its region is already active.
  test.seed_hit(195, true);
  test.access(195); // A useful prefetch hit forms 65 -> 195 across regions.
  test.l2.invalidate_entry(line_address(195));
  test.access(195); // It passes SMS again, but Triage skips the repeated line.

  const auto& stats = test.l2.sim_stats.sms_filter;
  CHECK(stats.passed_l2_misses == 2);
  CHECK(stats.passed_l2_prefetch_hits == 1);
  CHECK(stats.pairs_from_misses.same_region == 1);
  CHECK(stats.pairs_from_misses.cross_region == 0);
  CHECK(stats.pairs_from_prefetch_hits.same_region == 0);
  CHECK(stats.pairs_from_prefetch_hits.cross_region == 1);
  CHECK(stats.all_pairs().total() == 2);
  CHECK(test.llc.lookup_metadata(champsim::block_number{64}) == champsim::block_number{65});
  CHECK(test.llc.lookup_metadata(champsim::block_number{65}) == champsim::block_number{195});

  test.l2.end_phase(0);
  CHECK(test.l2.roi_stats.sms_filter.all_pairs().total() == 2);
  test.l2.begin_phase();
  CHECK(test.l2.sim_stats.sms_filter.all_pairs().total() == 0);
}

TEST_CASE("SMS ends LRU generations when either table fills", "[sms-filter]")
{
  sms_test_cache test;
  triage_sms sms{&test.l2};
  sms.prefetcher_initialize();
  const auto observe = [&](uint64_t region, uint64_t offset, bool hit = true) {
    sms.prefetcher_on_demand_load(line_address(region * 64 + offset), champsim::address{0x100}, hit, false);
  };

  SECTION("Replacing a filter entry ends a single-line generation")
  {
    for (uint64_t region = 0; region < 32; ++region)
      observe(region, 0);
    observe(0, 0);
    observe(32, 0);
  }
  SECTION("Replacing an accumulation entry ends a multi-line generation")
  {
    for (uint64_t region = 0; region < 32; ++region) {
      observe(region, 0);
      observe(region, 1);
    }
    observe(0, 0);
    observe(32, 0);
    observe(32, 1);
  }
  observe(0, 0, false);
  CHECK(test.l2.sim_stats.sms_filter.passed_l2_misses == 0);
  observe(1, 0, false);
  CHECK(test.l2.sim_stats.sms_filter.passed_l2_misses == 1);
}

TEST_CASE("An SMS accumulation generation ends when an observed line leaves L2", "[sms-filter]")
{
  sms_test_cache test;
  test.access(64);
  test.access(65);
  test.seed_hit(66);
  test.l2.invalidate_entry(line_address(66)); // It was resident but absent from the footprint.
  test.access(66);
  CHECK(test.l2.sim_stats.sms_filter.passed_l2_misses == 1);
  test.access(193); // The fill replaces line 65, which belongs to the active footprint.
  CHECK_FALSE(test.l2.block.at(65).address == line_address(65));
  test.access(65);
  CHECK(test.l2.sim_stats.sms_filter.passed_l2_misses == 3);
}

TEST_CASE("Demand-triggered SMS replays a footprint once per line and attributes useful prefetches", "[sms-prefetch]")
{
  using source = champsim::triage_prefetch_source;
  sms_test_cache<triage_sms_spatial> test{2};
  for (const uint64_t line : {67, 69, 73, 75}) {
    test.seed_hit(line);
    test.access(line);
  }
  test.l2.invalidate_entry(line_address(69));

  // Both predictors want line 133. SMS can issue before the LLC metadata lookup finishes.
  test.llc.update_metadata(champsim::block_number{131}, champsim::block_number{133});
  test.seed_hit(131, true);
  test.access(131);
  test.tick();

  const auto& stats = test.l2.sim_stats.triage_prefetch;
  CHECK(stats.mode == "sms_spatial");
  REQUIRE(test.downstream.PQ.size() == 2);
  CHECK(test.downstream.PQ.at(0).address == line_address(133));
  CHECK(test.downstream.PQ.at(1).address == line_address(137));
  CHECK(stats.source(source::sms_demand).issued == 2);
  CHECK(stats.source(source::triage).duplicates == 1);

  test.complete_prefetch(133);
  test.access(133);
  test.access(137); // The request merges with an outstanding SMS prefetch.
  CHECK(stats.source(source::sms_demand).timely_hits == 1);
  CHECK(stats.source(source::sms_demand).late_merges == 1);
  CHECK(stats.source(source::sms_lookahead).issued == 0);
  test.l2.end_phase(0);
  CHECK(test.l2.roi_stats.triage_prefetch.source(source::sms_demand).late_merges == 1);
}

TEMPLATE_TEST_CASE("Only lookahead SMS expands a temporal prediction before demand reaches its target region", "[sms-prefetch]", triage_sms_spatial,
                   triage_sms_lookahead, triage_sms_lookahead_current_pc)
{
  using source = champsim::triage_prefetch_source;
  sms_test_cache<TestType> test;
  const bool stable_pattern = GENERATE(false, true);

  // Two completed footprints share offset 5. Offset 13 appears only in the newest footprint.
  for (const uint64_t base : {64, 128}) {
    if (base == 64 && !stable_pattern)
      continue;
    for (const auto offset : {3, 5, 9, 13})
      test.seed_hit(base + offset);
    test.access(base + 3);
    test.access(base + 5);
    test.access(base + (base == 64 ? 9 : 13));
    for (const auto offset : {3, 5, 9, 13})
      test.l2.invalidate_entry(line_address(base + offset));
  }

  constexpr uint64_t trigger = 260;
  constexpr uint64_t target = 323;
  constexpr uint64_t stable_neighbor = target + 2;
  constexpr uint64_t latest_neighbor = target + 10;
  test.seed_hit(trigger, true);
  test.seed_hit(target, true);
  test.seed_hit(stable_neighbor);
  test.seed_hit(latest_neighbor);
  test.access(trigger);
  test.access(target);
  for (const auto line : {trigger, target, stable_neighbor, latest_neighbor})
    test.l2.invalidate_entry(line_address(line));

  test.l2.begin_phase();
  test.access(trigger);
  const auto& stats = test.l2.sim_stats.triage_prefetch;
  const bool expanded = !std::is_same_v<TestType, triage_sms_spatial> && stable_pattern;
  REQUIRE(test.downstream.PQ.size() == (expanded ? 2 : 1));
  CHECK(test.downstream.PQ.front().address == line_address(target));
  CHECK(stats.source(source::triage).issued == 1);
  CHECK(stats.source(source::sms_demand).issued == 0);
  CHECK(stats.source(source::sms_lookahead).issued == (expanded ? 1 : 0));
  if (expanded)
    CHECK(test.downstream.PQ.back().address == line_address(stable_neighbor));

  test.complete_prefetch(target);
  test.access(target);
  test.tick();
  CHECK(stats.source(source::triage).timely_hits == 1);
  CHECK(stats.source(source::sms_demand).issued == (expanded ? 1 : 2));
  CHECK(test.downstream.PQ.back().address == line_address(latest_neighbor));
}

TEMPLATE_TEST_CASE("SMS lookahead selects the history for the recorded or current PC", "[sms-prefetch]", triage_sms_lookahead,
                   triage_sms_lookahead_current_pc)
{
  using source = champsim::triage_prefetch_source;
  sms_test_cache<TestType> test;
  const champsim::address recorded_pc{0x100};
  const champsim::address current_pc{0x200};
  constexpr uint64_t trigger = 260;
  constexpr uint64_t target = 323;

  for (const uint64_t base : {64, 128}) {
    for (const auto& [pc, offset] : {std::pair{recorded_pc, 5}, std::pair{current_pc, 7}}) {
      test.seed_hit(base + 3);
      test.seed_hit(base + offset);
      test.access(base + 3, false, pc);
      test.access(base + offset, false, pc);
      test.l2.invalidate_entry(line_address(base + 3));
      test.l2.invalidate_entry(line_address(base + offset));
    }
  }

  test.seed_hit(trigger, true);
  test.seed_hit(target, true);
  test.seed_hit(target + 2);
  test.access(trigger, false, recorded_pc);
  test.access(target, false, recorded_pc);
  for (const auto line : {trigger, target, target + 2})
    test.l2.invalidate_entry(line_address(line));

  test.l2.begin_phase();
  test.access(trigger, false, current_pc);

  constexpr bool uses_current_pc = std::is_same_v<TestType, triage_sms_lookahead_current_pc>;
  const auto& stats = test.l2.sim_stats.triage_prefetch;
  CHECK(stats.mode == (uses_current_pc ? "sms_lookahead_current_pc" : "sms_lookahead"));
  REQUIRE(test.downstream.PQ.size() == 2);
  CHECK(test.downstream.PQ.front().address == line_address(target));
  CHECK(test.downstream.PQ.back().address == line_address(target + (uses_current_pc ? 4 : 2)));
  CHECK(stats.source(source::sms_lookahead).issued == 1);
}

TEMPLATE_TEST_CASE("Only current-PC lookahead expands a cross-region prediction without an exact transition", "[sms-prefetch]", triage_sms_lookahead,
                   triage_sms_lookahead_current_pc)
{
  using source = champsim::triage_prefetch_source;
  sms_test_cache<TestType> test;
  const bool cross_region = GENERATE(false, true);
  constexpr uint64_t trigger = 260;
  const uint64_t target = cross_region ? 323 : 259;

  for (const uint64_t base : {64, 128}) {
    test.seed_hit(base + 3);
    test.seed_hit(base + 5);
    test.access(base + 3);
    test.access(base + 5);
    test.l2.invalidate_entry(line_address(base + 3));
    test.l2.invalidate_entry(line_address(base + 5));
  }

  // Supply a Markov prediction directly without observing a training pair at L2.
  test.llc.update_metadata(champsim::block_number{trigger}, champsim::block_number{target});
  test.seed_hit(trigger, true);
  test.access(trigger);
  test.tick();

  const bool expanded = cross_region && std::is_same_v<TestType, triage_sms_lookahead_current_pc>;
  REQUIRE(test.downstream.PQ.size() == (expanded ? 2 : 1));
  CHECK(test.downstream.PQ.front().address == line_address(target));
  if (expanded)
    CHECK(test.downstream.PQ.back().address == line_address(target + 2));
  CHECK(test.l2.sim_stats.triage_prefetch.source(source::sms_lookahead).issued == (expanded ? 1 : 0));
}
