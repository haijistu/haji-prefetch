#include <cstdint>

#include <catch.hpp>

#include "../../../prefetcher/triage/triage_sms.h"
#include "cache.h"
#include "defaults.hpp"

namespace
{
constexpr champsim::data::bits offset_bits{6};
constexpr uint64_t l2_sets = 128;

champsim::address line_address(uint64_t line) { return champsim::address{champsim::block_number{line}}; }

struct sms_test_cache {
  champsim::channel upstream{8, 8, 8, offset_bits, false};
  champsim::channel downstream{1, 8, 8, offset_bits, false};
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}.sets(128).ways(16).metadata_ways(4).hit_latency(1)};
  CACHE l2{champsim::cache_builder{champsim::defaults::default_l2c}
               .sets(l2_sets)
               .ways(1)
               .hit_latency(1)
               .fill_latency(1)
               .upper_levels({&upstream})
               .lower_level(&downstream)
               .prefetcher<triage_sms>()};

  sms_test_cache()
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

  void issue(uint64_t line, bool instruction = false)
  {
    champsim::channel::request_type request;
    request.address = line_address(line);
    request.ip = champsim::address{0x100};
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

  void access(uint64_t line, bool instruction = false)
  {
    issue(line, instruction);
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
