#include <cstdint>

#include <catch.hpp>

#include "../../../prefetcher/triangel/triangel.h"
#include "cache.h"
#include "defaults.hpp"

namespace
{
constexpr champsim::data::bits offset_bits{6};
const champsim::address ip{0x100};

champsim::block_number block_number(uint64_t line) { return champsim::block_number{line}; }
champsim::address line_address(uint64_t line) { return champsim::address{block_number(line)}; }
} // namespace

TEST_CASE("Triangel learns lookahead-2 and reuses Markov lookups in a repeating sequence", "[triangel]")
{
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}.sets(8).ways(4).metadata_ways(2).hit_latency(4)};
  CACHE l2{champsim::cache_builder{champsim::defaults::default_l2c}.sets(8).pq_size(16)};
  l2.connect_lower_cache(llc);
  triangel prefetcher{&l2};
  prefetcher.prefetcher_initialize();

  SECTION("Training consumes misses and tagged hits, and predictions wait for LLC metadata")
  {
    for (uint64_t i = 0; i < 16; ++i) {
      prefetcher.prefetcher_on_demand_load(line_address(100 + i), ip, true, false, triangel::prefetch_tag);
      prefetcher.prefetcher_on_demand_load(line_address(100 + i), ip, true, true, 0);
    }
    CHECK(l2.sim_stats.triangel.training_accesses == 0);

    for (uint64_t i = 0; i < 900; ++i)
      prefetcher.prefetcher_on_demand_load(line_address(100 + i % 9), ip, true, true, triangel::prefetch_tag);

    const auto& stats = l2.sim_stats.triangel;
    CHECK(stats.training_accesses == 900);
    CHECK(stats.history_samples > 0);
    CHECK(stats.history_matches > 0);
    CHECK(stats.lookahead2_correlations > 0);
    CHECK(stats.markov_lookups > 0);
    CHECK(stats.reuse_buffer_hits > 0);
    CHECK(llc.lookup_metadata(block_number(100)) == block_number(102));

    prefetcher.prefetcher_cycle_operate();
    CHECK(stats.predictions_issued == 0);
    l2.current_time += llc.HIT_LATENCY * 4;
    prefetcher.prefetcher_cycle_operate();
    CHECK(stats.predictions_issued > 1);
    CHECK(stats.predictions_issued <= l2.PQ_SIZE);
  }

  SECTION("The second chance sampler rewards a nearby inexact successor")
  {
    constexpr uint64_t sequence[]{100, 101, 104, 100, 102, 101, 105};
    for (uint64_t i = 0; i < 900; ++i)
      prefetcher.prefetcher_on_demand_load(line_address(sequence[i % 7]), ip, false, false, 0);

    CHECK(l2.sim_stats.triangel.history_matches > 0);
    CHECK(l2.sim_stats.triangel.second_chance_hits > 0);
  }
}

TEST_CASE("Triangel duels metadata and data capacities and resizes the existing LLC partition", "[triangel]")
{
  champsim::channel upper{8, 8, 8, offset_bits, false};
  champsim::channel memory{8, 8, 8, offset_bits, false};
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}
                .sets(1)
                .ways(2)
                .metadata_max_ways(1)
                .metadata_partition_epoch(4)
                .upper_levels({&upper})
                .lower_level(&memory)
                .hit_latency(1)
                .fill_latency(1)};
  llc.enable_metadata_dueller();
  llc.warmup = false;
  llc.begin_phase();

  const auto access = [&](uint64_t line) {
    champsim::channel::request_type request;
    request.address = line_address(line);
    request.cpu = 0;
    REQUIRE(upper.add_rq(request));
    for (int tick = 0; tick < 12; ++tick) {
      llc._operate();
      if (!memory.RQ.empty()) {
        memory.returned.emplace_back(memory.RQ.front());
        memory.RQ.pop_front();
      }
    }
    REQUIRE(upper.returned.size() == 1);
    upper.returned.clear();
  };

  const auto trigger = block_number(10);
  const auto target = block_number(11);
  for (int i = 0; i < 4; ++i) {
    llc.update_metadata(trigger, target);
    CHECK_FALSE(llc.lookup_metadata(trigger)); // The dueller can see hits at hypothetical sizes.
    access(10);
  }
  REQUIRE(llc.metadata_ways() == 1);

  llc.update_metadata(trigger, target);
  llc.update_metadata(trigger, target);
  const auto revision = llc.metadata_revision(trigger);
  llc.update_metadata(trigger, block_number(12));
  CHECK(llc.metadata_revision(trigger) == revision); // Confidence initially protects the target.
  llc.update_metadata(trigger, block_number(12));
  CHECK(llc.metadata_revision(trigger) != revision);
  REQUIRE(llc.lookup_metadata(trigger) == block_number(12));

  const auto before_shrink = llc.metadata_revision(trigger);
  for (int i = 0; i < 4; ++i)
    access(10 + (i + 1) % 2);
  CHECK(llc.metadata_ways() == 0);
  CHECK(llc.metadata_revision(trigger) != before_shrink);
  CHECK_FALSE(llc.lookup_metadata(trigger));
  CHECK(llc.sim_stats.partition.changes == 2);
}

TEST_CASE("L2 forwards the resident prefetch tag and the real demand PC to Triangel", "[triangel]")
{
  champsim::channel upper{8, 8, 8, offset_bits, false};
  champsim::channel lower{8, 8, 8, offset_bits, false};
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}.sets(8).ways(4).metadata_ways(2)};
  CACHE l2{champsim::cache_builder{champsim::defaults::default_l2c}
               .sets(8)
               .ways(1)
               .upper_levels({&upper})
               .lower_level(&lower)
               .hit_latency(1)
               .prefetcher<triangel>()};
  l2.connect_lower_cache(llc);
  l2.initialize();
  l2.warmup = false;
  l2.begin_phase();

  auto access = [&](uint64_t line, uint32_t tag, champsim::address pc) {
    CAPTURE(line, tag, pc);
    auto& resident = l2.block.at(line % 8);
    resident.address = line_address(line);
    resident.valid = true;
    resident.prefetch = true;
    resident.pf_metadata = tag;
    champsim::channel::request_type demand;
    demand.address = resident.address;
    demand.ip = pc;
    demand.cpu = 0;
    REQUIRE(upper.add_rq(demand));
    for (int tick = 0; tick < 16 && upper.returned.empty(); ++tick)
      l2._operate();
    REQUIRE(upper.returned.size() == 1);
    upper.returned.clear();
  };

  for (uint64_t i = 0; i < 40; ++i) {
    access(100 + i % 4, triangel::prefetch_tag, ip);
    access(200 + i % 4, 0, ip);
    access(300 + i % 4, triangel::prefetch_tag, champsim::address{0x104});
  }

  CHECK(l2.sim_stats.triangel.training_accesses == 80);
  CHECK(llc.lookup_metadata(block_number(100)) == block_number(102));
  CHECK(llc.lookup_metadata(block_number(300)) == block_number(302));
  CHECK_FALSE(llc.lookup_metadata(block_number(200)));
  l2.end_phase(0);
  CHECK(l2.roi_stats.triangel.training_accesses == 80);
}
