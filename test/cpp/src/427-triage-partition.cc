#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <catch.hpp>

#include "../../../prefetcher/triage/triage.h"
#include "cache.h"
#include "defaults.hpp"
#include "mocks.hpp"

TEST_CASE("A fixed LLC partition limits data and Markov metadata independently", "[triage-partition]")
{
  const auto reserved = GENERATE(0u, 1u, 2u, 4u, 8u);
  CAPTURE(reserved);
  constexpr auto physical_ways = 16u;
  constexpr auto sets = 2u;
  const auto data_ways = physical_ways - reserved;
  const auto metadata_pairs = reserved * 12;

  do_nothing_MRC memory;
  to_rq_MRP requests;
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}
                .sets(sets)
                .ways(physical_ways)
                .metadata_ways(reserved)
                .hit_latency(1)
                .fill_latency(1)
                .upper_levels({&requests.queues})
                .lower_level(&memory.queues)};
  const std::array<champsim::operable*, 3> components{&requests, &llc, &memory};
  auto tick = [&] {
    for (auto* component : components)
      component->_operate();
  };
  for (auto* component : components) {
    component->warmup = false;
    component->initialize();
    component->begin_phase();
  }

  auto access = [&](uint64_t line) {
    champsim::channel::request_type request;
    request.address = champsim::address{line * 64};
    request.cpu = 0;
    request.instr_id = line;
    REQUIRE(requests.issue(request));
    for (int cycle = 0; cycle < 40; ++cycle)
      tick();
  };
  auto find_data = [&](uint64_t line) {
    const auto addr = champsim::address{line * 64};
    return std::any_of(llc.block.begin(), llc.block.end(), [&](const auto& block) { return block.valid && block.address == addr; });
  };

  access(1); // A line in set one must survive all accesses to set zero.
  for (uint64_t i = 1; i <= data_ways; ++i)
    access(2 * i);
  access(2); // Make line 2 MRU so line 4 is the next LRU victim.
  access(2 * (data_ways + 1));
  CHECK(find_data(1));
  CHECK(find_data(2));
  CHECK_FALSE(find_data(4));
  CHECK(find_data(2 * (data_ways + 1)));
  CHECK(llc.NUM_WAY == physical_ways);
  CHECK(llc.data_ways() == data_ways);
  CHECK(std::count_if(llc.block.begin(), llc.block.begin() + physical_ways, [](const auto& block) { return block.valid; }) == data_ways);
  for (std::size_t set = 0; set < sets; ++set) {
    for (std::size_t way = data_ways; way < physical_ways; ++way)
      CHECK_FALSE(llc.block.at(set * physical_ways + way).valid);
  }

  llc.update_metadata(champsim::block_number{1}, champsim::block_number{101});
  if (reserved == 0) {
    CHECK_FALSE(llc.lookup_metadata(champsim::block_number{1}));
  } else {
    for (uint64_t i = 0; i < metadata_pairs; ++i)
      llc.update_metadata(champsim::block_number{2 * i}, champsim::block_number{0x7fff'ffff - i});
    for (uint64_t i = 0; i < metadata_pairs; ++i)
      CHECK(llc.lookup_metadata(champsim::block_number{2 * i}) == champsim::block_number{0x7fff'ffff - i});
    llc.update_metadata(champsim::block_number{2 * metadata_pairs}, champsim::block_number{1000});
    CHECK_FALSE(llc.lookup_metadata(champsim::block_number{0}));
    CHECK(llc.lookup_metadata(champsim::block_number{2 * metadata_pairs}) == champsim::block_number{1000});
    CHECK(llc.lookup_metadata(champsim::block_number{1}) == champsim::block_number{101});
    CHECK(find_data(1));
    CHECK(find_data(2));
  }
}

TEST_CASE("Markov entries use a folded trigger tag and preserve 31-bit target line numbers", "[triage-partition]")
{
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}.sets(1).ways(2).metadata_ways(1)};
  const champsim::block_number low{0};
  const champsim::block_number high{(uint64_t{1} << 31) - 1};
  const champsim::block_number overflow{uint64_t{1} << 31};
  const champsim::block_number alternating{0x05a69c3};
  const champsim::block_number alias{1025}; // Folds to tag zero in a one-set LLC.

  CHECK_FALSE(llc.lookup_metadata(low));
  llc.update_metadata(low, high);
  llc.update_metadata(high, low);
  CHECK(llc.lookup_metadata(low) == high);
  CHECK(llc.lookup_metadata(high) == low);
  CHECK(llc.lookup_metadata(alias) == high);

  llc.update_metadata(high, alternating);
  llc.update_metadata(low, overflow);
  CHECK(llc.lookup_metadata(high) == alternating);
  CHECK(llc.lookup_metadata(low) == high);

  llc.update_metadata(alias, alternating);
  CHECK(llc.lookup_metadata(low) == alternating);
}

TEST_CASE("Markov confidence protects a target for one conflicting observation", "[triage-partition]")
{
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}.sets(1).ways(2).metadata_ways(1)};
  const champsim::block_number trigger{1};
  const champsim::block_number first{10};
  const champsim::block_number second{20};
  const champsim::block_number third{30};

  llc.update_metadata(trigger, first);
  CHECK(llc.lookup_metadata(trigger) == first);
  llc.update_metadata(trigger, second);
  CHECK(llc.lookup_metadata(trigger) == second);
  llc.update_metadata(trigger, second);
  llc.update_metadata(trigger, third);
  CHECK(llc.lookup_metadata(trigger) == second);
  llc.update_metadata(trigger, third);
  CHECK(llc.lookup_metadata(trigger) == third);
}

TEST_CASE("LLC repartition preserves dirty data and waits for write queue space", "[triage-partition]")
{
  const auto offset = champsim::data::bits{6};
  champsim::channel upper{8, 8, 8, offset, false};
  champsim::channel memory{8, 8, 1, offset, false};
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}
                .sets(2)
                .ways(4)
                .metadata_max_ways(2)
                .upper_levels({&upper})
                .lower_level(&memory)
                .hit_latency(1)
                .fill_latency(1)};
  llc.initialize();
  llc.warmup = false;
  llc.begin_phase();
  const auto tick = [&llc](int cycles) {
    for (int cycle = 0; cycle < cycles; ++cycle)
      llc._operate();
  };

  const champsim::block_number line{6};
  const champsim::address address{line};
  const champsim::address data{0xabcd};
  champsim::channel::request_type store;
  store.address = address;
  store.data = champsim::address{0x1234};
  store.type = access_type::WRITE;
  store.response_requested = false;
  // Fill set zero in way order so the fourth store becomes the retiring line.
  for (uint64_t i = 0; i < 4; ++i) {
    store.address = champsim::address{champsim::block_number{2 * i}};
    REQUIRE(upper.add_wq(store));
    tick(4);
  }
  REQUIRE(llc.block.at(3).address == address);
  REQUIRE(llc.block.at(3).dirty);
  CHECK(llc.block.at(3).data == champsim::address{0x1234});
  store.data = data;
  store.pf_metadata = 5;
  REQUIRE(upper.add_wq(store));
  tick(4);
  const champsim::block_number clean_line{7};
  llc.block.at(7).valid = true;
  llc.block.at(7).address = champsim::address{clean_line};

  store.address = champsim::address{0x1000}; // Occupy the only downstream write slot.
  REQUIRE(memory.add_wq(store));
  REQUIRE(llc.request_metadata_ways(1));
  CHECK_FALSE(llc.request_metadata_ways(2));
  tick(1);
  CHECK(llc.data_ways(0) == 4);
  CHECK(llc.sim_stats.partition.writeback_stalls == 1);

  champsim::channel::request_type load;
  load.address = address;
  REQUIRE(upper.add_rq(load));
  tick(4);
  REQUIRE(upper.returned.size() == 1);
  CHECK(upper.returned.front().data == data);
  upper.returned.clear();

  memory.WQ.pop_front();
  tick(1);
  CHECK(llc.data_ways(0) == 3);
  CHECK(llc.data_ways(1) == 4);
  CHECK_FALSE(llc.block.at(3).valid);
  REQUIRE(memory.WQ.size() == 1);
  CHECK(memory.WQ.front().address == address);
  CHECK(memory.WQ.front().data == data);
  CHECK(memory.WQ.front().pf_metadata == 5);
  CHECK_FALSE(memory.WQ.front().response_requested);
  llc.update_metadata(line, clean_line);
  CHECK(llc.lookup_metadata(line) == clean_line);

  REQUIRE(upper.add_rq(load));
  tick(4);
  CHECK(llc.metadata_ways() == 1);
  CHECK(llc.data_ways(1) == 3);
  CHECK_FALSE(llc.block.at(7).valid);
  CHECK(memory.RQ.empty());
  CHECK(upper.returned.empty());

  const auto written = memory.WQ.front();
  memory.WQ.pop_front();
  tick(4);
  REQUIRE(memory.RQ.size() == 1);
  CHECK(memory.RQ.front().address == address);
  memory.RQ.pop_front();
  memory.returned.emplace_back(written);
  tick(4);
  REQUIRE(upper.returned.size() == 1);
  CHECK(upper.returned.front().data == data);
  CHECK_FALSE(llc.block.at(3).valid);
  CHECK(llc.sim_stats.partition.dirty_writebacks == 1);
  CHECK(llc.sim_stats.partition.data_evictions == 2);

  REQUIRE(llc.request_metadata_ways(0));
  tick(2);
  CHECK(llc.data_ways(0) == 4);
  CHECK(llc.sim_stats.partition.metadata_evictions == 1);
  REQUIRE(llc.request_metadata_ways(1));
  tick(2);
  CHECK_FALSE(llc.lookup_metadata(line));
}

TEST_CASE("The dynamic partition controller can grow from zero and return idle ways to data", "[triage-partition]")
{
  champsim::channel upper, memory;
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}
                .sets(1)
                .ways(2)
                .metadata_max_ways(1)
                .metadata_partition_epoch(2)
                .upper_levels({&upper})
                .lower_level(&memory)
                .hit_latency(1)};
  llc.initialize();
  llc.warmup = false;
  llc.begin_phase();
  const champsim::block_number trigger{1}, target{2};
  llc.block.front().valid = true;
  llc.block.front().address = champsim::address{trigger};
  const auto load = [&] {
    champsim::channel::request_type request;
    request.address = champsim::address{trigger};
    REQUIRE(upper.add_rq(request));
    for (int cycle = 0; cycle < 4; ++cycle)
      llc._operate();
    REQUIRE(upper.returned.size() == 1);
    upper.returned.clear();
  };

  for (int i = 0; i < 4; ++i) {
    llc.update_metadata(trigger, target);
    load();
  }
  CHECK(llc.metadata_ways() == 1);
  llc.update_metadata(trigger, target);
  for (int i = 0; i < 4; ++i)
    load();
  CHECK(llc.metadata_ways() == 0);
  CHECK_FALSE(llc.lookup_metadata(trigger));
  CHECK(llc.sim_stats.partition.changes == 2);
}

TEST_CASE("L2 Triage trains a Markov pair in LLC and replays it after the lookup delay", "[triage-partition]")
{
  const auto reserved = GENERATE(0u, 1u);
  constexpr int lookup_delay = 12;
  do_nothing_MRC memory;
  to_rq_MRP requests;
  champsim::channel l2_llc{16, 16, 16, champsim::data::bits{6}, false};
  CACHE llc{champsim::cache_builder{champsim::defaults::default_llc}
                .sets(1)
                .ways(16)
                .metadata_ways(reserved)
                .hit_latency(lookup_delay)
                .fill_latency(1)
                .upper_levels({&l2_llc})
                .lower_level(&memory.queues)};
  CACHE l2{champsim::cache_builder{champsim::defaults::default_l2c}
               .sets(1)
               .ways(1)
               .hit_latency(1)
               .fill_latency(1)
               .upper_levels({&requests.queues})
               .lower_level(&l2_llc)
               .prefetcher<triage>()};
  l2.connect_lower_cache(llc);

  const std::array<champsim::operable*, 4> components{&requests, &l2, &llc, &memory};
  for (auto* component : components) {
    component->initialize();
    component->warmup = false;
    component->begin_phase();
  }
  auto tick = [&](int cycles) {
    for (int cycle = 0; cycle < cycles; ++cycle) {
      for (auto* component : components)
        component->_operate();
    }
  };
  auto issue = [&](uint64_t line, uint64_t id) {
    champsim::channel::request_type request;
    request.cpu = 0;
    request.instr_id = id;
    request.ip = champsim::address{0x100};
    request.address = champsim::address{line * 64};
    REQUIRE(requests.issue(request));
  };

  issue(3, 1);
  tick(60);
  issue(7, 2);
  tick(60);
  CHECK(l2.sim_stats.pf_issued == 0);
  if (reserved != 0)
    CHECK(llc.lookup_metadata(champsim::block_number{3}) == champsim::block_number{7});
  else
    CHECK_FALSE(llc.lookup_metadata(champsim::block_number{3}));

  issue(3, 3);
  tick(lookup_delay);
  CHECK(l2.sim_stats.pf_issued == 0);
  tick(60);
  CHECK(l2.sim_stats.pf_issued == reserved);
  if (reserved != 0) {
    REQUIRE(l2.block.front().valid);
    CHECK(l2.block.front().address == champsim::address{7 * 64});
  }
}
