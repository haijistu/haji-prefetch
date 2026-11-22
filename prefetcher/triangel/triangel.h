#ifndef TRIANGEL_H
#define TRIANGEL_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <optional>
#include <random>
#include <utility>

#include "address.h"
#include "chrono.h"
#include "modules.h"
#include "msl/fwcounter.h"

struct triangel : champsim::modules::prefetcher {
  using champsim::modules::prefetcher::prefetcher;

  static constexpr uint32_t prefetch_tag = 0x54524941;

  void prefetcher_initialize();
  void prefetcher_on_demand_load(champsim::address addr, champsim::address ip, bool hit, bool useful_prefetch, uint32_t fill_metadata);
  void prefetcher_cycle_operate();

private:
  using time_point = champsim::chrono::clock::time_point;
  using confidence = champsim::msl::fwcounter<4>;

  template <typename Entry, std::size_t Sets, std::size_t Ways>
  struct table {
    struct slot {
      uint64_t key{};
      uint64_t stamp{};
      Entry value{};
    };

    std::array<slot, Sets * Ways> entries{};
    uint64_t clock{};

    static constexpr auto capacity = Sets * Ways;

    auto set(uint64_t key)
    {
      auto first = std::next(entries.begin(), static_cast<std::ptrdiff_t>((key % Sets) * Ways));
      return std::pair{first, std::next(first, Ways)};
    }

    slot* find(uint64_t key)
    {
      auto [first, last] = set(key);
      auto hit = std::find_if(first, last, [key](const auto& candidate) { return candidate.stamp != 0 && candidate.key == key; });
      return hit == last ? nullptr : &*hit;
    }

    slot& select(uint64_t key)
    {
      if (auto* hit = find(key); hit != nullptr)
        return *hit;
      auto [first, last] = set(key);
      return *std::min_element(first, last, [](const auto& lhs, const auto& rhs) { return lhs.stamp < rhs.stamp; });
    }

    void touch(slot& entry) { entry.stamp = ++clock; }
    void write(slot& entry, uint64_t key, const Entry& value) { entry = {key, ++clock, value}; }
  };

  struct training_entry {
    uint64_t generation{};
    uint64_t sequence{};
    uint64_t last_line{};
    uint64_t previous_line{};
    confidence reuse{8};
    confidence pattern{8};
    confidence high_pattern{8};
    confidence sample_rate{8};
    bool two_distance{};
  };

  struct owner {
    uint64_t pc{};
    uint64_t generation{};
  };

  struct history_entry {
    owner origin{};
    uint64_t next{};
    uint64_t sequence{};
    bool reused{};
  };

  struct second_chance_entry {
    owner origin{};
    uint64_t timestamp{};
    bool used{};
  };

  struct reuse_entry {
    uint64_t target{};
    uint64_t revision{};
    time_point ready_at{};
  };

  struct prediction {
    uint64_t target{};
    time_point ready_at{};
  };

  table<training_entry, 32, 16> training;
  table<history_entry, 256, 2> history;
  table<second_chance_entry, 32, 2> second_chance;
  table<reuse_entry, 128, 2> reuse_buffer;
  std::deque<prediction> pending;
  std::minstd_rand random{1};
  champsim::msl::fwcounter<7> global_reuse{64};
  champsim::msl::fwcounter<7> global_pattern{64};
  champsim::msl::fwcounter<7> global_high_pattern{64};
  uint64_t training_generation{};
  uint64_t timestamp{};
  uint64_t markov_capacity{};

  training_entry* find_owner(owner origin);
  bool should_sample(const training_entry& entry);
  void change_reuse(training_entry& entry, int delta);
  void change_pattern(training_entry& entry, bool matched);
  void check_second_chance(uint64_t pc, uint64_t line);
  void give_second_chance(owner origin, uint64_t expected);
  void sample_history(uint64_t pc, training_entry& entry, uint64_t line);
  std::optional<prediction> lookup(uint64_t line, time_point earliest);
  void enqueue(prediction next);
  void predict(uint64_t line, unsigned degree);
};

#endif
