#ifndef TRIAGE_H
#define TRIAGE_H

#include <cstdint>
#include <deque>
#include <optional>

#include "address.h"
#include "chrono.h"
#include "modules.h"
#include "msl/lru_table.h"

struct triage : champsim::modules::prefetcher {
  using champsim::modules::prefetcher::prefetcher;

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  void prefetcher_cycle_operate();

protected:
  // Returns the source line when training submits a correlation to the Markov table.
  std::optional<champsim::block_number> train(champsim::block_number current, champsim::address ip);

private:
  struct training_entry {
    champsim::address ip{};
    champsim::block_number last_address{};

    auto index() const
    {
      using namespace champsim::data::data_literals;
      return ip.slice_upper<2_b>();
    }
    auto tag() const { return ip; }
  };

  struct prediction {
    champsim::block_number target;
    champsim::chrono::clock::time_point ready_at;
  };

  champsim::msl::lru_table<training_entry> training_table{32, 16};
  std::deque<prediction> pending;
};

#endif
