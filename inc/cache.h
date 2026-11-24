/*
 *    Copyright 2023 The ChampSim Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CACHE_H
#define CACHE_H

#ifdef CHAMPSIM_MODULE
#define SET_ASIDE_CHAMPSIM_MODULE
#undef CHAMPSIM_MODULE
#endif

#include <array>
#include <cstddef> // for size_t
#include <cstdint> // for uint64_t, uint32_t, uint8_t
#include <deque>
#include <iterator> // for size
#include <limits>   // for numeric_limits
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "address.h"
#include "bandwidth.h"
#include "block.h"
#include "cache_builder.h"
#include "cache_stats.h"
#include "champsim.h"
#include "channel.h"
#include "chrono.h"
#include "modules.h"
#include "operable.h"
#include "util/to_underlying.h" // for to_underlying
#include "waitable.h"

class lru;

class CACHE : public champsim::operable
{
  enum [[deprecated(
      "Prefetchers may not specify arbitrary fill levels. Use CACHE::prefetch_line(pf_addr, fill_this_level, prefetch_metadata) instead.")]] FILL_LEVEL{
      FILL_L1 = 1, FILL_L2 = 2, FILL_LLC = 4, FILL_DRC = 8, FILL_DRAM = 16};

  using channel_type = champsim::channel;
  using request_type = typename channel_type::request_type;
  using response_type = typename channel_type::response_type;

  struct tag_lookup_type {
    champsim::address address;
    champsim::address v_address;
    champsim::address data;
    champsim::address ip;
    uint64_t instr_id;

    uint32_t pf_metadata;
    uint32_t cpu;

    access_type type;
    bool prefetch_from_this;
    bool skip_fill;
    bool is_translated;
    bool is_instruction;
    bool translate_issued = false;

    uint8_t asid[2] = {std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::max()};

    champsim::chrono::clock::time_point event_cycle = champsim::chrono::clock::time_point::max();

    std::vector<uint64_t> instr_depend_on_me{};
    std::vector<std::deque<response_type>*> to_return{};
    champsim::triage_prefetch_source prefetch_source{};

    explicit tag_lookup_type(request_type req) : tag_lookup_type(req, false, false) {}
    tag_lookup_type(const request_type& req, bool local_pref, bool skip);
  };

public:
  struct fill_type {
    champsim::address address;
    champsim::address v_address;
    champsim::address ip;
    uint64_t instr_id;

    struct returned_value {
      champsim::address data;
      uint32_t pf_metadata;
    };
    champsim::waitable<returned_value> data_promise{};
    uint32_t cpu;

    access_type type;
    bool prefetch_from_this;
    champsim::triage_prefetch_source prefetch_source{};

    uint8_t asid[2] = {std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::max()};

    champsim::chrono::clock::time_point time_enqueued;

    std::vector<uint64_t> instr_depend_on_me{};
    std::vector<std::deque<response_type>*> to_return{};

    fill_type(const tag_lookup_type& req, champsim::chrono::clock::time_point _time_enqueued);
    static fill_type merge(fill_type predecessor, fill_type successor);
  };

private:
  bool try_hit(const tag_lookup_type& handle_pkt);
  void notify_prefetcher_demand_load(const tag_lookup_type& packet, bool hit, bool useful_prefetch, uint32_t fill_metadata = 0);
  bool handle_fill(const fill_type& fill);
  bool enqueue_writeback(const champsim::cache_block& victim, uint32_t cpu, uint64_t instr_id);
  bool handle_miss(const tag_lookup_type& handle_pkt);
  bool handle_write(const tag_lookup_type& handle_pkt);
  void finish_packet(const response_type& packet);
  void finish_translation(const response_type& packet);

  void issue_translation(tag_lookup_type& q_entry) const;

public:
  using BLOCK = champsim::cache_block;

private:
  static BLOCK fill_block(fill_type fill, uint32_t metadata);
  using set_type = std::vector<BLOCK>;

  static constexpr unsigned metadata_tag_bits = 10;
  static constexpr unsigned metadata_target_bits = 31;
  static constexpr unsigned metadata_confidence_bit = metadata_tag_bits + metadata_target_bits;
  static constexpr unsigned metadata_entry_bits = metadata_confidence_bit + 1;
  static constexpr uint64_t metadata_tag_mask = (uint64_t{1} << metadata_tag_bits) - 1;
  static constexpr uint64_t metadata_target_mask = (uint64_t{1} << metadata_target_bits) - 1;
  static constexpr std::size_t metadata_entries_per_way = 512 / metadata_entry_bits;

  struct metadata_line {
    // Twelve 42-bit entries in a 64-byte line, leaving the top eight bits unused.
    std::array<uint64_t, 8> payload{};
    // LRU timestamps model replacement state separately. A zero timestamp is invalid.
    std::array<uint64_t, metadata_entries_per_way> last_used{};

    [[nodiscard]] uint64_t read(std::size_t slot) const;
    void write(std::size_t slot, uint64_t entry);
  };

  struct metadata_position {
    metadata_line* line{};
    std::size_t slot{};
    bool hit{};
  };

  struct metadata_dueller {
    struct sampled_set {
      std::vector<uint64_t> data;
      std::vector<uint64_t> markov;
    };

    std::vector<int> sample_by_set;
    std::vector<sampled_set> samples;
    std::vector<uint64_t> scores;
  };

  struct metadata_partition {
    std::vector<uint32_t> ways_by_set;
    uint32_t target;
    uint32_t epoch_length;
    uint32_t next_set{};
    uint32_t demand_accesses{};
    uint32_t demand_misses{};
    uint64_t lookups{};
    uint64_t hits{};
    uint64_t training_pairs{};
    int last_vote{};
    std::optional<metadata_dueller> dueller{};
  };

  [[nodiscard]] uint64_t metadata_tag(uint64_t trigger) const;
  [[nodiscard]] metadata_position find_metadata(uint64_t trigger, uint64_t tag);
  [[nodiscard]] uint32_t metadata_ways_for_set(std::size_t set) const;
  void observe_partition_access(champsim::address addr, bool hit);
  void observe_dueller_metadata(uint64_t trigger, bool allocate);
  void observe_dueller_data(uint64_t line);
  long advance_metadata_partition();
  uint32_t reserved_metadata_ways{};
  uint32_t metadata_capacity_ways{};
  std::optional<metadata_partition> partition{};
  std::vector<metadata_line> metadata_table{};
  uint64_t metadata_clock{};
  std::vector<uint64_t> metadata_set_revisions{};
  CACHE* lower_cache{};
  uint32_t sms_degree{};

  std::pair<set_type::iterator, set_type::iterator> get_set_span(champsim::address address);
  [[nodiscard]] std::pair<set_type::const_iterator, set_type::const_iterator> get_set_span(champsim::address address) const;
  [[nodiscard]] long get_set_index(champsim::address address) const;

  template <typename T>
  bool should_activate_prefetcher(const T& pkt) const;

  template <bool>
  auto initiate_tag_check(champsim::channel* ul = nullptr);

  template <typename T>
  champsim::address module_address(const T& element) const;

  auto matches_address(champsim::address address) const;
  std::pair<fill_type, request_type> mshr_and_forward_packet(const tag_lookup_type& handle_pkt);

  std::deque<tag_lookup_type> internal_PQ{};
  std::deque<tag_lookup_type> inflight_tag_check{};
  std::deque<tag_lookup_type> translation_stash{};

public:
  std::vector<channel_type*> upper_levels;
  channel_type* lower_level;
  channel_type* lower_translate;

  uint32_t cpu = 0;
  std::string NAME;
  uint32_t NUM_SET, NUM_WAY, MSHR_SIZE;
  [[nodiscard]] uint32_t data_ways() const { return NUM_WAY - reserved_metadata_ways; }
  [[nodiscard]] uint32_t data_ways(std::size_t set) const { return NUM_WAY - metadata_ways_for_set(set); }
  [[nodiscard]] uint32_t metadata_ways() const { return reserved_metadata_ways; }
  [[nodiscard]] uint32_t max_metadata_ways() const { return metadata_capacity_ways; }
  [[nodiscard]] uint64_t max_metadata_entries() const { return uint64_t{NUM_SET} * metadata_capacity_ways * metadata_entries_per_way; }
  bool request_metadata_ways(uint32_t ways);
  void enable_metadata_dueller();
  void observe_reused_metadata(champsim::block_number trigger);
  [[nodiscard]] uint64_t metadata_revision(champsim::block_number trigger) const;
  [[nodiscard]] bool has_demanded_line(champsim::block_number line) const;
  [[nodiscard]] bool has_pending_or_cached_line(champsim::block_number line) const;
  [[nodiscard]] uint32_t triage_sms_degree() const { return sms_degree; }
  void connect_lower_cache(CACHE& cache) { lower_cache = &cache; }
  [[nodiscard]] CACHE* get_lower_cache() const { return lower_cache; }
  [[nodiscard]] std::optional<champsim::block_number> lookup_metadata(champsim::block_number trigger);
  void update_metadata(champsim::block_number trigger, champsim::block_number target);
  std::size_t PQ_SIZE;
  champsim::chrono::clock::duration HIT_LATENCY;
  champsim::chrono::clock::duration FILL_LATENCY;
  champsim::data::bits OFFSET_BITS;
  set_type block{static_cast<typename set_type::size_type>(NUM_SET * NUM_WAY)};
  champsim::bandwidth::maximum_type MAX_TAG, MAX_FILL;
  bool prefetch_as_load;
  bool match_offset_bits;
  bool virtual_prefetch;
  std::vector<access_type> pref_activate_mask;

  using stats_type = cache_stats;

  stats_type sim_stats, roi_stats;

  std::deque<fill_type> MSHR;
  std::deque<fill_type> inflight_fills;

  long operate() final;
  void initialize() final;
  void begin_phase() final;
  void end_phase(unsigned cpu) final;

  [[deprecated]] std::size_t get_occupancy(uint8_t queue_type, champsim::address address) const;
  [[deprecated]] std::size_t get_size(uint8_t queue_type, champsim::address address) const;

  // NOLINTBEGIN
  [[deprecated("get_occupancy() returns 0 for every input except 0 (MSHR). Use get_mshr_occupancy() instead.")]] std::size_t
  get_occupancy(uint8_t queue_type, uint64_t address) const;
  [[deprecated("get_size() returns 0 for every input except 0 (MSHR). Use get_mshr_size() instead.")]] std::size_t get_size(uint8_t queue_type,
                                                                                                                            uint64_t address) const;
  // NOLINTEND

  [[nodiscard]] std::size_t get_mshr_occupancy() const;
  [[nodiscard]] std::size_t get_mshr_size() const;
  [[nodiscard]] double get_mshr_occupancy_ratio() const;

  [[nodiscard]] std::vector<std::size_t> get_rq_occupancy() const;
  [[nodiscard]] std::vector<std::size_t> get_rq_size() const;
  [[nodiscard]] std::vector<double> get_rq_occupancy_ratio() const;

  [[nodiscard]] std::vector<std::size_t> get_wq_occupancy() const;
  [[nodiscard]] std::vector<std::size_t> get_wq_size() const;
  [[nodiscard]] std::vector<double> get_wq_occupancy_ratio() const;

  [[nodiscard]] std::vector<std::size_t> get_pq_occupancy() const;
  [[nodiscard]] std::vector<std::size_t> get_pq_size() const;
  [[nodiscard]] std::vector<double> get_pq_occupancy_ratio() const;

  [[deprecated("Use get_set_index() instead.")]] [[nodiscard]] uint64_t get_set(uint64_t address) const;
  [[deprecated("This function should not be used to access the blocks directly.")]] [[nodiscard]] uint64_t get_way(uint64_t address, uint64_t set) const;

  long invalidate_entry(champsim::address inval_addr);
  bool prefetch_line(champsim::address pf_addr, bool fill_this_level, uint32_t prefetch_metadata,
                     champsim::triage_prefetch_source source = champsim::triage_prefetch_source::none);

  [[deprecated]] bool prefetch_line(uint64_t pf_addr, bool fill_this_level, uint32_t prefetch_metadata);

  [[deprecated("Use CACHE::prefetch_line(pf_addr, fill_this_level, prefetch_metadata) instead.")]] bool
  prefetch_line(uint64_t ip, uint64_t base_addr, uint64_t pf_addr, bool fill_this_level, uint32_t prefetch_metadata);

  void print_deadlock() final;

#include "module_decl.inc"

  struct prefetcher_module_concept {
    virtual ~prefetcher_module_concept() = default;

    virtual void bind(CACHE* cache) = 0;

    virtual void impl_prefetcher_initialize() = 0;
    virtual void impl_prefetcher_on_demand_load(champsim::address addr, champsim::address ip, bool hit, bool useful_prefetch, uint32_t fill_metadata) = 0;
    virtual void impl_prefetcher_on_evict(champsim::address addr) = 0;
    virtual uint32_t impl_prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                                   uint32_t metadata_in) = 0;
    virtual uint32_t impl_prefetcher_cache_fill(champsim::address addr, long set, long way, bool prefetch, champsim::address evicted_addr,
                                                uint32_t metadata_in) = 0;
    virtual void impl_prefetcher_cycle_operate() = 0;
    virtual void impl_prefetcher_final_stats() = 0;
    virtual void impl_prefetcher_branch_operate(champsim::address ip, uint8_t branch_type, champsim::address branch_target) = 0;
  };

  struct replacement_module_concept {
    virtual ~replacement_module_concept() = default;

    virtual void bind(CACHE* cache) = 0;

    virtual void impl_initialize_replacement() = 0;
    virtual long impl_find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const BLOCK* current_set, champsim::address ip,
                                  champsim::address full_addr, access_type type) = 0;
    virtual void impl_update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                               champsim::address victim_addr, access_type type, bool hit) = 0;
    virtual void impl_replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                             champsim::address victim_addr, access_type type) = 0;
    virtual void impl_replacement_final_stats() = 0;
  };

  template <typename... Ps>
  struct prefetcher_module_model final : prefetcher_module_concept {
    std::tuple<Ps...> intern_;
    explicit prefetcher_module_model(CACHE* cache) : intern_(Ps{cache}...) { (void)cache; /* silence -Wunused-but-set-parameter when sizeof...(Ps) == 0 */ }
    void bind(CACHE* cache)
    {
      std::apply([cache = cache](auto&... p) { (..., p.bind(cache)); }, intern_);
    }

    void impl_prefetcher_initialize() final;
    void impl_prefetcher_on_demand_load(champsim::address addr, champsim::address ip, bool hit, bool useful_prefetch, uint32_t fill_metadata) final
    {
      [[maybe_unused]] auto notify = [&](auto& p) {
        using prefetcher = champsim::modules::prefetcher;
        if constexpr (prefetcher::has_demand_load<decltype(p), champsim::address, champsim::address, bool, bool, uint32_t>)
          p.prefetcher_on_demand_load(addr, ip, hit, useful_prefetch, fill_metadata);
        else if constexpr (prefetcher::has_demand_load<decltype(p), champsim::address, champsim::address, bool, bool>)
          p.prefetcher_on_demand_load(addr, ip, hit, useful_prefetch);
      };
      std::apply([&](auto&... p) { (..., notify(p)); }, intern_);
    }

    void impl_prefetcher_on_evict(champsim::address addr) final
    {
      [[maybe_unused]] auto notify = [&](auto& p) {
        if constexpr (champsim::modules::prefetcher::has_eviction<decltype(p), champsim::address>)
          p.prefetcher_on_evict(addr);
      };
      std::apply([&](auto&... p) { (..., notify(p)); }, intern_);
    }
    [[nodiscard]] uint32_t impl_prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                                         uint32_t metadata_in) final;
    [[nodiscard]] uint32_t impl_prefetcher_cache_fill(champsim::address addr, long set, long way, bool prefetch, champsim::address evicted_addr,
                                                      uint32_t metadata_in) final;
    void impl_prefetcher_cycle_operate() final;
    void impl_prefetcher_final_stats() final;
    void impl_prefetcher_branch_operate(champsim::address ip, uint8_t branch_type, champsim::address branch_target) final;
  };

  template <typename... Rs>
  struct replacement_module_model final : replacement_module_concept {
    // Assert that at least one has an update state
    // static_assert(std::disjunction<champsim::is_detected<has_update_state, Rs>...>::value, "At least one replacement policy must update its state");

    std::tuple<Rs...> intern_;
    explicit replacement_module_model(CACHE* cache) : intern_(Rs{cache}...) { (void)cache; /* silence -Wunused-but-set-parameter when sizeof...(Rs) == 0 */ }
    void bind(CACHE* cache)
    {
      std::apply([cache = cache](auto&... r) { (..., r.bind(cache)); }, intern_);
    }

    void impl_initialize_replacement() final;
    [[nodiscard]] long impl_find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const BLOCK* current_set, champsim::address ip,
                                        champsim::address full_addr, access_type type) final;
    void impl_update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                       champsim::address victim_addr, access_type type, bool hit) final;
    void impl_replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                     champsim::address victim_addr, access_type type) final;
    void impl_replacement_final_stats() final;
  };

  std::unique_ptr<prefetcher_module_concept> pref_module_pimpl;
  std::unique_ptr<replacement_module_concept> repl_module_pimpl;

  // NOLINTBEGIN(readability-make-member-function-const): legacy modules use non-const hooks
  void impl_prefetcher_initialize() const;
  [[nodiscard]] uint32_t impl_prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                                       uint32_t metadata_in) const;
  [[nodiscard]] uint32_t impl_prefetcher_cache_fill(champsim::address addr, long set, long way, bool prefetch, champsim::address evicted_addr,
                                                    uint32_t metadata_in) const;
  void impl_prefetcher_cycle_operate() const;
  void impl_prefetcher_final_stats() const;
  void impl_prefetcher_branch_operate(champsim::address ip, uint8_t branch_type, champsim::address branch_target) const;

  void impl_initialize_replacement() const;
  [[nodiscard]] long impl_find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const BLOCK* current_set, champsim::address ip,
                                      champsim::address full_addr, access_type type) const;
  void impl_update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                     champsim::address victim_addr, access_type type, bool hit) const;
  void impl_replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                   champsim::address victim_addr, access_type type) const;
  void impl_replacement_final_stats() const;
  // NOLINTEND(readability-make-member-function-const)

  template <typename... Ps, typename... Rs>
  explicit CACHE(champsim::cache_builder<champsim::cache_builder_module_type_holder<Ps...>, champsim::cache_builder_module_type_holder<Rs...>> b)
      : champsim::operable(b.m_clock_period), reserved_metadata_ways(b.m_metadata_ways),
        metadata_capacity_ways(b.m_metadata_max_ways == 0 ? b.m_metadata_ways : b.m_metadata_max_ways), sms_degree(b.m_triage_sms_degree),
        upper_levels(b.m_uls), lower_level(b.m_ll),
        lower_translate(b.m_lt), NAME(b.m_name), NUM_SET(b.get_num_sets()),
        NUM_WAY(b.get_num_ways()), MSHR_SIZE(b.get_num_mshrs()), PQ_SIZE(b.m_pq_size), HIT_LATENCY(b.get_hit_latency() * b.m_clock_period),
        FILL_LATENCY(b.get_fill_latency() * b.m_clock_period), OFFSET_BITS(b.m_offset_bits), MAX_TAG(b.get_tag_bandwidth()), MAX_FILL(b.get_fill_bandwidth()),
        prefetch_as_load(b.m_pref_load), match_offset_bits(b.m_wq_full_addr), virtual_prefetch(b.m_va_pref), pref_activate_mask(b.m_pref_act_mask),
        pref_module_pimpl(std::make_unique<prefetcher_module_model<Ps...>>(this)), repl_module_pimpl(std::make_unique<replacement_module_model<Rs...>>(this))
  {
    if (b.m_metadata_partition_epoch != 0 && b.m_metadata_max_ways == 0)
      throw std::invalid_argument{"A metadata partition epoch requires metadata_max_ways"};
    if (metadata_capacity_ways != 0) {
      if (reserved_metadata_ways > metadata_capacity_ways || metadata_capacity_ways >= NUM_WAY || OFFSET_BITS != champsim::data::bits{6})
        throw std::invalid_argument{"Metadata partition requires initial ways <= maximum ways, at least one data way, and 64-byte cache lines"};
      if (NUM_SET == 0 || (NUM_SET & (NUM_SET - 1)) != 0)
        throw std::invalid_argument{"Metadata partition requires a positive power-of-two set count"};
      if constexpr (!std::is_same_v<champsim::cache_builder_module_type_holder<Rs...>, champsim::cache_builder_module_type_holder<lru>>)
        throw std::invalid_argument{"Metadata partition requires the lru replacement module"};
      metadata_table.resize(static_cast<std::size_t>(NUM_SET) * metadata_capacity_ways);
      if (b.m_metadata_max_ways != 0)
        partition = metadata_partition{std::vector<uint32_t>(NUM_SET, reserved_metadata_ways), reserved_metadata_ways, b.m_metadata_partition_epoch};
    }
  }

  CACHE(const CACHE&) = delete;
  CACHE(CACHE&&);
  CACHE& operator=(const CACHE&) = delete;
  CACHE& operator=(CACHE&&);
};

template <typename... Ps>
void CACHE::prefetcher_module_model<Ps...>::impl_prefetcher_initialize()
{
  [[maybe_unused]] auto process_one = [&](auto& p) {
    using namespace champsim::modules;
    if constexpr (prefetcher::has_initialize<decltype(p)>)
      p.prefetcher_initialize();
  };

  std::apply([&](auto&... p) { (..., process_one(p)); }, intern_);
}

template <typename... Ps>
uint32_t CACHE::prefetcher_module_model<Ps...>::impl_prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit,
                                                                              bool useful_prefetch, access_type type, uint32_t metadata_in)
{
  using return_type = uint32_t;
  [[maybe_unused]] auto process_one = [&](auto& p) {
    using namespace champsim::modules;
    /* Strong addresses */
    if constexpr (prefetcher::has_cache_operate<decltype(p), champsim::address, champsim::address, bool, bool, access_type, uint32_t>)
      return return_type{p.prefetcher_cache_operate(addr, ip, cache_hit, useful_prefetch, type, metadata_in)};

    /* Strong addresses, raw integer access type */
    if constexpr (prefetcher::has_cache_operate<decltype(p), champsim::address, champsim::address, bool, bool, std::underlying_type_t<access_type>, uint32_t>)
      return return_type{p.prefetcher_cache_operate(addr, ip, cache_hit, useful_prefetch, champsim::to_underlying(type), metadata_in)};

    /* Raw integer addresses, no useful_prefetch parameter, raw integer access type */
    if constexpr (prefetcher::has_cache_operate<decltype(p), uint64_t, uint64_t, bool, std::underlying_type_t<access_type>, uint32_t>)
      return return_type{p.prefetcher_cache_operate(addr.to<uint64_t>(), ip.to<uint64_t>(), cache_hit, champsim::to_underlying(type), metadata_in)};

    return return_type{};
  };

  return std::apply([&](auto&... p) { return (return_type{} ^ ... ^ process_one(p)); }, intern_);
}

template <typename... Ps>
uint32_t CACHE::prefetcher_module_model<Ps...>::impl_prefetcher_cache_fill(champsim::address addr, long set, long way, bool prefetch,
                                                                           champsim::address evicted_addr, uint32_t metadata_in)
{
  using return_type = uint32_t;
  [[maybe_unused]] auto process_one = [&](auto& p) {
    using namespace champsim::modules;
    if constexpr (prefetcher::has_cache_fill<decltype(p), champsim::address, long, long, bool, champsim::address, uint32_t>)
      return return_type{p.prefetcher_cache_fill(addr, set, way, prefetch, evicted_addr, metadata_in)};
    if constexpr (prefetcher::has_cache_fill<decltype(p), uint64_t, long, long, bool, uint64_t, uint32_t>)
      return return_type{p.prefetcher_cache_fill(addr.to<uint64_t>(), set, way, prefetch, evicted_addr.to<uint64_t>(), metadata_in)};
    return return_type{};
  };

  return std::apply([&](auto&... p) { return (return_type{} ^ ... ^ process_one(p)); }, intern_);
}

template <typename... Ps>
void CACHE::prefetcher_module_model<Ps...>::impl_prefetcher_cycle_operate()
{
  [[maybe_unused]] auto process_one = [&](auto& p) {
    using namespace champsim::modules;
    if constexpr (prefetcher::has_cycle_operate<decltype(p)>)
      p.prefetcher_cycle_operate();
  };

  std::apply([&](auto&... p) { (..., process_one(p)); }, intern_);
}

template <typename... Ps>
void CACHE::prefetcher_module_model<Ps...>::impl_prefetcher_final_stats()
{
  [[maybe_unused]] auto process_one = [&](auto& p) {
    using namespace champsim::modules;
    if constexpr (prefetcher::has_final_stats<decltype(p)>)
      p.prefetcher_final_stats();
  };

  std::apply([&](auto&... p) { (..., process_one(p)); }, intern_);
}

template <typename... Ps>
void CACHE::prefetcher_module_model<Ps...>::impl_prefetcher_branch_operate(champsim::address ip, uint8_t branch_type, champsim::address branch_target)
{
  [[maybe_unused]] auto process_one = [&](auto& p) {
    using namespace champsim::modules;
    if constexpr (prefetcher::has_branch_operate<decltype(p), champsim::address, uint8_t, champsim::address>)
      p.prefetcher_branch_operate(ip, branch_type, branch_target);
    if constexpr (prefetcher::has_branch_operate<decltype(p), uint64_t, uint8_t, uint64_t>)
      p.prefetcher_branch_operate(ip.to<uint64_t>(), branch_type, branch_target.to<uint64_t>());
  };

  std::apply([&](auto&... p) { (..., process_one(p)); }, intern_);
}

template <typename... Rs>
void CACHE::replacement_module_model<Rs...>::impl_initialize_replacement()
{
  [[maybe_unused]] auto process_one = [&](auto& r) {
    using namespace champsim::modules;
    if constexpr (replacement::has_initialize<decltype(r)>)
      r.initialize_replacement();
  };

  std::apply([&](auto&... r) { (..., process_one(r)); }, intern_);
}

template <typename... Rs>
long CACHE::replacement_module_model<Rs...>::impl_find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const BLOCK* current_set,
                                                              champsim::address ip, champsim::address full_addr, access_type type)
{
  using return_type = long;
  [[maybe_unused]] auto process_one = [&](auto& r) {
    using namespace champsim::modules;

    /* Strong addresses */
    if constexpr (replacement::has_find_victim<decltype(r), uint32_t, uint64_t, long, const BLOCK*, champsim::address, champsim::address, access_type>)
      return return_type{r.find_victim(triggering_cpu, instr_id, set, current_set, ip, full_addr, type)};

    /* Raw integer addresses */
    if constexpr (replacement::has_find_victim<decltype(r), uint32_t, uint64_t, long, const BLOCK*, champsim::address, champsim::address,
                                               std::underlying_type_t<access_type>>)
      return return_type{r.find_victim(triggering_cpu, instr_id, set, current_set, ip, full_addr, champsim::to_underlying(type))};

    /* Raw integer addresses, raw integer access type */
    if constexpr (replacement::has_find_victim<decltype(r), uint32_t, uint64_t, long, const BLOCK*, uint64_t, uint64_t, std::underlying_type_t<access_type>>)
      return return_type{r.find_victim(triggering_cpu, instr_id, set, current_set, ip.to<uint64_t>(), full_addr.to<uint64_t>(), champsim::to_underlying(type))};

    return return_type{};
  };

  if constexpr (sizeof...(Rs) > 0) {
    return std::apply([&](auto&... r) { return (..., process_one(r)); }, intern_);
  }
  return return_type{};
}

template <typename... Rs>
void CACHE::replacement_module_model<Rs...>::impl_update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr,
                                                                           champsim::address ip, champsim::address victim_addr, access_type type, bool hit)
{
  [[maybe_unused]] auto process_one = [&](auto& r) {
    using namespace champsim::modules;

    if (hit || replacement::has_cache_fill<decltype(r), uint32_t, long, long, champsim::address, champsim::address, champsim::address, access_type>) {
      auto new_victim_addr = hit ? champsim::address{} : victim_addr;

      /* Strong addresses */
      if constexpr (replacement::has_update_state<decltype(r), uint32_t, long, long, champsim::address, champsim::address, access_type, bool>)
        r.update_replacement_state(triggering_cpu, set, way, full_addr, ip, type, hit);

      /* Strong addresses */
      else if constexpr (replacement::has_update_state<decltype(r), uint32_t, long, long, champsim::address, champsim::address, champsim::address, access_type,
                                                       bool>)
        r.update_replacement_state(triggering_cpu, set, way, full_addr, ip, new_victim_addr, type, hit);

      /* Raw integer access type */
      else if constexpr (replacement::has_update_state<decltype(r), uint32_t, long, long, champsim::address, champsim::address, champsim::address,
                                                       std::underlying_type_t<access_type>, bool>)
        r.update_replacement_state(triggering_cpu, set, way, full_addr, ip, new_victim_addr, champsim::to_underlying(type), hit);

      /* Raw integer addresses, raw integer access type */
      else if constexpr (replacement::has_update_state<decltype(r), uint32_t, long, long, uint64_t, uint64_t, uint64_t, std::underlying_type_t<access_type>,
                                                       bool>)
        r.update_replacement_state(triggering_cpu, set, way, full_addr.to<uint64_t>(), ip.to<uint64_t>(), new_victim_addr.to<uint64_t>(),
                                   champsim::to_underlying(type), hit);
    }
  };

  std::apply([&](auto&... r) { (..., process_one(r)); }, intern_);
}

template <typename... Rs>
void CACHE::replacement_module_model<Rs...>::impl_replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr,
                                                                         champsim::address ip, champsim::address victim_addr, access_type type)
{
  [[maybe_unused]] auto process_one = [&](auto& r) {
    using namespace champsim::modules;

    /* Strong addresses */
    if constexpr (replacement::has_cache_fill<decltype(r), uint32_t, long, long, champsim::address, champsim::address, champsim::address, access_type>)
      r.replacement_cache_fill(triggering_cpu, set, way, full_addr, ip, victim_addr, type);

    else
      impl_update_replacement_state(triggering_cpu, set, way, full_addr, ip, victim_addr, type, false);
  };

  std::apply([&](auto&... r) { (..., process_one(r)); }, intern_);
}

template <typename... Rs>
void CACHE::replacement_module_model<Rs...>::impl_replacement_final_stats()
{
  [[maybe_unused]] auto process_one = [&](auto& r) {
    using namespace champsim::modules;
    if constexpr (replacement::has_final_stats<decltype(r)>)
      r.replacement_final_stats();
  };

  std::apply([&](auto&... r) { (..., process_one(r)); }, intern_);
}

#ifdef SET_ASIDE_CHAMPSIM_MODULE
#undef SET_ASIDE_CHAMPSIM_MODULE
#define CHAMPSIM_MODULE
#endif

#endif
