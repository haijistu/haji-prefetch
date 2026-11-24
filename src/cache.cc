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

#include "cache.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <utility>
#include <fmt/core.h>

#include "bandwidth.h"
#include "champsim.h"
#include "chrono.h"
#include "deadlock.h"
#include "instruction.h"
#include "util/algorithm.h"
#include "util/bits.h"
#include "util/span.h"

CACHE::CACHE(CACHE&& other)
    : operable(other),
      reserved_metadata_ways(other.reserved_metadata_ways), metadata_capacity_ways(other.metadata_capacity_ways), partition(std::move(other.partition)),
      metadata_table(std::move(other.metadata_table)), metadata_clock(other.metadata_clock), metadata_set_revisions(std::move(other.metadata_set_revisions)),
      lower_cache(other.lower_cache), sms_degree(other.sms_degree),

      upper_levels(std::move(other.upper_levels)), lower_level(std::move(other.lower_level)), lower_translate(std::move(other.lower_translate)),

      cpu(other.cpu), NAME(std::move(other.NAME)), NUM_SET(other.NUM_SET), NUM_WAY(other.NUM_WAY), MSHR_SIZE(other.MSHR_SIZE), PQ_SIZE(other.PQ_SIZE),
      HIT_LATENCY(other.HIT_LATENCY), FILL_LATENCY(other.FILL_LATENCY), OFFSET_BITS(other.OFFSET_BITS), block(std::move(other.block)), MAX_TAG(other.MAX_TAG),
      MAX_FILL(other.MAX_FILL), prefetch_as_load(other.prefetch_as_load), match_offset_bits(other.match_offset_bits), virtual_prefetch(other.virtual_prefetch),
      pref_activate_mask(std::move(other.pref_activate_mask)),

      sim_stats(std::move(other.sim_stats)), roi_stats(std::move(other.roi_stats)),

      pref_module_pimpl(std::move(other.pref_module_pimpl)), repl_module_pimpl(std::move(other.repl_module_pimpl))
{
  pref_module_pimpl->bind(this);
  repl_module_pimpl->bind(this);
}

auto CACHE::operator=(CACHE&& other) -> CACHE&
{
  this->clock_period = other.clock_period;
  this->current_time = other.current_time;
  this->warmup = other.warmup;

  this->reserved_metadata_ways = other.reserved_metadata_ways;
  this->metadata_capacity_ways = other.metadata_capacity_ways;
  this->partition = std::move(other.partition);
  this->metadata_table = std::move(other.metadata_table);
  this->metadata_clock = other.metadata_clock;
  this->metadata_set_revisions = std::move(other.metadata_set_revisions);
  this->lower_cache = other.lower_cache;
  this->sms_degree = other.sms_degree;

  this->upper_levels = std::move(other.upper_levels);
  this->lower_level = std::move(other.lower_level);
  this->lower_translate = std::move(other.lower_translate);

  this->cpu = other.cpu;
  this->NAME = std::move(other.NAME);
  this->NUM_SET = other.NUM_SET;
  this->NUM_WAY = other.NUM_WAY;
  ;
  this->MSHR_SIZE = other.MSHR_SIZE;
  ;
  this->PQ_SIZE = other.PQ_SIZE;
  this->HIT_LATENCY = other.HIT_LATENCY;
  this->FILL_LATENCY = other.FILL_LATENCY;
  this->OFFSET_BITS = other.OFFSET_BITS;
  ;
  this->block = std::move(other.block);
  this->MAX_TAG = other.MAX_TAG;
  this->MAX_FILL = other.MAX_FILL;
  this->prefetch_as_load = other.prefetch_as_load;
  this->match_offset_bits = other.match_offset_bits;
  this->virtual_prefetch = other.virtual_prefetch;
  this->pref_activate_mask = std::move(other.pref_activate_mask);

  this->sim_stats = std::move(other.sim_stats);
  this->roi_stats = std::move(other.roi_stats);

  this->pref_module_pimpl = std::move(other.pref_module_pimpl);
  this->repl_module_pimpl = std::move(other.repl_module_pimpl);

  pref_module_pimpl->bind(this);
  repl_module_pimpl->bind(this);

  return *this;
}

CACHE::tag_lookup_type::tag_lookup_type(const request_type& req, bool local_pref, bool skip)
    : address(req.address), v_address(req.v_address), data(req.data), ip(req.ip), instr_id(req.instr_id), pf_metadata(req.pf_metadata), cpu(req.cpu),
      type(req.type), prefetch_from_this(local_pref), skip_fill(skip), is_translated(req.is_translated), is_instruction(req.is_instruction),
      instr_depend_on_me(req.instr_depend_on_me)
{
}

CACHE::fill_type::fill_type(const tag_lookup_type& req, champsim::chrono::clock::time_point _time_enqueued)
    : address(req.address), v_address(req.v_address), ip(req.ip), instr_id(req.instr_id), cpu(req.cpu), type(req.type),
      prefetch_from_this(req.prefetch_from_this), prefetch_source(req.prefetch_source), time_enqueued(_time_enqueued),
      instr_depend_on_me(req.instr_depend_on_me), to_return(req.to_return)
{
}

CACHE::fill_type CACHE::fill_type::merge(fill_type predecessor, fill_type successor)
{
  std::vector<uint64_t> merged_instr{};
  std::vector<std::deque<response_type>*> merged_return{};

  std::set_union(std::begin(predecessor.instr_depend_on_me), std::end(predecessor.instr_depend_on_me), std::begin(successor.instr_depend_on_me),
                 std::end(successor.instr_depend_on_me), std::back_inserter(merged_instr));
  std::set_union(std::begin(predecessor.to_return), std::end(predecessor.to_return), std::begin(successor.to_return), std::end(successor.to_return),
                 std::back_inserter(merged_return));

  fill_type retval{(successor.type == access_type::PREFETCH) ? predecessor : successor};

  // set the time enqueued to the predecessor unless its a demand into prefetch, in which case we use the successor
  retval.time_enqueued =
      ((successor.type != access_type::PREFETCH && predecessor.type == access_type::PREFETCH)) ? successor.time_enqueued : predecessor.time_enqueued;
  retval.instr_depend_on_me = merged_instr;
  retval.to_return = merged_return;
  retval.data_promise = predecessor.data_promise;

  if constexpr (champsim::debug_print) {
    if (successor.type == access_type::PREFETCH) {
      fmt::print("[MSHR] {} address {} type: {} into address {} type: {}\n", __func__, successor.address,
                 access_type_names.at(champsim::to_underlying(successor.type)), predecessor.address,
                 access_type_names.at(champsim::to_underlying(successor.type)));
    } else {
      fmt::print("[MSHR] {} address {} type: {} into address {} type: {}\n", __func__, predecessor.address,
                 access_type_names.at(champsim::to_underlying(predecessor.type)), successor.address,
                 access_type_names.at(champsim::to_underlying(successor.type)));
    }
  }

  return retval;
}

auto CACHE::fill_block(fill_type fill, uint32_t metadata) -> BLOCK
{
  CACHE::BLOCK to_fill;
  to_fill.valid = true;
  to_fill.prefetch = fill.prefetch_from_this;
  to_fill.dirty = (fill.type == access_type::WRITE);
  to_fill.address = fill.address;
  to_fill.v_address = fill.v_address;
  to_fill.data = fill.data_promise->data;
  to_fill.pf_metadata = metadata;
  to_fill.cpu = fill.cpu;
  to_fill.prefetch_source = fill.prefetch_source;

  return to_fill;
}

auto CACHE::matches_address(champsim::address addr) const
{
  return [match = addr.slice_upper(OFFSET_BITS), shamt = OFFSET_BITS](const auto& entry) {
    return entry.address.slice_upper(shamt) == match;
  };
}

template <typename T>
champsim::address CACHE::module_address(const T& element) const
{
  auto address = virtual_prefetch ? element.v_address : element.address;
  return champsim::address{address.slice_upper(match_offset_bits ? champsim::data::bits{} : OFFSET_BITS)};
}

bool CACHE::enqueue_writeback(const BLOCK& victim, uint32_t triggering_cpu, uint64_t instr_id)
{
  request_type packet;
  packet.cpu = triggering_cpu;
  packet.address = victim.address;
  packet.data = victim.data;
  packet.instr_id = instr_id;
  packet.type = access_type::WRITE;
  packet.pf_metadata = victim.pf_metadata;
  packet.response_requested = false;
  return lower_level->add_wq(packet);
}

bool CACHE::handle_fill(const fill_type& fill)
{
  cpu = fill.cpu;

  // find victim
  auto [set_begin, set_end] = get_set_span(fill.address);
  auto way = std::find_if_not(set_begin, set_end, [](auto x) { return x.valid; });
  if (way == set_end) {
    const auto victim = impl_find_victim(fill.cpu, fill.instr_id, get_set_index(fill.address), &*set_begin, fill.ip, fill.address, fill.type);
    assert(victim == NUM_WAY || (victim >= 0 && victim < std::distance(set_begin, set_end)));
    if (victim != NUM_WAY)
      way = std::next(set_begin, victim);
  }
  assert(set_begin <= way);
  assert(way <= set_end);
  assert(way != set_end || fill.type != access_type::WRITE); // Writes may not bypass
  const auto way_idx = way == set_end ? static_cast<long>(NUM_WAY) : std::distance(set_begin, way);

  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} instr_id: {} address: {} v_address: {} set: {} way: {} type: {} prefetch_metadata: {} cycle_enqueued: {} cycle: {}\n", NAME, __func__,
               fill.instr_id, fill.address, fill.v_address, get_set_index(fill.address), way_idx, access_type_names.at(champsim::to_underlying(fill.type)),
               fill.data_promise->pf_metadata, (fill.time_enqueued.time_since_epoch()) / clock_period, (current_time.time_since_epoch()) / clock_period);
  }

  if (way != set_end && way->valid && way->dirty) {
    if constexpr (champsim::debug_print) {
      fmt::print("[{}] {} evict address: {} v_address: {} prefetch_metadata: {}\n", NAME, __func__, way->address, way->v_address, way->pf_metadata);
    }
    if (!enqueue_writeback(*way, fill.cpu, fill.instr_id)) {
      return false;
    }
  }

  champsim::address evicting_address{};
  if (way != set_end && way->valid) {
    evicting_address = module_address(*way);
  }

  auto metadata_thru = impl_prefetcher_cache_fill(module_address(fill), get_set_index(fill.address), way_idx, (fill.type == access_type::PREFETCH),
                                                  evicting_address, fill.data_promise->pf_metadata);
  impl_replacement_cache_fill(fill.cpu, get_set_index(fill.address), way_idx, module_address(fill), fill.ip, evicting_address, fill.type);

  if (way != set_end) {
    if (way->valid)
      pref_module_pimpl->impl_prefetcher_on_evict(way->address);
    if (way->valid && way->prefetch) {
      ++sim_stats.pf_useless;
    }

    if (fill.type == access_type::PREFETCH) {
      ++sim_stats.pf_fill;
    }

    *way = fill_block(fill, metadata_thru);
  }

  // COLLECT STATS
  if (fill.type != access_type::PREFETCH)
    sim_stats.total_miss_latency_cycles += (current_time - (fill.time_enqueued + clock_period)) / clock_period;
  sim_stats.fill.increment(std::pair{fill.type, fill.cpu});

  response_type response{fill.address, fill.v_address, fill.data_promise->data, metadata_thru, fill.instr_depend_on_me};
  for (auto* ret : fill.to_return) {
    ret->push_back(response);
  }

  return true;
}

bool CACHE::try_hit(const tag_lookup_type& handle_pkt)
{
  cpu = handle_pkt.cpu;

  // access cache
  auto [set_begin, set_end] = get_set_span(handle_pkt.address);
  auto way = std::find_if(set_begin, set_end, [matcher = matches_address(handle_pkt.address)](const auto& x) { return x.valid && matcher(x); });
  const auto hit = (way != set_end);
  const auto useful_prefetch = (hit && way->prefetch && !handle_pkt.prefetch_from_this);

  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} instr_id: {} address: {} v_address: {} data: {} set: {} way: {} ({}) type: {} cycle: {}\n", NAME, __func__, handle_pkt.instr_id,
               handle_pkt.address, handle_pkt.v_address, handle_pkt.data, get_set_index(handle_pkt.address), std::distance(set_begin, way),
               hit ? "HIT" : "MISS", access_type_names.at(champsim::to_underlying(handle_pkt.type)), current_time.time_since_epoch() / clock_period);
  }

  auto metadata_thru = handle_pkt.pf_metadata;
  if (should_activate_prefetcher(handle_pkt)) {
    metadata_thru = impl_prefetcher_cache_operate(module_address(handle_pkt), handle_pkt.ip, hit, useful_prefetch, handle_pkt.type, metadata_thru);
  }

  // update replacement policy
  const auto way_idx = hit ? std::distance(set_begin, way) : static_cast<long>(NUM_WAY);
  impl_update_replacement_state(handle_pkt.cpu, get_set_index(handle_pkt.address), way_idx, module_address(handle_pkt), handle_pkt.ip, {}, handle_pkt.type,
                                hit);

  if (hit) {
    notify_prefetcher_demand_load(handle_pkt, true, useful_prefetch, way->pf_metadata);
    sim_stats.hits.increment(std::pair{handle_pkt.type, handle_pkt.cpu});

    response_type response{handle_pkt.address, handle_pkt.v_address, way->data, metadata_thru, handle_pkt.instr_depend_on_me};
    for (auto* ret : handle_pkt.to_return) {
      ret->push_back(response);
    }

    if (handle_pkt.type == access_type::WRITE) {
      way->dirty = true;
      way->cpu = handle_pkt.cpu;
      if (!match_offset_bits) {
        way->data = handle_pkt.data;
        way->pf_metadata = handle_pkt.pf_metadata;
      }
    }
    if (handle_pkt.type == access_type::LOAD || handle_pkt.type == access_type::RFO)
      observe_partition_access(handle_pkt.address, true);

    // update prefetch stats and reset prefetch bit
    if (useful_prefetch) {
      ++sim_stats.pf_useful;
      if (way->prefetch_source != champsim::triage_prefetch_source::none)
        ++sim_stats.triage_prefetch.source(way->prefetch_source).timely_hits;
      way->prefetch = false;
    }
  }

  return hit;
}

void CACHE::notify_prefetcher_demand_load(const tag_lookup_type& packet, bool hit, bool useful_prefetch, uint32_t fill_metadata)
{
  if (packet.type == access_type::LOAD && !packet.is_instruction)
    pref_module_pimpl->impl_prefetcher_on_demand_load(packet.address, packet.ip, hit, useful_prefetch, fill_metadata);
}

auto CACHE::mshr_and_forward_packet(const tag_lookup_type& handle_pkt) -> std::pair<fill_type, request_type>
{
  fill_type to_allocate{handle_pkt, current_time};

  request_type fwd_pkt;

  fwd_pkt.asid[0] = handle_pkt.asid[0];
  fwd_pkt.asid[1] = handle_pkt.asid[1];
  fwd_pkt.type = (handle_pkt.type == access_type::WRITE) ? access_type::RFO : handle_pkt.type;
  fwd_pkt.pf_metadata = handle_pkt.pf_metadata;
  fwd_pkt.cpu = handle_pkt.cpu;

  fwd_pkt.address = handle_pkt.address;
  fwd_pkt.v_address = handle_pkt.v_address;
  fwd_pkt.data = handle_pkt.data;
  fwd_pkt.instr_id = handle_pkt.instr_id;
  fwd_pkt.ip = handle_pkt.ip;
  fwd_pkt.is_instruction = handle_pkt.is_instruction;

  fwd_pkt.instr_depend_on_me = handle_pkt.instr_depend_on_me;
  fwd_pkt.response_requested = (!handle_pkt.prefetch_from_this || !handle_pkt.skip_fill);

  return std::pair{std::move(to_allocate), std::move(fwd_pkt)};
}

bool CACHE::handle_miss(const tag_lookup_type& handle_pkt)
{
  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} instr_id: {} address: {} v_address: {} type: {} local_prefetch: {} cycle: {}\n", NAME, __func__, handle_pkt.instr_id,
               handle_pkt.address, handle_pkt.v_address, access_type_names.at(champsim::to_underlying(handle_pkt.type)), handle_pkt.prefetch_from_this,
               current_time.time_since_epoch() / clock_period);
  }

  fill_type to_allocate{handle_pkt, current_time};

  cpu = handle_pkt.cpu;

  auto mshr_pkt = mshr_and_forward_packet(handle_pkt);

  // check mshr
  auto fill_entry = std::find_if(std::begin(MSHR), std::end(MSHR), matches_address(handle_pkt.address));
  bool mshr_full = (MSHR.size() == MSHR_SIZE);

  // check inflight fills
  if (fill_entry == MSHR.end()) {
    fill_entry = std::find_if(inflight_fills.begin(), inflight_fills.end(), matches_address(handle_pkt.address));
  }

  if (fill_entry != inflight_fills.end()) // miss or fill already inflight
  {
    if (fill_entry->type == access_type::PREFETCH && handle_pkt.type != access_type::PREFETCH) {
      // Mark the prefetch as useful
      if (fill_entry->prefetch_from_this) {
        ++sim_stats.pf_useful;
        if (fill_entry->prefetch_source != champsim::triage_prefetch_source::none)
          ++sim_stats.triage_prefetch.source(fill_entry->prefetch_source).late_merges;
      }
    }

    // COLLECT STATS
    sim_stats.miss_merge.increment(std::pair{to_allocate.type, to_allocate.cpu});

    *fill_entry = fill_type::merge(*fill_entry, to_allocate);
  } else {
    if (mshr_full) { // not enough MSHR resource
      return false;  // TODO should we allow prefetches anyway if they will not be filled to this level?
    }

    // Do not let a refetch overtake a dirty line retired into the link write queue.
    if (partition && std::any_of(lower_level->WQ.begin(), lower_level->WQ.end(), matches_address(handle_pkt.address)))
      return false;

    const bool send_to_rq = (prefetch_as_load || handle_pkt.type != access_type::PREFETCH);
    bool success = send_to_rq ? lower_level->add_rq(mshr_pkt.second) : lower_level->add_pq(mshr_pkt.second);

    if (!success) {
      return false;
    }

    // Allocate an MSHR
    if (mshr_pkt.second.response_requested) {
      MSHR.emplace_back(std::move(mshr_pkt.first));
    }
  }

  sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});
  notify_prefetcher_demand_load(handle_pkt, false, false);
  if (handle_pkt.type == access_type::LOAD || handle_pkt.type == access_type::RFO)
    observe_partition_access(handle_pkt.address, false);

  return true;
}

bool CACHE::handle_write(const tag_lookup_type& handle_pkt)
{
  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} instr_id: {} address: {} v_address: {} type: {} local_prefetch: {} cycle: {}\n", NAME, __func__, handle_pkt.instr_id,
               handle_pkt.address, handle_pkt.v_address, access_type_names.at(champsim::to_underlying(handle_pkt.type)), handle_pkt.prefetch_from_this,
               current_time.time_since_epoch() / clock_period);
  }

  fill_type to_allocate{handle_pkt, current_time};
  to_allocate.data_promise = champsim::waitable{fill_type::returned_value{handle_pkt.data, handle_pkt.pf_metadata},
                                               current_time + (warmup ? champsim::chrono::clock::duration{} : FILL_LATENCY)};
  inflight_fills.push_back(to_allocate);

  sim_stats.misses.increment(std::pair{handle_pkt.type, handle_pkt.cpu});

  return true;
}

template <bool UpdateRequest>
auto CACHE::initiate_tag_check(champsim::channel* ul)
{
  return [time = current_time + (warmup ? champsim::chrono::clock::duration{} : HIT_LATENCY), ul](const auto& entry) {
    CACHE::tag_lookup_type retval{entry};
    retval.event_cycle = time;

    if constexpr (UpdateRequest) {
      if (entry.response_requested) {
        retval.to_return = {&ul->returned};
      }
    } else {
      (void)ul; // supress warning about ul being unused
    }

    if constexpr (champsim::debug_print) {
      fmt::print("[TAG] initiate_tag_check instr_id: {} address: {} v_address: {} type: {} response_requested: {}\n", retval.instr_id, retval.address,
                 retval.v_address, access_type_names.at(champsim::to_underlying(retval.type)), !std::empty(retval.to_return));
    }

    return retval;
  };
}

long CACHE::operate()
{
  long progress{0};

  auto is_ready = [time = current_time](const auto& entry) {
    return entry.event_cycle <= time;
  };
  auto is_translated = [](const auto& entry) {
    return entry.is_translated;
  };

  // Finish returns
  std::for_each(std::cbegin(lower_level->returned), std::cend(lower_level->returned), [this](const auto& pkt) { this->finish_packet(pkt); });
  progress += std::distance(std::cbegin(lower_level->returned), std::cend(lower_level->returned));
  lower_level->returned.clear();

  // Finish translations
  if (lower_translate != nullptr) {
    std::for_each(std::cbegin(lower_translate->returned), std::cend(lower_translate->returned), [this](const auto& pkt) { this->finish_translation(pkt); });
    progress += std::distance(std::cbegin(lower_translate->returned), std::cend(lower_translate->returned));
    lower_translate->returned.clear();
  }

  progress += advance_metadata_partition();

  // Perform fills
  champsim::bandwidth fill_bw{MAX_FILL};
  auto [fill_begin, fill_end] = champsim::get_span_p(std::cbegin(inflight_fills), std::cend(inflight_fills), fill_bw,
                                                     [time = current_time](const auto& x) { return x.data_promise.is_ready_at(time); });
  auto complete_end = std::find_if_not(fill_begin, fill_end, [this](const auto& x) { return this->handle_fill(x); });
  fill_bw.consume(std::distance(fill_begin, complete_end));
  inflight_fills.erase(fill_begin, complete_end);

  // Initiate tag checks
  const champsim::bandwidth::maximum_type bandwidth_from_tag_checks{champsim::to_underlying(MAX_TAG) * (long)(HIT_LATENCY / clock_period)
                                                                    - (long)std::size(inflight_tag_check)};
  champsim::bandwidth initiate_tag_bw{std::clamp(bandwidth_from_tag_checks, champsim::bandwidth::maximum_type{0}, MAX_TAG)};
  auto can_translate = [avail = (std::size(translation_stash) < static_cast<std::size_t>(MSHR_SIZE))](const auto& entry) {
    return avail || entry.is_translated;
  };
  auto stash_bandwidth_consumed =
      champsim::transform_while_n(translation_stash, std::back_inserter(inflight_tag_check), initiate_tag_bw, is_translated, initiate_tag_check<false>());
  initiate_tag_bw.consume(stash_bandwidth_consumed);
  std::vector<long long> channels_bandwidth_consumed{};

  if (std::size(upper_levels) > 1) {
    std::rotate(upper_levels.begin(), upper_levels.begin() + 1, upper_levels.end());
  }

  // upper levels get an equal portion of the remaining bandwidth
  champsim::bandwidth::maximum_type per_upper_bandwidth =
      std::size(upper_levels) >= 1
          ? (champsim::bandwidth::maximum_type)std::max((size_t)initiate_tag_bw.amount_remaining() / std::size(upper_levels), size_t{1})
          : champsim::bandwidth::maximum_type{};

  for (auto* ul : upper_levels) {
    for (auto q : {std::ref(ul->WQ), std::ref(ul->RQ), std::ref(ul->PQ)}) {
      // this needs to be in this loop, we need to ensure that for cases where bandwidth doesn't divide nicely across upstreams,
      // we don't accidentally consume more bandwidth than expected
      champsim::bandwidth per_upper_tag_bw{std::min(per_upper_bandwidth, champsim::bandwidth::maximum_type{initiate_tag_bw.amount_remaining()})};
      auto bandwidth_consumed =
          champsim::transform_while_n(q.get(), std::back_inserter(inflight_tag_check), per_upper_tag_bw, can_translate, initiate_tag_check<true>(ul));
      channels_bandwidth_consumed.push_back(bandwidth_consumed);
      initiate_tag_bw.consume(bandwidth_consumed);
    }
  }

  auto pq_bandwidth_consumed =
      champsim::transform_while_n(internal_PQ, std::back_inserter(inflight_tag_check), initiate_tag_bw, can_translate, initiate_tag_check<false>());
  initiate_tag_bw.consume(pq_bandwidth_consumed);

  // Issue translations
  std::for_each(std::begin(inflight_tag_check), std::end(inflight_tag_check), [this](auto& x) { this->issue_translation(x); });
  std::for_each(std::begin(translation_stash), std::end(translation_stash), [this](auto& x) { this->issue_translation(x); });

  // Find entries that would be ready except that they have not finished translation, move them to the stash
  auto [last_not_missed, stash_end] = champsim::extract_if(std::begin(inflight_tag_check), std::end(inflight_tag_check), std::back_inserter(translation_stash),
                                                           [is_ready, is_translated](const auto& x) { return is_ready(x) && !is_translated(x); });
  progress += std::distance(last_not_missed, std::end(inflight_tag_check));
  inflight_tag_check.erase(last_not_missed, std::end(inflight_tag_check));

  // Perform tag checks
  auto do_handle_miss = [this](const auto& pkt) {
    if (pkt.type == access_type::WRITE && !this->match_offset_bits) {
      return this->handle_write(pkt); // Treat writes (that is, writebacks) like fills
    }
    return this->handle_miss(pkt); // Treat writes (that is, stores) like reads
  };
  champsim::bandwidth tag_check_bw{MAX_TAG};
  auto [tag_check_ready_begin, tag_check_ready_end] =
      champsim::get_span_p(std::begin(inflight_tag_check), std::end(inflight_tag_check), tag_check_bw,
                           [is_ready, is_translated](const auto& pkt) { return is_ready(pkt) && is_translated(pkt); });
  auto hits_end = std::stable_partition(tag_check_ready_begin, tag_check_ready_end, [this](const auto& pkt) { return this->try_hit(pkt); });
  auto finish_tag_check_end = std::stable_partition(hits_end, tag_check_ready_end, do_handle_miss);
  tag_check_bw.consume(std::distance(tag_check_ready_begin, finish_tag_check_end));
  inflight_tag_check.erase(tag_check_ready_begin, finish_tag_check_end);

  impl_prefetcher_cycle_operate();

  if constexpr (champsim::debug_print) {
    fmt::print("[{}] {} cycle completed: {} tags checked: {} remaining: {} stash consumed: {} remaining: {} channel consumed: {} pq consumed {} unused consume "
               "bw {}\n",
               NAME, __func__, current_time.time_since_epoch() / clock_period, tag_check_bw.amount_consumed(), std::size(inflight_tag_check),
               stash_bandwidth_consumed, std::size(translation_stash), channels_bandwidth_consumed, pq_bandwidth_consumed, initiate_tag_bw.amount_remaining());
  }

  return progress + fill_bw.amount_consumed() + initiate_tag_bw.amount_consumed() + tag_check_bw.amount_consumed();
}

// LCOV_EXCL_START exclude deprecated function
uint64_t CACHE::get_set(uint64_t address) const { return static_cast<uint64_t>(get_set_index(champsim::address{address})); }
// LCOV_EXCL_STOP

long CACHE::get_set_index(champsim::address address) const { return address.slice(champsim::dynamic_extent{OFFSET_BITS, champsim::lg2(NUM_SET)}).to<long>(); }

template <typename It>
std::pair<It, It> get_span(It anchor, typename std::iterator_traits<It>::difference_type set_idx, typename std::iterator_traits<It>::difference_type num_way)
{
  auto begin = std::next(anchor, set_idx * num_way);
  return {std::move(begin), std::next(begin, num_way)};
}

auto CACHE::get_set_span(champsim::address address) -> std::pair<set_type::iterator, set_type::iterator>
{
  const auto set_idx = get_set_index(address);
  assert(set_idx < NUM_SET);
  const auto begin = std::next(std::begin(block), static_cast<set_type::difference_type>(set_idx) * NUM_WAY);
  return {begin, std::next(begin, data_ways(static_cast<std::size_t>(set_idx)))};
}

auto CACHE::get_set_span(champsim::address address) const -> std::pair<set_type::const_iterator, set_type::const_iterator>
{
  const auto set_idx = get_set_index(address);
  assert(set_idx < NUM_SET);
  const auto begin = std::next(std::cbegin(block), static_cast<set_type::difference_type>(set_idx) * NUM_WAY);
  return {begin, std::next(begin, data_ways(static_cast<std::size_t>(set_idx)))};
}

uint32_t CACHE::metadata_ways_for_set(std::size_t set) const
{
  return partition ? partition->ways_by_set.at(set) : reserved_metadata_ways;
}

bool CACHE::request_metadata_ways(uint32_t ways)
{
  if (!partition || ways > metadata_capacity_ways || partition->target != reserved_metadata_ways)
    return false;

  partition->target = ways;
  partition->next_set = 0;
  return true;
}

void CACHE::enable_metadata_dueller()
{
  if (metadata_capacity_ways != 0 && metadata_set_revisions.empty())
    metadata_set_revisions.resize(NUM_SET);
  if (!partition || partition->epoch_length == 0 || partition->dueller)
    return;

  auto& dueller = partition->dueller.emplace();
  const auto count = std::min(NUM_SET, 64u);
  dueller.sample_by_set.resize(NUM_SET, -1);
  dueller.samples.resize(count);
  dueller.scores.resize(metadata_capacity_ways + 1);
  std::minstd_rand rng{1};
  for (uint32_t sample = 0; sample < count; ++sample) {
    const auto first = sample * NUM_SET / count;
    const auto last = (sample + 1) * NUM_SET / count;
    const auto set = std::uniform_int_distribution<uint32_t>{first, last - 1}(rng);
    dueller.sample_by_set[set] = static_cast<int>(sample);
    dueller.samples[sample].data.reserve(NUM_WAY);
    dueller.samples[sample].markov.reserve(metadata_capacity_ways * metadata_entries_per_way);
  }
}

uint64_t CACHE::metadata_revision(champsim::block_number trigger) const
{
  return metadata_set_revisions.empty() ? 0 : metadata_set_revisions[trigger.to<uint64_t>() % NUM_SET];
}

void CACHE::observe_reused_metadata(champsim::block_number trigger)
{
  // The shadow partition estimates mapping reuse even when the request bypasses the real LLC metadata.
  observe_dueller_metadata(trigger.to<uint64_t>(), false);
}

bool CACHE::has_demanded_line(champsim::block_number line) const
{
  const champsim::address addr{line};
  const auto [first, last] = get_set_span(addr);
  return std::any_of(first, last, [matches = matches_address(addr)](const auto& entry) { return entry.valid && !entry.prefetch && matches(entry); });
}

bool CACHE::has_pending_or_cached_line(champsim::block_number line) const
{
  const champsim::address addr{line};
  const auto matches = matches_address(addr);
  const auto [first, last] = get_set_span(addr);
  const auto translated_and_matches = [&](const auto& entry) { return entry.is_translated && matches(entry); };
  return std::any_of(first, last, [&](const auto& entry) { return entry.valid && matches(entry); })
         || std::any_of(MSHR.begin(), MSHR.end(), matches) || std::any_of(inflight_fills.begin(), inflight_fills.end(), matches)
         || std::any_of(internal_PQ.begin(), internal_PQ.end(), translated_and_matches)
         || std::any_of(inflight_tag_check.begin(), inflight_tag_check.end(), translated_and_matches)
         || std::any_of(translation_stash.begin(), translation_stash.end(), translated_and_matches);
}

namespace
{
std::optional<std::size_t> access_lru_stack(std::vector<uint64_t>& stack, uint64_t key, std::size_t capacity, bool allocate)
{
  auto it = std::find(stack.begin(), stack.end(), key);
  if (it != stack.end()) {
    const auto rank = static_cast<std::size_t>(std::distance(stack.begin(), it));
    std::rotate(stack.begin(), it, std::next(it));
    return rank;
  }
  if (allocate) {
    if (stack.size() == capacity)
      stack.pop_back();
    stack.insert(stack.begin(), key);
  }
  return std::nullopt;
}
} // namespace

void CACHE::observe_dueller_data(uint64_t line)
{
  auto& dueller = *partition->dueller;
  const auto sample = dueller.sample_by_set[line % NUM_SET];
  if (sample < 0)
    return;

  const auto rank = access_lru_stack(dueller.samples[sample].data, line, NUM_WAY, true);
  if (rank)
    for (std::size_t ways = 0; ways < dueller.scores.size() && *rank < NUM_WAY - ways; ++ways)
      dueller.scores[ways] += 2;
}

void CACHE::observe_dueller_metadata(uint64_t trigger, bool allocate)
{
  if (!partition || !partition->dueller || partition->target != reserved_metadata_ways)
    return;

  auto& dueller = *partition->dueller;
  const auto sample = dueller.sample_by_set[trigger % NUM_SET];
  if (sample < 0)
    return;

  const auto rank = access_lru_stack(dueller.samples[sample].markov, metadata_tag(trigger), metadata_capacity_ways * metadata_entries_per_way, allocate);
  if (rank && !allocate)
    for (auto ways = 1 + *rank / metadata_entries_per_way; ways < dueller.scores.size(); ++ways)
      ++dueller.scores[ways];
}

long CACHE::advance_metadata_partition()
{
  if (!partition || partition->target == reserved_metadata_ways)
    return 0;

  const auto set = partition->next_set;
  auto& metadata_ways = partition->ways_by_set[set];
  auto& stats = sim_stats.partition;
  if (metadata_ways < partition->target) {
    auto& retired = block[static_cast<std::size_t>(set) * NUM_WAY + NUM_WAY - metadata_ways - 1];
    if (retired.valid && retired.dirty) {
      if (!enqueue_writeback(retired, retired.cpu, 0)) {
        ++stats.writeback_stalls;
        return 0;
      }
      ++stats.dirty_writebacks;
    }
    stats.data_evictions += retired.valid;
    sim_stats.pf_useless += retired.valid && retired.prefetch;
    if (retired.valid)
      pref_module_pimpl->impl_prefetcher_on_evict(retired.address);
    retired = {};
    ++metadata_ways;
  } else {
    auto& retired = metadata_table[static_cast<std::size_t>(set) * metadata_capacity_ways + metadata_ways - 1];
    stats.metadata_evictions += static_cast<uint64_t>(std::count_if(retired.last_used.begin(), retired.last_used.end(), [](auto stamp) { return stamp != 0; }));
    retired = {};
    if (!metadata_set_revisions.empty())
      ++metadata_set_revisions[set];
    --metadata_ways;
  }

  ++stats.way_transfers;
  if (metadata_ways == partition->target && ++partition->next_set == NUM_SET) {
    reserved_metadata_ways = partition->target;
    stats.ways_at_end = reserved_metadata_ways;
    ++stats.changes;
  }
  return 1;
}

void CACHE::observe_partition_access(champsim::address addr, bool hit)
{
  if (!partition || partition->epoch_length == 0 || partition->target != reserved_metadata_ways)
    return;

  auto& state = *partition;
  if (state.dueller) {
    observe_dueller_data(champsim::block_number{addr}.to<uint64_t>());
    if (++state.demand_accesses == state.epoch_length) {
      auto& scores = state.dueller->scores;
      const auto best = std::max_element(scores.begin(), scores.end());
      const auto current = scores[reserved_metadata_ways];
      if (*best > current + current / 16)
        request_metadata_ways(static_cast<uint32_t>(std::distance(scores.begin(), best)));
      std::fill(scores.begin(), scores.end(), 0);
      state.demand_accesses = 0;
    }
    return;
  }

  state.demand_misses += !hit;
  if (++state.demand_accesses < state.epoch_length)
    return;

  const auto accesses = std::exchange(state.demand_accesses, 0);
  const auto misses = std::exchange(state.demand_misses, 0);
  const auto lookups = std::exchange(state.lookups, 0);
  const auto hits = std::exchange(state.hits, 0);
  const auto pairs = std::exchange(state.training_pairs, 0);

  // Compare observed demand pressure and metadata reuse. Require two agreeing epochs.
  const bool pressured = uint64_t{misses} * 4 >= uint64_t{accesses} * 3;
  int vote = 0;
  if (reserved_metadata_ways == 0) {
    vote = pairs > 0 ? 1 : 0; // Trial allocation after observing training at zero ways.
  } else if (lookups == 0 || hits * 8 < lookups || (pressured && hits * 4 < lookups)) {
    vote = -1;
  } else if (!pressured && hits * 4 >= lookups && hits * 4 <= lookups * 3 && reserved_metadata_ways < metadata_capacity_ways) {
    vote = 1;
  }

  if (vote != 0 && vote == state.last_vote) {
    const auto next = vote > 0 ? std::clamp(static_cast<uint64_t>(reserved_metadata_ways) * 2, uint64_t{1}, uint64_t{metadata_capacity_ways})
                               : reserved_metadata_ways / 2;
    request_metadata_ways(static_cast<uint32_t>(next));
    state.last_vote = 0;
  } else {
    state.last_vote = vote;
  }
}

uint64_t CACHE::metadata_line::read(std::size_t slot) const
{
  constexpr uint64_t entry_mask = (uint64_t{1} << metadata_entry_bits) - 1;
  const auto bit = slot * metadata_entry_bits;
  const auto word = bit / 64;
  const auto shift = bit % 64;
  auto entry = payload[word] >> shift;
  if (shift + metadata_entry_bits > 64)
    entry |= payload[word + 1] << (64 - shift);
  return entry & entry_mask;
}

void CACHE::metadata_line::write(std::size_t slot, uint64_t entry)
{
  constexpr uint64_t entry_mask = (uint64_t{1} << metadata_entry_bits) - 1;
  const auto bit = slot * metadata_entry_bits;
  const auto word = bit / 64;
  const auto shift = bit % 64;
  payload[word] = (payload[word] & ~(entry_mask << shift)) | (entry << shift);
  if (shift + metadata_entry_bits > 64) {
    const auto low_bits = 64 - shift;
    payload[word + 1] = (payload[word + 1] & ~(entry_mask >> low_bits)) | (entry >> low_bits);
  }
}

uint64_t CACHE::metadata_tag(uint64_t trigger) const
{
  uint64_t tag = 0;
  // The low line-number bits select the set. XOR-fold the remaining bits into the tag.
  for (auto high = trigger / NUM_SET; high != 0; high >>= metadata_tag_bits)
    tag ^= high & metadata_tag_mask;
  return tag;
}

CACHE::metadata_position CACHE::find_metadata(uint64_t trigger, uint64_t tag)
{
  const auto set = static_cast<std::size_t>(trigger % NUM_SET);
  const auto active_ways = metadata_ways_for_set(set);
  const auto first_line = set * metadata_capacity_ways;
  metadata_position oldest{&metadata_table[first_line], 0, false};
  for (std::size_t way = 0; way < active_ways; ++way) {
    auto& line = metadata_table[first_line + way];
    for (std::size_t slot = 0; slot < metadata_entries_per_way; ++slot) {
      const auto stamp = line.last_used[slot];
      if (stamp != 0 && (line.read(slot) & metadata_tag_mask) == tag)
        return {&line, slot, true};
      if (stamp < oldest.line->last_used[oldest.slot])
        oldest = {&line, slot, false};
    }
  }
  return oldest;
}

std::optional<champsim::block_number> CACHE::lookup_metadata(champsim::block_number trigger)
{
  if (metadata_table.empty())
    return std::nullopt;

  const auto trigger_line = trigger.to<uint64_t>();
  observe_dueller_metadata(trigger_line, false);
  const bool sampling = partition && !partition->dueller && partition->epoch_length != 0 && partition->target == reserved_metadata_ways;
  if (sampling)
    ++partition->lookups;
  if (metadata_ways_for_set(static_cast<std::size_t>(trigger_line % NUM_SET)) == 0)
    return std::nullopt;

  const auto entry = find_metadata(trigger_line, metadata_tag(trigger_line));
  if (!entry.hit)
    return std::nullopt;

  if (sampling)
    ++partition->hits;
  entry.line->last_used[entry.slot] = ++metadata_clock;
  const auto target = (entry.line->read(entry.slot) >> metadata_tag_bits) & metadata_target_mask;
  return champsim::block_number{target};
}

void CACHE::update_metadata(champsim::block_number trigger, champsim::block_number target)
{
  auto target_line = target.to<uint64_t>();
  if (metadata_table.empty() || target_line > metadata_target_mask)
    return;

  const auto trigger_line = trigger.to<uint64_t>();
  observe_dueller_metadata(trigger_line, true);
  if (partition && !partition->dueller && partition->epoch_length != 0 && partition->target == reserved_metadata_ways)
    ++partition->training_pairs;
  if (metadata_ways_for_set(static_cast<std::size_t>(trigger_line % NUM_SET)) == 0)
    return;

  const auto tag = metadata_tag(trigger_line);
  const auto entry = find_metadata(trigger_line, tag);
  bool changed_prediction = entry.line->last_used[entry.slot] != 0;
  bool confident = false;
  if (entry.hit) {
    const auto previous = entry.line->read(entry.slot);
    const auto old_target = (previous >> metadata_tag_bits) & metadata_target_mask;
    confident = old_target == target_line;
    if ((previous >> metadata_confidence_bit) != 0)
      target_line = old_target;
    changed_prediction = old_target != target_line;
  }
  if (changed_prediction && !metadata_set_revisions.empty())
    ++metadata_set_revisions[trigger_line % NUM_SET];

  const auto encoded = tag | (target_line << metadata_tag_bits) | (uint64_t{confident} << metadata_confidence_bit);
  entry.line->write(entry.slot, encoded);
  entry.line->last_used[entry.slot] = ++metadata_clock;
}

// LCOV_EXCL_START exclude deprecated function
uint64_t CACHE::get_way(uint64_t address, uint64_t /*unused set index*/) const
{
  champsim::address intern_addr{address};
  auto [begin, end] = get_set_span(intern_addr);
  const auto way = std::find_if(begin, end, matches_address(intern_addr));
  return way == end ? NUM_WAY : static_cast<uint64_t>(std::distance(begin, way));
}
// LCOV_EXCL_STOP

long CACHE::invalidate_entry(champsim::address inval_addr)
{
  auto [begin, end] = get_set_span(inval_addr);
  auto inv_way = std::find_if(begin, end, [match = matches_address(inval_addr)](const auto& block) { return block.valid && match(block); });

  if (inv_way != end) {
    pref_module_pimpl->impl_prefetcher_on_evict(inv_way->address);
    inv_way->valid = false;
  }

  return inv_way == end ? static_cast<long>(NUM_WAY) : std::distance(begin, inv_way);
}

bool CACHE::prefetch_line(champsim::address pf_addr, bool fill_this_level, uint32_t prefetch_metadata, champsim::triage_prefetch_source source)
{
  ++sim_stats.pf_requested;

  if (std::size(internal_PQ) >= PQ_SIZE) {
    return false;
  }

  request_type pf_packet;
  pf_packet.type = access_type::PREFETCH;
  pf_packet.pf_metadata = prefetch_metadata;
  pf_packet.cpu = cpu;
  pf_packet.address = pf_addr;
  pf_packet.v_address = virtual_prefetch ? pf_addr : champsim::address{};
  pf_packet.is_translated = !virtual_prefetch;

  internal_PQ.emplace_back(pf_packet, true, !fill_this_level);
  internal_PQ.back().prefetch_source = source;
  ++sim_stats.pf_issued;
  if (source != champsim::triage_prefetch_source::none)
    ++sim_stats.triage_prefetch.source(source).issued;

  return true;
}

// LCOV_EXCL_START exclude deprecated function
bool CACHE::prefetch_line(uint64_t pf_addr, bool fill_this_level, uint32_t prefetch_metadata)
{
  return prefetch_line(champsim::address{pf_addr}, fill_this_level, prefetch_metadata);
}

bool CACHE::prefetch_line(uint64_t /*deprecated*/, uint64_t /*deprecated*/, uint64_t pf_addr, bool fill_this_level, uint32_t prefetch_metadata)
{
  return prefetch_line(champsim::address{pf_addr}, fill_this_level, prefetch_metadata);
}
// LCOV_EXCL_STOP

void CACHE::finish_packet(const response_type& packet)
{
  // check MSHR information
  auto mshr_entry = std::find_if(std::begin(MSHR), std::end(MSHR), matches_address(packet.address));

  // sanity check
  if (mshr_entry == MSHR.end()) {
    fmt::print(stderr, "[{}_MSHR] {} cannot find a matching entry! address: {} v_address: {}\n", NAME, __func__, packet.address, packet.v_address);
    assert(0);
  }

  // MSHR holds the most updated information about this request
  fill_type::returned_value finished_value{packet.data, packet.pf_metadata};
  mshr_entry->data_promise = champsim::waitable{finished_value, current_time + (warmup ? champsim::chrono::clock::duration{} : FILL_LATENCY)};
  if constexpr (champsim::debug_print) {
    fmt::print("[{}_MSHR] finish_packet instr_id: {} address: {} data: {} type: {} current: {}\n", this->NAME, mshr_entry->instr_id, mshr_entry->address,
               mshr_entry->data_promise->data, access_type_names.at(champsim::to_underlying(mshr_entry->type)), current_time.time_since_epoch() / clock_period);
  }

  std::iter_swap(mshr_entry, std::begin(MSHR));
  inflight_fills.push_back(MSHR.front());
  MSHR.pop_front();
}

void CACHE::finish_translation(const response_type& packet)
{
  auto matches_vpage = [page_num = champsim::page_number{packet.v_address}](const auto& entry) {
    return (champsim::page_number{entry.v_address} == page_num) && !entry.is_translated;
  };
  auto mark_translated = [p_page = champsim::page_number{packet.data}, this](auto& entry) {
    [[maybe_unused]] auto old_address = entry.address;
    entry.address = champsim::address{champsim::splice(p_page, champsim::page_offset{entry.v_address})}; // translated address
    entry.is_translated = true;                                                                          // This entry is now translated

    if constexpr (champsim::debug_print) {
      fmt::print("[{}_TRANSLATE] finish_translation old: {} paddr: {} vaddr: {} type: {} cycle: {}\n", this->NAME, old_address, entry.address, entry.v_address,
                 access_type_names.at(champsim::to_underlying(entry.type)), this->current_time.time_since_epoch() / this->clock_period);
    }
  };

  // Restart stashed translations
  auto finish_begin = std::find_if_not(std::begin(translation_stash), std::end(translation_stash), [](const auto& x) { return x.is_translated; });
  auto finish_end = std::stable_partition(finish_begin, std::end(translation_stash), matches_vpage);
  std::for_each(finish_begin, finish_end, mark_translated);

  // Find all packets that match the page of the returned packet
  for (auto& entry : inflight_tag_check) {
    if (matches_vpage(entry)) {
      mark_translated(entry);
    }
  }
}

void CACHE::issue_translation(tag_lookup_type& q_entry) const
{
  if (!q_entry.translate_issued && !q_entry.is_translated) {
    request_type fwd_pkt;
    fwd_pkt.asid[0] = q_entry.asid[0];
    fwd_pkt.asid[1] = q_entry.asid[1];
    fwd_pkt.type = access_type::LOAD;
    fwd_pkt.cpu = q_entry.cpu;

    fwd_pkt.address = q_entry.address;
    fwd_pkt.v_address = q_entry.v_address;
    fwd_pkt.data = q_entry.data;
    fwd_pkt.instr_id = q_entry.instr_id;
    fwd_pkt.ip = q_entry.ip;

    fwd_pkt.instr_depend_on_me = q_entry.instr_depend_on_me;
    fwd_pkt.is_translated = true;

    q_entry.translate_issued = lower_translate->add_rq(fwd_pkt);
    if constexpr (champsim::debug_print) {
      if (q_entry.translate_issued) {
        fmt::print("[TRANSLATE] do_issue_translation instr_id: {} paddr: {} vaddr: {} type: {}\n", q_entry.instr_id, q_entry.address, q_entry.v_address,
                   access_type_names.at(champsim::to_underlying(q_entry.type)));
      }
    }
  }
}

std::size_t CACHE::get_mshr_occupancy() const { return std::size(MSHR); }

std::vector<std::size_t> CACHE::get_rq_occupancy() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->rq_occupancy(); });
  return retval;
}

std::vector<std::size_t> CACHE::get_wq_occupancy() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->wq_occupancy(); });
  return retval;
}

std::vector<std::size_t> CACHE::get_pq_occupancy() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->pq_occupancy(); });
  retval.push_back(std::size(internal_PQ));
  return retval;
}

// LCOV_EXCL_START exclude deprecated function
std::size_t CACHE::get_occupancy(uint8_t queue_type, uint64_t /*deprecated*/) const
{
  if (queue_type == 0) {
    return get_mshr_occupancy();
  }
  return 0;
}

std::size_t CACHE::get_occupancy(uint8_t queue_type, champsim::address /*deprecated*/) const
{
  if (queue_type == 0) {
    return get_mshr_occupancy();
  }
  return 0;
}
// LCOV_EXCL_STOP

std::size_t CACHE::get_mshr_size() const { return MSHR_SIZE; }
std::vector<std::size_t> CACHE::get_rq_size() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->rq_size(); });
  return retval;
}

std::vector<std::size_t> CACHE::get_wq_size() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->wq_size(); });
  return retval;
}

std::vector<std::size_t> CACHE::get_pq_size() const
{
  std::vector<std::size_t> retval;
  std::transform(std::begin(upper_levels), std::end(upper_levels), std::back_inserter(retval), [](auto ulptr) { return ulptr->pq_size(); });
  retval.push_back(PQ_SIZE);
  return retval;
}

// LCOV_EXCL_START exclude deprecated function
std::size_t CACHE::get_size(uint8_t queue_type, champsim::address /*deprecated*/) const
{
  if (queue_type == 0) {
    return get_mshr_size();
  }
  return 0;
}

std::size_t CACHE::get_size(uint8_t queue_type, uint64_t /*deprecated*/) const
{
  if (queue_type == 0) {
    return get_mshr_size();
  }
  return 0;
}
// LCOV_EXCL_STOP

namespace
{
double occupancy_ratio(std::size_t occ, std::size_t sz) { return std::ceil(occ) / std::ceil(sz); }

std::vector<double> occupancy_ratio_vec(std::vector<std::size_t> occ, std::vector<std::size_t> sz)
{
  std::vector<double> retval;
  std::transform(std::begin(occ), std::end(occ), std::begin(sz), std::back_inserter(retval), occupancy_ratio);
  return retval;
}
} // namespace

double CACHE::get_mshr_occupancy_ratio() const { return ::occupancy_ratio(get_mshr_occupancy(), get_mshr_size()); }

std::vector<double> CACHE::get_rq_occupancy_ratio() const { return ::occupancy_ratio_vec(get_rq_occupancy(), get_rq_size()); }

std::vector<double> CACHE::get_wq_occupancy_ratio() const { return ::occupancy_ratio_vec(get_wq_occupancy(), get_wq_size()); }

std::vector<double> CACHE::get_pq_occupancy_ratio() const { return ::occupancy_ratio_vec(get_pq_occupancy(), get_pq_size()); }

void CACHE::impl_prefetcher_initialize() const { pref_module_pimpl->impl_prefetcher_initialize(); }

uint32_t CACHE::impl_prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                              uint32_t metadata_in) const
{
  return pref_module_pimpl->impl_prefetcher_cache_operate(addr, ip, cache_hit, useful_prefetch, type, metadata_in);
}

uint32_t CACHE::impl_prefetcher_cache_fill(champsim::address addr, long set, long way, bool prefetch, champsim::address evicted_addr,
                                           uint32_t metadata_in) const
{
  return pref_module_pimpl->impl_prefetcher_cache_fill(addr, set, way, prefetch, evicted_addr, metadata_in);
}

void CACHE::impl_prefetcher_cycle_operate() const { pref_module_pimpl->impl_prefetcher_cycle_operate(); }

void CACHE::impl_prefetcher_final_stats() const { pref_module_pimpl->impl_prefetcher_final_stats(); }

void CACHE::impl_prefetcher_branch_operate(champsim::address ip, uint8_t branch_type, champsim::address branch_target) const
{
  pref_module_pimpl->impl_prefetcher_branch_operate(ip, branch_type, branch_target);
}

void CACHE::impl_initialize_replacement() const { repl_module_pimpl->impl_initialize_replacement(); }

long CACHE::impl_find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const BLOCK* current_set, champsim::address ip, champsim::address full_addr,
                             access_type type) const
{
  return repl_module_pimpl->impl_find_victim(triggering_cpu, instr_id, set, current_set, ip, full_addr, type);
}

void CACHE::impl_update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                          champsim::address victim_addr, access_type type, bool hit) const
{
  repl_module_pimpl->impl_update_replacement_state(triggering_cpu, set, way, full_addr, ip, victim_addr, type, hit);
}

void CACHE::impl_replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                        champsim::address victim_addr, access_type type) const
{
  repl_module_pimpl->impl_replacement_cache_fill(triggering_cpu, set, way, full_addr, ip, victim_addr, type);
}

void CACHE::impl_replacement_final_stats() const { repl_module_pimpl->impl_replacement_final_stats(); }

void CACHE::initialize()
{
  impl_prefetcher_initialize();
  impl_initialize_replacement();
}

void CACHE::begin_phase()
{
  stats_type new_roi_stats;
  stats_type new_sim_stats;

  new_roi_stats.name = NAME;
  new_sim_stats.name = NAME;
  new_roi_stats.partition.enabled = new_sim_stats.partition.enabled = partition.has_value();
  new_roi_stats.partition.ways_at_end = new_sim_stats.partition.ways_at_end = reserved_metadata_ways;
  new_roi_stats.sms_filter.enabled = new_sim_stats.sms_filter.enabled = sim_stats.sms_filter.enabled;
  new_roi_stats.triage_prefetch.mode = new_sim_stats.triage_prefetch.mode = sim_stats.triage_prefetch.mode;
  new_roi_stats.triangel.enabled = new_sim_stats.triangel.enabled = sim_stats.triangel.enabled;

  roi_stats = new_roi_stats;
  sim_stats = new_sim_stats;

  for (auto* ul : upper_levels) {
    channel_type::stats_type ul_new_roi_stats;
    channel_type::stats_type ul_new_sim_stats;
    ul->roi_stats = ul_new_roi_stats;
    ul->sim_stats = ul_new_sim_stats;
  }
}

void CACHE::end_phase(unsigned finished_cpu)
{
  finished_cpu = finished_cpu;
  roi_stats.total_miss_latency_cycles = sim_stats.total_miss_latency_cycles;

  roi_stats.hits = sim_stats.hits;
  roi_stats.misses = sim_stats.misses;
  roi_stats.miss_merge = sim_stats.miss_merge;
  roi_stats.fill = sim_stats.fill;

  roi_stats.pf_requested = sim_stats.pf_requested;
  roi_stats.pf_issued = sim_stats.pf_issued;
  roi_stats.pf_useful = sim_stats.pf_useful;
  roi_stats.pf_useless = sim_stats.pf_useless;
  roi_stats.pf_fill = sim_stats.pf_fill;
  roi_stats.partition = sim_stats.partition;
  roi_stats.sms_filter = sim_stats.sms_filter;
  roi_stats.triage_prefetch = sim_stats.triage_prefetch;
  roi_stats.triangel = sim_stats.triangel;

  for (auto* ul : upper_levels) {
    ul->roi_stats.RQ_ACCESS = ul->sim_stats.RQ_ACCESS;
    ul->roi_stats.RQ_FULL = ul->sim_stats.RQ_FULL;
    ul->roi_stats.RQ_TO_CACHE = ul->sim_stats.RQ_TO_CACHE;

    ul->roi_stats.PQ_ACCESS = ul->sim_stats.PQ_ACCESS;
    ul->roi_stats.PQ_FULL = ul->sim_stats.PQ_FULL;
    ul->roi_stats.PQ_TO_CACHE = ul->sim_stats.PQ_TO_CACHE;

    ul->roi_stats.WQ_ACCESS = ul->sim_stats.WQ_ACCESS;
    ul->roi_stats.WQ_FULL = ul->sim_stats.WQ_FULL;
    ul->roi_stats.WQ_TO_CACHE = ul->sim_stats.WQ_TO_CACHE;
  }
}

template <typename T>
bool CACHE::should_activate_prefetcher(const T& pkt) const
{
  return !pkt.prefetch_from_this && std::count(std::begin(pref_activate_mask), std::end(pref_activate_mask), pkt.type) > 0;
}

// LCOV_EXCL_START Exclude the following function from LCOV
void CACHE::print_deadlock()
{
  std::string_view mshr_write{"instr_id: {} address: {} v_addr: {} type: {} ready: {}"};
  auto mshr_pack = [time = current_time](const auto& entry) {
    return std::tuple{entry.instr_id, entry.address, entry.v_address, access_type_names.at(champsim::to_underlying(entry.type)),
                      entry.data_promise.is_ready_at(time)};
  };

  std::string_view tag_check_write{"instr_id: {} address: {} v_addr: {} is_translated: {} translate_issued: {} event_cycle: {}"};
  auto tag_check_pack = [period = clock_period](const auto& entry) {
    return std::tuple{entry.instr_id,      entry.address,          entry.v_address,
                      entry.is_translated, entry.translate_issued, entry.event_cycle.time_since_epoch() / period};
  };

  champsim::range_print_deadlock(MSHR, NAME + "_MSHR", mshr_write, mshr_pack);
  champsim::range_print_deadlock(inflight_tag_check, NAME + "_tags", tag_check_write, tag_check_pack);
  champsim::range_print_deadlock(translation_stash, NAME + "_translation", tag_check_write, tag_check_pack);

  std::string_view q_writer{"instr_id: {} address: {} v_addr: {} type: {} translated: {}"};
  auto q_entry_pack = [](const auto& entry) {
    return std::tuple{entry.instr_id, entry.address, entry.v_address, access_type_names.at(champsim::to_underlying(entry.type)), entry.is_translated};
  };

  for (auto* ul : upper_levels) {
    champsim::range_print_deadlock(ul->RQ, NAME + "_RQ", q_writer, q_entry_pack);
    champsim::range_print_deadlock(ul->WQ, NAME + "_WQ", q_writer, q_entry_pack);
    champsim::range_print_deadlock(ul->PQ, NAME + "_PQ", q_writer, q_entry_pack);
  }
}
// LCOV_EXCL_STOP
