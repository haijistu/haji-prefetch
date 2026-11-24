#include "triage.h"

#include <stdexcept>

#include "cache.h"

void triage::prefetcher_initialize()
{
  if (intern_->get_lower_cache() == nullptr || intern_->virtual_prefetch)
    throw std::invalid_argument{"Triage needs a lower cache for metadata and physical prefetch addresses"};
  intern_->sim_stats.triage_prefetch.mode = "triage";
}

uint32_t triage::prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                         uint32_t metadata_in)
{
  if (type == access_type::LOAD && (!cache_hit || useful_prefetch))
    train(champsim::block_number{addr}, ip);
  return metadata_in;
}

std::optional<champsim::block_number> triage::train(champsim::block_number current, champsim::address ip)
{
  auto* llc = intern_->get_lower_cache();
  if (llc->max_metadata_ways() == 0)
    return std::nullopt;

  const auto previous = training_table.check_hit({ip, {}});
  if (previous && previous->last_address == current)
    return std::nullopt;

  training_table.fill({ip, current});

  if (pending.size() < intern_->PQ_SIZE) {
    if (auto target = llc->lookup_metadata(current); target)
      pending.push_back({current, *target, ip, intern_->current_time + llc->HIT_LATENCY});
  }
  if (previous) {
    llc->update_metadata(previous->last_address, current);
    return previous->last_address;
  }

  return std::nullopt;
}

void triage::prefetcher_cycle_operate()
{
  while (!pending.empty() && pending.front().ready_at <= intern_->current_time) {
    if (!intern_->prefetch_line(champsim::address{pending.front().target}, true, 0, champsim::triage_prefetch_source::triage))
      break;
    pending.pop_front();
  }
}
