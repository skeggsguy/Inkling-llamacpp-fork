#pragma once

// The utility lane (study UTILITY-LANE-PLAN step 1, Tom 2026-11-24): --utility-lane N keeps the last slot
// (n_parallel - 1) for short background jobs (titles, naming, memory capture) and caps it at N tokens. The other
// slots are the people lanes.
//
// - A request reaches the utility lane only with the header `X-Lane: utility` (the Flash-Next fork's header,
//   server-lane.h there); no other request is ever placed on it, and a utility request goes nowhere else.
// - A request longer than N is refused by the server's usual "exceeds the available context size" error.
// - When the lane is busy, the next utility request waits for it (the server's deferred tasks, by slot id).
// - The lane keeps its own memory between jobs: it is never saved to the RAM store (--cache-ram), never loaded
//   from it, and never cleared by the idle save-and-clear loop.
//
// Off by default (--utility-lane 0): no header is read and the server behaves as upstream. The rules live here,
// free of the server, so test-server-utility-lane covers them without a model; server-slot-sched.cpp keeps the
// people requests off the lane, server-context.cpp applies the rest.

#include <map>
#include <string>
#include <vector>

#define SERVER_UTILITY_HEADER "X-Lane"
#define SERVER_UTILITY_VALUE  "utility"

// Does this request ask for the utility lane? The header's name and value are matched ignoring case and
// surrounding spaces; any other value, or no such header, is a people request.
bool server_is_utility(const std::map<std::string, std::string> & headers);

// The utility lane's slot id: the last slot, or -1 when the lane is off (n_tokens <= 0).
int server_utility_slot(int n_parallel, int n_tokens);

// Why the server must refuse to start with --utility-lane n_tokens, or "" when it may: the lane needs at least one
// people lane beside it (n_parallel >= 2) and cannot be longer than a slot's own context (n_ctx_slot).
std::string server_utility_refusal(int n_parallel, int n_ctx_slot, int n_tokens);

// Which deferred task moves to the queue when slot id_slot_released frees up, given each deferred task's
// requested slot in queue order (-1: any slot) and the utility lane's slot (-1: off). -1 moves none.
// - the oldest task asking for the freed slot;
// - the utility lane freed: else the oldest task asking for some slot (a people request cannot use it, so it stays
//   in its place in the line);
// - a people lane freed: else the oldest task asking for no slot, else the oldest task.
// With the lane off it is upstream's rule: the oldest task asking for the freed slot, else the oldest task.
//
// Upstream takes the oldest task outright. With a utility lane, a utility request waiting on its busy lane would be
// taken when a people lane frees, go back to waiting, and leave a people request behind it waiting beside an empty
// lane; and a freed utility lane would send a waiting people request to the back of the line.
int server_deferred_pick(const std::vector<int> & id_slots, int id_slot_released, int id_slot_utility);
