#include "server-utility-lane.h"

#include <algorithm>
#include <cctype>

static std::string lower_trimmed(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) {
        b++;
    }
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        e--;
    }
    std::string out = s.substr(b, e - b);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return out;
}

bool server_is_utility(const std::map<std::string, std::string> & headers) {
    // the map keeps the client's spelling of each name, so look at every entry
    const std::string name  = lower_trimmed(SERVER_UTILITY_HEADER);
    const std::string value = lower_trimmed(SERVER_UTILITY_VALUE);
    for (const auto & [key, val] : headers) {
        if (lower_trimmed(key) == name) {
            return lower_trimmed(val) == value;
        }
    }
    return false;
}

int server_utility_slot(int n_parallel, int n_tokens) {
    return n_tokens > 0 && n_parallel > 0 ? n_parallel - 1 : -1;
}

std::string server_utility_refusal(int n_parallel, int n_ctx_slot, int n_tokens) {
    if (n_tokens <= 0) {
        return "";
    }
    if (n_parallel < 2) {
        return "--utility-lane needs at least 2 slots (-np 2 or more): the last one is the utility lane, the others "
               "are for people; got -np " + std::to_string(n_parallel);
    }
    if (n_tokens > n_ctx_slot) {
        return "--utility-lane " + std::to_string(n_tokens) + " is longer than a slot's context (" +
               std::to_string(n_ctx_slot) + "): lower it, or raise -c / --kv-unified-per-slot";
    }
    return "";
}

int server_deferred_pick(const std::vector<int> & id_slots, int id_slot_released, int id_slot_utility) {
    for (size_t i = 0; i < id_slots.size(); i++) {
        if (id_slots[i] == id_slot_released) {
            return (int) i;
        }
    }
    if (id_slot_utility < 0) {
        return id_slots.empty() ? -1 : 0; // the lane is off: upstream's default
    }
    const bool utility_freed = id_slot_released == id_slot_utility;
    for (size_t i = 0; i < id_slots.size(); i++) {
        if (utility_freed ? id_slots[i] != -1 : id_slots[i] == -1) {
            return (int) i;
        }
    }
    return id_slots.empty() || utility_freed ? -1 : 0;
}
