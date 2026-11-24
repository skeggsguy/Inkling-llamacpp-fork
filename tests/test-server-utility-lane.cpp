// The utility lane's rules (tools/server/server-utility-lane.h, study UTILITY-LANE-PLAN step 1), without a model.
//
// - the header: X-Lane: utility, name and value ignoring case and surrounding spaces; anything else is people
// - the lane's slot and the start-up refusals (fewer than 2 slots, longer than a slot's context)
// - server_slot_sched: select() never puts a people task on the lane, even when it is the only idle slot or the best
//   match; a task naming it gets it, busy or not, and never touches the prompt cache; get_free_slots(), the idle
//   save-and-clear list and the purge pick all leave it out; generated slot states hold the same rules
// - in test-server-utility-lane-queue.cpp: the deferred pick, the real server_queue, and the option

#include "testing.h"

#include "../tools/server/server-slot-sched.h"
#include "../tools/server/server-utility-lane.h"

#include <cstdlib>
#include <random>
#include <string>
#include <vector>

void test_deferred(testing & t); // test-server-utility-lane-queue.cpp
void test_queue(testing & t);
void test_option(testing & t);

static void test_header(testing & t) {
    t.assert_true("exact", server_is_utility({{"X-Lane", "utility"}}));
    // HTTP header names are case-insensitive, and the server's header map keeps the client's spelling
    t.assert_true("name lower", server_is_utility({{"x-lane", "utility"}}));
    t.assert_true("name upper", server_is_utility({{"X-LANE", "utility"}}));
    t.assert_true("value case", server_is_utility({{"X-Lane", "Utility"}}));
    t.assert_true("value spaces", server_is_utility({{"X-Lane", "  utility \t"}}));
    t.assert_true("name spaces", server_is_utility({{" X-Lane ", "utility"}}));
    t.assert_true("among others", server_is_utility({{"Content-Type", "application/json"}, {"x-lane", "UTILITY"}}));

    t.assert_true("no headers", !server_is_utility({}));
    t.assert_true("no such header", !server_is_utility({{"Content-Type", "application/json"}, {"User-Agent", "curl/8"}}));
    t.assert_true("Sidekick's code lane", !server_is_utility({{"X-Lane", "code"}}));
    t.assert_true("chat", !server_is_utility({{"X-Lane", "chat"}}));
    t.assert_true("empty value", !server_is_utility({{"X-Lane", ""}}));
    t.assert_true("longer value", !server_is_utility({{"X-Lane", "utility2"}}));
    t.assert_true("value inside", !server_is_utility({{"X-Lane", "not utility"}}));
    t.assert_true("another header", !server_is_utility({{"X-Lanes", "utility"}}));
    t.assert_true("value elsewhere", !server_is_utility({{"X-Role", "utility"}}));
}

static void test_slot_and_refusal(testing & t) {
    t.assert_equal("off", -1, server_utility_slot(4, 0));
    t.assert_equal("the last slot", 3, server_utility_slot(4, 50000));
    t.assert_equal("two slots", 1, server_utility_slot(2, 10));

    t.assert_true("off never refuses", server_utility_refusal(1, 100, 0).empty());
    t.assert_true("the daily shape", server_utility_refusal(4, 200000, 50000).empty());
    t.assert_true("as long as a slot", server_utility_refusal(2, 200000, 200000).empty());
    t.assert_true("one slot", !server_utility_refusal(1, 200000, 50000).empty());
    t.assert_true("zero slots", !server_utility_refusal(0, 200000, 50000).empty());
    t.assert_true("longer than a slot", !server_utility_refusal(4, 200000, 200001).empty());
}

// the scheduler's view of n slots; prompts[i] holds slot i's tokens
struct lanes {
    std::vector<server_prompt>    prompts;
    std::vector<server_slot_info> slots;

    explicit lanes(int n) : prompts(n) {
        for (int i = 0; i < n; i++) {
            slots.push_back({ i, nullptr, false, i, nullptr });
        }
        for (int i = 0; i < n; i++) {
            slots[i].prompt = &prompts[i];
        }
    }

    void hold(int i, const llama_tokens & tokens) {
        prompts[i].tokens = server_tokens(tokens, false);
    }
};

static server_task make_task(const llama_tokens & tokens, int id_slot = -1) {
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.tokens  = server_tokens(tokens, false);
    task.id_slot = id_slot;
    return task;
}

static server_slot_sched make_sched(int reserved, float similarity) {
    server_slot_sched sched;
    sched.id_slot_reserved       = reserved;
    sched.slot_prompt_similarity = similarity;
    return sched;
}

static void test_select(testing & t) {
    const std::vector<common_adapter_lora_info> lora;
    const llama_tokens prompt = {1, 2, 3, 4, 5, 6, 7, 8};

    for (float sim : {0.0f, 0.1f}) {
        const std::string tag = sim > 0 ? " (similarity on)" : " (similarity off)";
        {
            // the people lanes are busy and the lane is idle and empty: the task waits
            lanes l(4);
            for (int i = 0; i < 3; i++) {
                l.slots[i].is_processing = true;
            }
            const auto plan = make_sched(3, sim).select(l.slots, make_task(prompt), lora);
            t.assert_equal("only idle slot is the lane" + tag, -1, plan.id_slot);

            // with the lane off the same slot is taken
            t.assert_equal("lane off takes it" + tag, 3, make_sched(-1, sim).select(l.slots, make_task(prompt), lora).id_slot);
        }
        {
            // the lane holds this very prompt and is the least recently used: still a people lane
            lanes l(4);
            l.hold(3, prompt);
            l.hold(0, {9, 9});
            l.slots[3].t_last_used = -5;
            const auto plan = make_sched(3, sim).select(l.slots, make_task(prompt), lora);
            t.assert_true("best match is the lane" + tag, plan.id_slot >= 0 && plan.id_slot != 3);
        }
        {
            // a task naming the lane gets it, idle
            lanes l(4);
            l.hold(3, {9, 9, 9});
            const auto plan = make_sched(3, sim).select(l.slots, make_task(prompt, 3), lora);
            t.assert_equal("named, idle" + tag, 3, plan.id_slot);
            t.assert_true("named, no cache update" + tag, !plan.update_cache);

            // and busy: the caller defers it
            l.slots[3].is_processing = true;
            const auto busy = make_sched(3, sim).select(l.slots, make_task(prompt, 3), lora);
            t.assert_equal("named, busy" + tag, 3, busy.id_slot);
            t.assert_true("named, busy, no cache update" + tag, !busy.update_cache);
        }
        {
            // a people slot taken from another prompt still goes through the prompt cache
            lanes l(4);
            l.hold(0, {9, 9, 9});
            l.hold(1, {9, 9, 9});
            l.hold(2, {9, 9, 9});
            const auto plan = make_sched(3, sim).select(l.slots, make_task(prompt), lora);
            t.assert_true("people lane" + tag, plan.id_slot >= 0 && plan.id_slot < 3);
            t.assert_true("people lane updates the cache" + tag, plan.update_cache);

            // the same task named onto the lane, which holds another prompt: no cache update
            l.hold(3, {9, 9, 9});
            t.assert_true("lane with another prompt" + tag, !make_sched(3, sim).select(l.slots, make_task(prompt, 3), lora).update_cache);
            // (upstream saves a slot named by id only through the similarity pass)
            t.assert_true("lane off, same slot by id" + tag,
                          make_sched(-1, sim).select(l.slots, make_task(prompt, 3), lora).update_cache == (sim > 0));
        }
    }
}

static void test_lists(testing & t) {
    lanes l(4);
    l.hold(3, {1, 2, 3});
    l.slots[3].t_last_used = -10; // the lane would be first by age
    l.slots[1].is_processing = true;
    const auto sched = make_sched(3, 0.0f);

    t.assert_true("free slots", sched.get_free_slots(l.slots, 4, -1) == std::vector<int>({0, 2}));
    t.assert_true("free slots, one", sched.get_free_slots(l.slots, 1, -1) == std::vector<int>({0}));
    t.assert_true("free slots, lane off", make_sched(-1, 0.0f).get_free_slots(l.slots, 4, -1) == std::vector<int>({0, 2, 3}));

    t.assert_true("idle to cache", sched.get_idle_slots_to_cache(l.slots) == std::vector<int>({0, 2}));
    t.assert_true("idle to cache, lane off", make_sched(-1, 0.0f).get_idle_slots_to_cache(l.slots) == std::vector<int>({0, 2, 3}));

    t.assert_equal("purge: only the lane holds tokens", -1, sched.get_idle_slot_to_purge(l.slots));
    t.assert_equal("purge, lane off", 3, make_sched(-1, 0.0f).get_idle_slot_to_purge(l.slots));
    l.hold(2, {4, 5});
    t.assert_equal("purge a people lane", 2, sched.get_idle_slot_to_purge(l.slots));
}

// generated slot states: busy or idle, empty, holding the task's prompt or another, any age
static void test_generated(testing & t) {
    std::mt19937 rng(42);
    const std::vector<common_adapter_lora_info> lora;
    const llama_tokens prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    int n_bad_select = 0, n_bad_named = 0, n_bad_lists = 0, n_bad_off = 0;

    for (int iter = 0; iter < 2000; iter++) {
        const int n = 2 + (int) (rng() % 5);
        const int reserved = n - 1;
        lanes l(n);
        for (int i = 0; i < n; i++) {
            l.slots[i].is_processing = rng() % 3 == 0;
            l.slots[i].t_last_used   = (int64_t) (rng() % 100);
            switch (rng() % 3) {
                case 0: break;
                case 1: l.hold(i, prompt); break;
                default: l.hold(i, {(llama_token) (1 + rng() % 2), 9, 9}); break;
            }
        }
        const float sim = rng() % 2 ? 0.1f : 0.0f;
        const auto sched = make_sched(reserved, sim);

        // a people task never lands on the lane, and gets a slot whenever a people lane is idle
        bool people_idle = false;
        for (int i = 0; i < reserved; i++) {
            people_idle |= !l.slots[i].is_processing;
        }
        const auto plan = sched.select(l.slots, make_task(prompt), lora);
        n_bad_select += plan.id_slot == reserved || (plan.id_slot == -1) == people_idle;

        // a task naming the lane always gets it, and never with a cache update
        const auto named = sched.select(l.slots, make_task(prompt, reserved), lora);
        n_bad_named += named.id_slot != reserved || named.update_cache;

        // the lists never hold the lane
        for (int id : sched.get_free_slots(l.slots, n, -1)) {
            n_bad_lists += id == reserved;
        }
        for (int id : sched.get_idle_slots_to_cache(l.slots)) {
            n_bad_lists += id == reserved;
        }
        n_bad_lists += sched.get_idle_slot_to_purge(l.slots) == reserved;

        // with the lane off, the people task is placed as before: the same pick for the same state
        const auto off = make_sched(-1, sim).select(l.slots, make_task(prompt), lora);
        n_bad_off += (off.id_slot == -1) != (plan.id_slot == -1 && l.slots[reserved].is_processing);
    }
    t.assert_equal("people tasks keep off the lane", 0, n_bad_select);
    t.assert_equal("named tasks get the lane", 0, n_bad_named);
    t.assert_equal("lists leave the lane out", 0, n_bad_lists);
    t.assert_equal("lane off: a slot whenever one is idle", 0, n_bad_off);
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("header",          test_header);
    t.test("slot_refusal",    test_slot_and_refusal);
    t.test("select",          test_select);
    t.test("lists",           test_lists);
    t.test("generated",       test_generated);
    t.test("deferred",        test_deferred);
    t.test("queue",           test_queue);
    t.test("option",          test_option);

    return t.summary();
}
