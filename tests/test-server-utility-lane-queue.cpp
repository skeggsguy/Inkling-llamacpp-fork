// The utility lane's waiting and its option (test-server-utility-lane's second file, UTILITY-LANE-PLAN step 1).
//
// - server_deferred_pick(): examples, upstream's rule with the lane off, and generated queues against the rules
// - the real server_queue: a freed people lane moves the people task, not the utility task ahead of it; a freed
//   utility lane moves the utility task, and nothing when only people tasks wait
// - --utility-lane: off by default, from the command line and the environment, a negative value refused

#include "testing.h"

#include "arg.h"
#include "common.h"

#include "../tools/server/server-queue.h"
#include "../tools/server/server-utility-lane.h"

#include <cstdlib>
#include <random>
#include <vector>

void test_deferred(testing & t);
void test_queue(testing & t);
void test_option(testing & t);

// upstream's pop_deferred_task(): the oldest task asking for the freed slot, else the oldest task
static int upstream_pick(const std::vector<int> & ids, int released) {
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] == released) {
            return (int) i;
        }
    }
    return ids.empty() ? -1 : 0;
}

void test_deferred(testing & t) {
    const int U = 3, P = -1; // the utility lane is slot 3 of 4
    t.assert_equal("nothing waits", -1, server_deferred_pick({}, 0, U));
    t.assert_equal("people lane freed, people task behind a utility task", 1, server_deferred_pick({U, P}, 0, U));
    t.assert_equal("people lane freed, only utility tasks", 0, server_deferred_pick({U, U}, 0, U));
    t.assert_equal("utility lane freed, utility task behind a people task", 1, server_deferred_pick({P, U}, U, U));
    t.assert_equal("utility lane freed, only people tasks", -1, server_deferred_pick({P, P}, U, U));
    t.assert_equal("utility lane freed, a task naming slot 7 (wraps to 3)", 1, server_deferred_pick({P, 7}, U, U));
    t.assert_equal("a task naming the freed people lane", 2, server_deferred_pick({U, P, 1}, 1, U));
    t.assert_equal("lane off: upstream", 0, server_deferred_pick({U, P}, 0, -1));
    t.assert_equal("lane off: upstream, freed slot named", 1, server_deferred_pick({P, 2}, 2, -1));

    std::mt19937 rng(7);
    int n_bad_off = 0, n_bad_named = 0, n_bad_utility = 0, n_bad_people = 0;
    for (int iter = 0; iter < 5000; iter++) {
        const int n_slots = 2 + (int) (rng() % 4);
        const int utility = n_slots - 1;
        std::vector<int> ids(rng() % 6);
        for (auto & id : ids) {
            id = rng() % 2 ? -1 : (int) (rng() % n_slots);
        }
        const int released = (int) (rng() % n_slots);

        n_bad_off += server_deferred_pick(ids, released, -1) != upstream_pick(ids, released);

        const int pick = server_deferred_pick(ids, released, utility);
        int first_named = -1, first_any = -1, first_specific = -1;
        for (int i = (int) ids.size() - 1; i >= 0; i--) {
            first_named    = ids[i] == released ? i : first_named;
            first_any      = ids[i] == -1 ? i : first_any;
            first_specific = ids[i] != -1 ? i : first_specific;
        }
        if (first_named >= 0) {
            n_bad_named += pick != first_named;
        } else if (released == utility) {
            // a freed utility lane never moves a people task
            n_bad_utility += pick != first_specific || (pick >= 0 && ids[pick] == -1);
        } else {
            // a freed people lane moves the oldest people task when there is one
            n_bad_people += pick != (first_any >= 0 ? first_any : (ids.empty() ? -1 : 0));
        }
    }
    t.assert_equal("lane off is upstream", 0, n_bad_off);
    t.assert_equal("the freed slot's own task first", 0, n_bad_named);
    t.assert_equal("utility lane freed", 0, n_bad_utility);
    t.assert_equal("people lane freed", 0, n_bad_people);
}

// defer tasks asking for these slots (ids 1, 2, ...), free slot `released`, and return the ids that come out
static std::vector<int> run_queue(const std::vector<int> & id_slots, int released, int utility) {
    server_queue q;
    q.set_utility_slot(utility);
    for (size_t i = 0; i < id_slots.size(); i++) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id      = (int) i + 1;
        task.id_slot = id_slots[i];
        q.defer(std::move(task));
    }
    q.pop_deferred_task(released);

    std::vector<int> out;
    q.on_new_task([&](server_task && task, bool) {
        out.push_back(task.id);
        return true;
    });
    q.on_update_slots([&]() { q.terminate(); });
    q.start_loop(-1);
    return out;
}

void test_queue(testing & t) {
    t.assert_true("people lane freed: the people task", run_queue({3, -1}, 0, 3) == std::vector<int>({2}));
    t.assert_true("utility lane freed: the utility task", run_queue({-1, 3}, 3, 3) == std::vector<int>({2}));
    t.assert_true("utility lane freed: no people task", run_queue({-1, -1}, 3, 3).empty());
    t.assert_true("lane off: the oldest", run_queue({3, -1}, 0, -1) == std::vector<int>({1}));
}

static bool parse(std::vector<const char *> argv, common_params & params) {
    argv.insert(argv.begin(), "llama-server");
    return common_params_parse((int) argv.size(), const_cast<char **>(argv.data()), params, LLAMA_EXAMPLE_SERVER);
}

void test_option(testing & t) {
    {
        common_params p;
        t.assert_equal("off by default", 0, p.utility_lane);
        t.assert_true("parses with nothing", parse({}, p));
        t.assert_equal("still off", 0, p.utility_lane);
    }
    {
        common_params p;
        t.assert_true("flag parses", parse({"--utility-lane", "50000"}, p));
        t.assert_equal("flag", 50000, p.utility_lane);
    }
    {
        common_params p;
        t.assert_true("negative refused", !parse({"--utility-lane", "-1"}, p));
    }
    {
        setenv("LLAMA_ARG_UTILITY_LANE", "40000", 1);
        common_params p;
        t.assert_true("env parses", parse({}, p));
        t.assert_equal("env", 40000, p.utility_lane);
        unsetenv("LLAMA_ARG_UTILITY_LANE");
    }
}
