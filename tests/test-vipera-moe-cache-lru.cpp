#include "../src/llama-moe-cache-lru.h"

#include <algorithm>
#include <cstdio>
#include <list>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>

#undef NDEBUG
#include <cassert>

// The eviction policy of the MoE expert cache, checked against a model that is
// obviously right. The real thing is an array of ints and a doubly linked list;
// the model is a map from slot to key and a std::list ordered least-recently
// used first. If the two disagree on any generated sequence of router
// selections, the linked list is wrong - and a wrong eviction here does not
// crash, it just uploads the wrong experts and decodes slowly.
//
// There is no model file, no device and no graph here: the pool, the slot maps
// and the router's picks are all the input this policy needs.

struct model_lru {
    int n_expert = 0;
    int n_slots  = 0;

    std::vector<std::vector<int>> slot_map; // [layer][expert] -> slot, or -1
    std::map<int, int> key_of_slot;         // slot -> il*n_expert + expert
    std::list<int> order;                   // least recently used first

    void init(int n_layer, int n_expert, int n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign((size_t) n_layer, std::vector<int>((size_t) n_expert, -1));
        key_of_slot.clear();
        order.clear();
        for (int s = 0; s < n_slots; ++s) {
            order.push_back(s);
        }
    }

    // The same contract as moe_cache_lru::plan, written the slow, obvious way.
    bool plan(int il, const std::vector<int> & ids, std::vector<std::pair<int, int>> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        std::vector<int> uniq;
        std::set<int> seen;
        for (int e : ids) {
            if (seen.insert(e).second) {
                uniq.push_back(e);
            }
        }
        if ((int) uniq.size() > n_slots) {
            return false;
        }

        for (int e : uniq) {
            const int s = slot_map[il][(size_t) e];
            if (s >= 0) {
                order.remove(s);
                order.push_back(s);
                ++n_hit;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int e : uniq) {
            if (slot_map[il][(size_t) e] >= 0) {
                continue;
            }
            const int s = order.front();
            const auto old = key_of_slot.find(s);
            if (old != key_of_slot.end()) {
                slot_map[(size_t) (old->second / n_expert)][(size_t) (old->second % n_expert)] = -1;
            }
            key_of_slot[s] = il * n_expert + e;
            slot_map[il][(size_t) e] = s;
            order.remove(s);
            order.push_back(s);
            fills.emplace_back(e, s);
        }
        return true;
    }
};

// Wire a real LRU to slot maps the caller owns, the way llama_moe_cache does
// with host tensors.
struct harness {
    moe_cache_lru lru;
    model_lru model;
    std::vector<std::vector<int32_t>> maps;

    harness(int n_layer, int n_expert, int n_slots) {
        lru.init(n_layer, n_expert, n_slots);
        maps.assign((size_t) n_layer, std::vector<int32_t>((size_t) n_expert, -1));
        for (int l = 0; l < n_layer; ++l) {
            lru.slot_map[(size_t) l] = maps[(size_t) l].data();
        }
        model.init(n_layer, n_expert, n_slots);
    }

    bool plan(int il, const std::vector<int32_t> & ids, std::vector<moe_cache_lru::fill> & fills, size_t & hits) {
        return lru.plan(il, ids.data(), ids.size(), fills, hits);
    }
};

static void test_eviction_crosses_layers_because_the_pool_is_shared() {
    // Two layers, two slots. An expert used again in layer 1 must keep its slot,
    // so the next miss in layer 0 evicts layer 0's own expert - not whatever was
    // touched longest ago in some other layer. The pool is shared, so "least
    // recently used" has to mean the group, not the layer being planned.
    harness h(2, 4, 2);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0;

    assert(h.plan(0, { 0 }, fills, hits));
    assert(hits == 0 && fills.size() == 1);
    const int32_t slot_of_0 = h.maps[0][0];
    assert(h.plan(1, { 1 }, fills, hits));
    const int32_t slot_of_1 = h.maps[1][1];
    assert(slot_of_1 >= 0 && slot_of_1 != slot_of_0);

    assert(h.plan(1, { 1 }, fills, hits));  // hit: layer 1's expert is now MRU
    assert(hits == 1);

    assert(h.plan(0, { 2 }, fills, hits));  // miss: must take layer 0's slot
    assert(hits == 0);
    assert(fills.size() == 1);
    assert(fills[0].expert == 2);
    assert(fills[0].slot == slot_of_0);
    assert(h.maps[0][0] == -1);  // evicted
    assert(h.maps[1][1] == slot_of_1);  // protected by the hit
}

static void test_more_distinct_experts_than_slots_is_refused_and_changes_nothing() {
    harness h(1, 4, 2);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0;

    assert(!h.plan(0, { 0, 1, 2 }, fills, hits));
    assert(fills.empty() && hits == 0);
    for (int32_t slot : h.maps[0]) {
        assert(slot == -1);
    }
}

static void test_the_same_expert_twice_in_one_token_is_one_slot() {
    // A router can name the same expert twice in a token; the second mention
    // must not consume a second slot or count as a second hit.
    harness h(1, 4, 2);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0;

    assert(h.plan(0, { 3, 3, 3 }, fills, hits));
    assert(hits == 0);
    assert(fills.size() == 1);
    assert(h.plan(0, { 3, 3 }, fills, hits));
    assert(hits == 1);
    assert(fills.empty());
}

static void test_against_a_model() {
    const int n_layer  = 3;
    const int n_expert = 8;
    const int n_slots  = 3;
    harness h(n_layer, n_expert, n_slots);

    std::mt19937 rng(0xC0FFEE);
    std::vector<moe_cache_lru::fill> fills;
    std::vector<std::pair<int, int>> model_fills;
    size_t hits = 0;
    size_t model_hits = 0;
    std::vector<int32_t> ids;

    for (int step = 0; step < 5000; ++step) {
        const int il = (int) (rng() % (uint32_t) n_layer);
        const int n  = 1 + (int) (rng() % 5u);
        ids.clear();
        for (int i = 0; i < n; ++i) {
            ids.push_back((int32_t) (rng() % (uint32_t) n_expert));
        }

        const bool ok = h.plan(il, ids, fills, hits);
        const std::vector<int> model_ids(ids.begin(), ids.end());
        const bool model_ok = h.model.plan(il, model_ids, model_fills, model_hits);
        assert(ok == model_ok);

        if (ok) {
            assert(hits == model_hits);
            assert(fills.size() == model_fills.size());
            for (size_t i = 0; i < fills.size(); ++i) {
                assert(fills[i].expert == model_fills[i].first);
                assert(fills[i].slot == model_fills[i].second);
            }
            // The caller's slot map is what the kernel reads, so it has to
            // agree after every step, not only at the end.
            for (int l = 0; l < n_layer; ++l) {
                for (int e = 0; e < n_expert; ++e) {
                    assert(h.maps[(size_t) l][(size_t) e] == h.model.slot_map[(size_t) l][(size_t) e]);
                }
            }
        }
    }
}

int main(void) {
    test_eviction_crosses_layers_because_the_pool_is_shared();
    test_more_distinct_experts_than_slots_is_refused_and_changes_nothing();
    test_the_same_expert_twice_in_one_token_is_one_slot();
    test_against_a_model();

    std::printf("test-vipera-moe-cache-lru: all tests passed\n");
    return 0;
}
