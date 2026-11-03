#include "../tools/vipera/place.h"

#include <climits>
#include <cstdio>

#undef NDEBUG
#include <cassert>

// Every case is a scripted probe: `feasible` decides what the card can take and
// nothing is spawned. The rung the search lands on is the thing under test - the
// two directions must not be interchangeable, and a failure that is not a
// placement answer must end the search instead of being read as one.

static constexpr int NO_FIT = 2;

// A card that fits every rung from `boundary` upward (more FFN on the CPU is
// easier), which is the shape `-ncffn auto` searches.
static int rising(int boundary, int n) {
    return n >= boundary ? 0 : NO_FIT;
}

// A card that fits every rung up to `boundary` (more layers on the GPU is
// harder), which is the shape both layer ladders search.
static int falling(int boundary, int n) {
    return n <= boundary ? 0 : NO_FIT;
}

static void test_smallest_edge_finds_the_lowest_rung_that_fits() {
    int probes = 0;
    const Bisect got = bisect_edge(1, 64, Edge::Smallest, NO_FIT, [&](int n) {
        ++probes;
        return rising(6, n);
    });

    // 7B on the reference 4 GB card: 5 does not fit, 6 is the smallest that does.
    assert(got.value == 6);
    assert(got.fatal_at == 0);
    // log2(64) + 1: a climb would have paid 6 spawns for the same answer, and the
    // point of the bisection is that the count does not grow when the range does.
    assert(probes <= 7);
}

static void test_largest_edge_finds_the_highest_rung_that_fits() {
    int probes = 0;
    const Bisect got = bisect_edge(1, 27, Edge::Largest, NO_FIT, [&](int n) {
        ++probes;
        return falling(20, n);
    });

    // The `-ngl` ladder case: 28 asked for, 22 never load, 21 is the most that do.
    assert(got.value == 20);
    assert(got.fatal_at == 0);
    assert(probes <= 6);
}

static void test_the_two_edges_land_on_opposite_ends_of_the_feasible_range() {
    // Rungs 10..20 fit. The same predicate, two answers: the search direction is
    // what picks the end, not the probe.
    const Bisect low = bisect_edge(1, 30, Edge::Smallest, NO_FIT,
                                   [](int n) { return n < 10 || n > 20 ? NO_FIT : 0; });
    const Bisect high = bisect_edge(1, 30, Edge::Largest, NO_FIT,
                                    [](int n) { return n < 10 || n > 20 ? NO_FIT : 0; });

    assert(low.value == 10);
    assert(high.value == 20);
}

static void test_nothing_feasible_answers_zero_from_either_edge() {
    const Bisect low = bisect_edge(1, 64, Edge::Smallest, NO_FIT,
                                   [](int) { return NO_FIT; });
    const Bisect high = bisect_edge(1, 200, Edge::Largest, NO_FIT,
                                    [](int) { return NO_FIT; });

    // "No rung fits" is a real answer - it is what sends `-ncffn auto` to the
    // mixed placement - so it must not be confused with a rung worth running.
    assert(low.value == 0 && low.fatal_at == 0);
    assert(high.value == 0 && high.fatal_at == 0);
}

static void test_everything_feasible_returns_the_edge_of_the_range() {
    const Bisect low = bisect_edge(1, 64, Edge::Smallest, NO_FIT, [](int) { return 0; });
    const Bisect high = bisect_edge(3, 200, Edge::Largest, NO_FIT, [](int) { return 0; });

    assert(low.value == 1);
    assert(high.value == 200);
}

static void test_a_fatal_probe_ends_the_search_and_discards_the_answer() {
    // 48 and 40 fit before 36 fails for an unrelated reason. Returning 40 here
    // would run a split the search never finished proving.
    const Bisect got = bisect_edge(1, 64, Edge::Smallest, NO_FIT, [](int n) {
        if (n == 36) {
            return 7;
        }
        return rising(40, n);
    });

    assert(got.value == 0);
    assert(got.fatal_at == 36);
    assert(got.fatal_code == 7);
}

static void test_a_fatal_probe_stops_immediately() {
    int probes = 0;
    const Bisect got = bisect_edge(1, 64, Edge::Largest, NO_FIT, [&](int) {
        ++probes;
        return 9;
    });

    assert(got.value == 0);
    assert(got.fatal_code == 9);
    // The first rung is the fatal one, so nothing past it may be spawned: a bad
    // model path would otherwise report itself once per rung.
    assert(probes == 1);
}

static void test_an_empty_range_probes_nothing() {
    int probes = 0;
    const Bisect got = bisect_edge(4, 3, Edge::Smallest, NO_FIT, [&](int) {
        ++probes;
        return 0;
    });

    assert(got.value == 0);
    assert(probes == 0);
}

int main(void) {
    test_smallest_edge_finds_the_lowest_rung_that_fits();
    test_largest_edge_finds_the_highest_rung_that_fits();
    test_the_two_edges_land_on_opposite_ends_of_the_feasible_range();
    test_nothing_feasible_answers_zero_from_either_edge();
    test_everything_feasible_returns_the_edge_of_the_range();
    test_a_fatal_probe_ends_the_search_and_discards_the_answer();
    test_a_fatal_probe_stops_immediately();
    test_an_empty_range_probes_nothing();

    std::printf("test-vipera-place: all tests passed\n");
    return 0;
}
