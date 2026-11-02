#include "../tools/vipera/accept.h"

#include <cstddef>
#include <cstdio>

#undef NDEBUG
#include <cassert>

// Every case is a scripted verify pass: `sample_row` hands back the token the
// model would have sampled for a row, and the row count it was asked for is what
// says whether the scan stopped where it should have. No context, no weights, no
// KV cache - the decision itself is the thing under test.

static void test_a_fully_accepted_draft_keeps_the_row_after_it() {
    const llama_tokens draft = {10, 11, 12};
    std::size_t sampled = 0;

    const std::size_t n_keep = accepted_rows(draft, [&](std::size_t i) {
        ++sampled;
        return (llama_token) (10 + i);  // 10, 11, 12, and 13 for the row after
    });

    // Three confirmed drafts plus the token that follows them: a fully accepted
    // pass produces one more token than it proposed.
    assert(n_keep == 4);
    assert(sampled == 4);
}

static void test_the_row_after_the_draft_is_kept_even_when_it_disagrees() {
    const llama_tokens draft = {5, 5};
    std::size_t sampled = 0;

    // Row 2 is not a candidate, so 6 must not be compared against anything -
    // this is the case that catches a scan that treats it as a third draft.
    const std::size_t n_keep = accepted_rows(draft, [&](std::size_t i) {
        ++sampled;
        return i == 2 ? 6 : 5;
    });

    assert(n_keep == 3);
    assert(sampled == 3);
}

static void test_rejecting_the_first_draft_survives_only_the_carried_token() {
    const llama_tokens draft = {10, 11, 12};
    std::size_t sampled = 0;

    const std::size_t n_keep = accepted_rows(draft, [&](std::size_t) {
        ++sampled;
        return 99;  // every row disagrees
    });

    assert(n_keep == 1);
    // Rows past the first mismatch are never sampled: a real pass must not pay
    // for logits it is about to discard, and sampling advances the sampler.
    assert(sampled == 1);
}

static void test_a_mismatch_mid_draft_keeps_the_prefix_before_it() {
    const llama_tokens draft = {10, 11, 12};
    std::size_t sampled = 0;

    const std::size_t n_keep = accepted_rows(draft, [&](std::size_t i) {
        ++sampled;
        return i == 1 ? 99 : draft[i];
    });

    // The first row matched (10), the second did not: the cache keeps the
    // carried token and the one confirmed draft, and is rolled back after that.
    assert(n_keep == 2);
    assert(sampled == 2);
}

static void test_an_empty_draft_still_keeps_the_carried_token() {
    const llama_tokens draft;
    std::size_t sampled = 0;

    const std::size_t n_keep = accepted_rows(draft, [&](std::size_t) {
        ++sampled;
        return 7;
    });

    assert(n_keep == 1);
    assert(sampled == 1);
}

static void test_repeated_tokens_are_matched_positionally() {
    const llama_tokens draft = {5, 5, 5};
    std::size_t sampled = 0;

    const std::size_t n_keep = accepted_rows(draft, [&](std::size_t i) {
        ++sampled;
        return i == 2 ? 7 : 5;
    });

    // Two of three drafts confirmed, then the replacement token: positional, so
    // the duplicate at index 2 does not extend the match.
    assert(n_keep == 3);
    assert(sampled == 3);
}

static void test_rows_are_sampled_in_order_from_zero() {
    const llama_tokens draft = {10, 11, 12};
    llama_tokens seen;

    accepted_rows(draft, [&](std::size_t i) {
        seen.push_back((llama_token) i);
        return draft[i];
    });

    // The scan indexes rows by position, so it has to ask for 0, 1, 2, 3 in
    // that order - sampling row i out of order would read the wrong logits.
    assert(seen.size() == 4);
    for (std::size_t i = 0; i < seen.size(); ++i) {
        assert(seen[i] == (llama_token) i);
    }
}

int main(void) {
    test_a_fully_accepted_draft_keeps_the_row_after_it();
    test_the_row_after_the_draft_is_kept_even_when_it_disagrees();
    test_rejecting_the_first_draft_survives_only_the_carried_token();
    test_a_mismatch_mid_draft_keeps_the_prefix_before_it();
    test_an_empty_draft_still_keeps_the_carried_token();
    test_repeated_tokens_are_matched_positionally();
    test_rows_are_sampled_in_order_from_zero();

    std::printf("test-vipera-accept: all tests passed\n");
    return 0;
}
