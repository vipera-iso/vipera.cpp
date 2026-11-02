#pragma once
// Self-speculation bookkeeping, kept out of the tool's main translation unit so
// the rule can be tested without a model. This rule decides how much of a verify
// batch survives, which is also what decides how much of the KV cache is rolled
// back - the part of --spec that is easy to get subtly wrong and impossible to
// assert from the outside.

#include "common.h"  // llama_tokens; common.h brings llama.h with it

#include <cstddef>

// How many rows of a verify pass the model keeps.
//
// Row i holds the token at pos + i and predicts the one after it, so row 0's
// sample decides whether draft[0] was right. The scan stops at the first sample
// that disagrees: rows past it are never sampled, exactly as a real pass would
// not ask for logits it is about to discard. The sample after the last draft is
// kept unconditionally - it is a new token, not a candidate, which is why a
// fully accepted pass yields one more token than it proposed.
//
// `sample_row(i)` must return the model's token for row i and is called in
// order, at most draft.size() + 1 times. The result is at least 1 (the carried
// token) and at most draft.size() + 1.
template <typename SampleRow>
inline std::size_t accepted_rows(const llama_tokens & draft, SampleRow sample_row) {
    std::size_t n_keep = 0;
    for (std::size_t i = 0; i <= draft.size(); ++i) {
        const llama_token t = sample_row(i);
        ++n_keep;
        // Order matters: the last row is the token that follows the draft, so
        // it is kept whatever it sampled, and indexing draft[i] there would be
        // out of bounds.
        if (i == draft.size() || t != draft[i]) {
            break;
        }
    }
    return n_keep;
}
