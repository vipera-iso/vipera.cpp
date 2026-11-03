#pragma once
// The placement search, kept out of the tool's main translation unit so the rule
// can be tested without a model. Every ladder in vipera walks a single number -
// FFN blocks read from RAM, or layers kept on the card - and every one of them
// relies on the same assumption: feasibility is monotone in that number. More
// FFN blocks in RAM can only ask less of the card; more layers on it can only ask
// more. That monotonicity is what makes a bisection the right search, and the two
// directions are not interchangeable: the FFN ladder looks for the smallest rung
// that fits (a smaller split decodes faster), the layer ladders for the largest.
//
// A probe has three outcomes, not two: fits, does not fit, and failed for a
// reason that says nothing about the placement (a bad model path, a build that
// cannot express the override, a crash). Reading the third as "does not fit" is
// how a ladder walks its whole range and then blames the card for a broken file.

enum class Edge {
    Smallest,  // feasibility rises with the rung: find the smallest that fits
    Largest,   // feasibility falls with the rung: find the largest that fits
};

struct Bisect {
    // The rung to run, or 0 when nothing in [lo, hi] fits. 0 is safe as "none"
    // because no ladder ever asks for 0 layers or 0 FFN blocks.
    int value = 0;
    // Set when a probe failed for a reason that is not "does not fit". The search
    // stops there, `value` is 0, and the caller reports fatal_code.
    int fatal_at = 0;
    int fatal_code = 0;
};

// Bisect [lo, hi] for the feasible edge. `probe(n)` returns an exit code: 0 for
// "this rung fits", `no_fit_code` for "it does not", anything else for a failure
// the search must not answer with.
template <typename Probe>
Bisect bisect_edge(int lo, int hi, Edge edge, int no_fit_code, Probe probe) {
    Bisect out;
    while (lo <= hi) {
        const int n = lo + (hi - lo) / 2;
        const int code = probe(n);
        if (code == 0) {
            out.value = n;
            // A feasible rung bounds the edge from one side and rules out every
            // rung beyond it in that direction.
            if (edge == Edge::Smallest) {
                hi = n - 1;
            } else {
                lo = n + 1;
            }
            continue;
        }
        if (code != no_fit_code) {
            // Not an answer about the placement. Keeping the best rung found so
            // far would run a split the search never finished proving.
            out.value = 0;
            out.fatal_at = n;
            out.fatal_code = code;
            return out;
        }
        if (edge == Edge::Smallest) {
            lo = n + 1;
        } else {
            hi = n - 1;
        }
    }
    return out;
}
