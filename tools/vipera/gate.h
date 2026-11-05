// gate.h - the one place a performance claim is judged.
//
// A number only counts when it beats the control by more than min_delta,
// measured as the median of three runs. Everything else is noise.
#pragma once

// median of three, without sorting the caller's values
static inline double vip_median3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { b = c; }
    return a > b ? a : b;
}

// strict: a +5.0% run does not pass a 5% gate
static inline bool vip_gate(double before, double after, double min_delta) {
    if (before <= 0.0) {
        return false;
    }
    return (after - before) / before > min_delta;
}
