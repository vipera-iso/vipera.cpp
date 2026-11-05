// the performance gate: median of three, and a delta that must be beaten
// strictly. Pure logic, no model, no device.

#include "../tools/vipera/gate.h"

#include <cstdio>
#include <cstdlib>

static void expect(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
}

int main() {
    expect(vip_median3(3.0, 1.0, 2.0) == 2.0, "median of 3,1,2 is 2");
    expect(vip_median3(1.0, 2.0, 3.0) == 2.0, "median of 1,2,3 is 2");
    expect(vip_median3(3.0, 2.0, 1.0) == 2.0, "median of 3,2,1 is 2");
    expect(vip_median3(5.0, 5.0, 9.0) == 5.0, "median of 5,5,9 is 5");
    expect(vip_median3(-1.0, 0.0, 1.0) == 0.0, "median with a negative value");

    // +5% must NOT pass a 5% gate
    expect(!vip_gate(6.0, 6.3, 0.05), "exactly +5% does not pass a 5% gate");
    expect(vip_gate(6.0, 6.4, 0.05), "+6.7% passes a 5% gate");
    expect(!vip_gate(6.0, 5.0, 0.05), "a regression never passes");
    expect(!vip_gate(0.0, 9.0, 0.05), "no control baseline means no claim");

    printf("ok\n");
    return 0;
}
