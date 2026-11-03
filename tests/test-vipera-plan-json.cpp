#include "../tools/vipera/plan_json.h"

#include <cstdio>
#include <string>

#undef NDEBUG
#include <cassert>

// The plan as data. No model, no GPU, no run: a `plan_report` filled by hand is
// the whole input, and what is under test is that a caller reading this object
// cannot be misled - no unescaped quote in a path, no 0 where the answer was
// "unknown", no empty string where the answer was "no card".

static plan_report sample() {
    plan_report r;
    r.path         = "/models/qwen2.5-1.5b.gguf";
    r.model_mib    = 1059;
    r.n_layer      = 28;
    r.n_head_kv    = 2;
    r.head_dim     = 128;
    r.train_ctx    = 32768;
    r.has_gpu      = true;
    r.gpu_name     = "NVIDIA GeForce GTX 1650";
    r.gpu_total_mib = 3730;
    r.gpu_free_mib  = 3674;
    r.headroom_mib  = 233;
    r.auto_placement = true;
    r.gpu_layers   = 28;
    r.kv_k         = "f16";
    r.kv_v         = "f16";
    r.kv_mib       = 896;
    r.resident_mib = 0;
    r.available_mib = 13345;
    r.available_known = true;
    r.ctx_requested = 32768;
    r.ctx_used     = 32768;
    r.threads      = 4;
    return r;
}

static bool has(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

static int count_of(const std::string & haystack, const std::string & needle) {
    int n = 0;
    for (auto at = haystack.find(needle); at != std::string::npos;
         at = haystack.find(needle, at + needle.size())) {
        ++n;
    }
    return n;
}

// Every `"` that opens a JSON string is either escaped or closes one, so an even
// count plus balanced braces is a cheap check that the object is well formed.
// The contract, frozen. A caller reads these keys and these types; if a field is
// added, renamed or retyped, this string changes and the test fails, which is
// the point - the alternative is a consumer that keeps parsing an object it no
// longer understands. Adding a field means bumping PLAN_JSON_SCHEMA and updating
// this fixture and tools/vipera/README.md in the same commit.
static void test_the_whole_object_is_the_documented_one() {
    const std::string expected =
        R"({"schema": 1, "model": {"size_mib": 1059, "n_layer": 28, "n_head_kv": 2, "head_dim": 128, "train_ctx": 32768, "path": "/models/qwen2.5-1.5b.gguf"}, "device": {"total_mib": 3730, "free_mib": 3674, "headroom_mib": 233, "name": "NVIDIA GeForce GTX 1650"}, "placement": {"auto": true, "fit_failed": false, "ffn_split": false, "gpu_layers": 28, "ffn_on_cpu": 0}, "kv": {"mib": 896, "lowered": false, "in_ram": false, "k": "f16", "v": "f16"}, "memory": {"resident_mib": 0, "available_known": true, "available_mib": 13345}, "context": {"requested": 32768, "used": 32768, "train_limit": 32768, "clamped": false, "limited_by_train": false, "reason": null}, "threads": 4})";

    const std::string out = plan_json(sample());

    if (out.compare(0, expected.size(), expected) != 0) {
        std::fprintf(stderr, "expected:\n%s\nactual:\n%s\n", expected.c_str(), out.c_str());
    }
    // Newline-terminated on purpose: a caller can read one line and stop.
    assert(out.size() == expected.size() + 1);
    assert(out.back() == '\n');
    assert(out.compare(0, expected.size(), expected) == 0);
}


static void test_the_schema_number_is_deliberate() {
    // A bump is a promise that a caller must re-read the object. Freezing the
    // number here makes the bump a decision rather than a side effect.
    assert(PLAN_JSON_SCHEMA == 1);
}


static void test_the_object_balances() {
    const std::string out = plan_json(sample());

    assert(out.front() == '{');
    assert(out.back() == '\n');
    assert(out[out.size() - 2] == '}');

    int braces = 0;
    int quotes = 0;
    bool escaped = false;
    for (const char ch : out) {
        if (ch == '\\' && !escaped) {
            escaped = true;
            continue;
        }
        if (ch == '"' && !escaped) {
            ++quotes;
        } else if (!escaped && ch == '{') {
            ++braces;
        } else if (!escaped && ch == '}') {
            --braces;
        }
        escaped = false;
    }
    assert(braces == 0);
    assert(quotes % 2 == 0);
}

static void test_the_placement_is_reported_as_numbers() {
    const std::string out = plan_json(sample());

    assert(has(out, "\"schema\": 1"));
    assert(has(out, "\"gpu_layers\": 28"));
    assert(has(out, "\"n_layer\": 28"));
    // Exactly one copy, under "model": two of them could disagree, and a caller
    // reading "placement" would have to guess which one the plan meant.
    assert(count_of(out, "\"n_layer\"") == 1);
    assert(has(out, "\"auto\": true"));
    assert(has(out, "\"ffn_split\": false"));
    assert(has(out, "\"size_mib\": 1059"));
    assert(has(out, "\"total_mib\": 3730"));
}

static void test_no_card_is_null_not_an_empty_object() {
    plan_report r = sample();
    r.has_gpu  = false;
    r.gpu_name = "";

    const std::string out = plan_json(r);

    assert(has(out, "\"device\": null"));
    // Nothing of the old card survives: no name, no sizes, no empty-string card.
    assert(!has(out, "NVIDIA"));
    assert(!has(out, "total_mib"));
}

static void test_a_path_is_escaped() {
    plan_report r = sample();
    r.path = "/models/a\"quote\\and\ttab\nnewline.gguf";

    const std::string out = plan_json(r);

    assert(has(out, "a\\\"quote\\\\and\\ttab\\nnewline.gguf"));
    // The raw byte sequence would end the string early; it must not appear.
    assert(!has(out, "a\"quote"));
}

static void test_an_unknown_available_ram_is_null_not_zero() {
    plan_report r = sample();
    r.available_mib   = 0;
    r.available_known = false;

    const std::string out = plan_json(r);

    assert(has(out, "\"available_mib\": null"));
    assert(has(out, "\"available_known\": false"));
}

static void test_a_clamp_carries_its_reason() {
    plan_report r = sample();
    r.clamped       = true;
    r.ctx_used      = 8192;
    r.ctx_requested = 32768;
    r.clamp_reason  = "the KV cache does not fit beside the model in RAM";

    const std::string out = plan_json(r);

    assert(has(out, "\"clamped\": true"));
    assert(has(out, "\"used\": 8192"));
    assert(has(out, "\"reason\": \"the KV cache does not fit beside the model in RAM\""));

    // The unclamped sample must not invent one.
    assert(has(plan_json(sample()), "\"reason\": null"));
}

static void test_control_characters_become_escapes() {
    plan_report r = sample();
    r.gpu_name = std::string("odd\x01name");

    const std::string out = plan_json(r);

    assert(has(out, "odd\\u0001name"));
}

int main(void) {
    test_the_whole_object_is_the_documented_one();
    test_the_schema_number_is_deliberate();
    test_the_object_balances();
    test_the_placement_is_reported_as_numbers();
    test_no_card_is_null_not_an_empty_object();
    test_a_path_is_escaped();
    test_an_unknown_available_ram_is_null_not_zero();
    test_a_clamp_carries_its_reason();
    test_control_characters_become_escapes();

    std::printf("test-vipera-plan-json: all tests passed\n");
    return 0;
}
