// vipera: the plan as data instead of as prose.
//
// `--plan` prints a block written for a person: aligned columns, ~ signs, and
// sentences that explain a number ("nothing fit the card; every layer stays in
// RAM"). A caller - the Python planner, a bench script, a comparison against
// another machine - cannot read any of that without parsing English, so this is
// the same plan as one JSON object, produced from the same values the printed
// block uses.
//
// It lives in its own header, like accept.h and place.h, so the escaping and the
// null-not-empty-string rule can be tested without a model, a GPU, or a run.

#pragma once

#include <cstddef>
#include <cstdio>
#include <string>

//: Bumped when a field changes meaning. A caller that reads this refuses a
//: version it does not know instead of misreading the same key.
inline constexpr int PLAN_JSON_SCHEMA = 1;

struct plan_report {
    // -- the model, from its header --------------------------------------
    std::string path;
    size_t      model_mib  = 0;
    int         n_layer    = 0;
    int         n_head_kv  = 0;
    int         head_dim   = 0;
    int         train_ctx  = 0;

    // -- the card, empty when there is none ------------------------------
    bool        has_gpu        = false;
    std::string gpu_name;
    size_t      gpu_total_mib  = 0;
    size_t      gpu_free_mib   = 0;
    size_t      headroom_mib   = 0;

    // -- placement -------------------------------------------------------
    bool auto_placement = false;  // the fit chose it, not -ngl
    bool fit_failed     = false;  // nothing fit; the whole stack is in RAM
    bool ffn_split      = false;  // -ncffn: attention on the card, FFN in RAM
    int  gpu_layers     = 0;      // layers whose attention (and KV) is on the card
    int  ffn_on_cpu     = 0;      // -ncffn count; 0 unless ffn_split

    // -- KV cache --------------------------------------------------------
    std::string kv_k            = "f16";
    std::string kv_v            = "f16";
    size_t      kv_mib          = 0;
    bool        kv_lowered      = false;  // f16 did not fit, q8_0 was used
    bool        kv_in_ram       = false;  // -nkvo

    // -- memory ----------------------------------------------------------
    size_t resident_mib    = 0;      // weight bytes that stay in system RAM
    size_t available_mib   = 0;
    bool   available_known = false;  // false: /proc/meminfo unreadable, no clamp

    // -- context ---------------------------------------------------------
    int         ctx_requested   = 0;
    int         ctx_used        = 0;
    bool        clamped         = false;
    bool        limited_by_train = false;
    std::string clamp_reason;  // empty when not clamped

    int threads = 1;
};

// Append `s` as a JSON string. Only what JSON requires is escaped - quote,
// backslash, and the controls below 0x20; bytes of UTF-8 pass through, which is
// what a JSON reader expects.
inline void plan_json_string(std::string & out, const std::string & s) {
    out += '"';
    for (const char ch : s) {
        const unsigned char c = (unsigned char) ch;
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char esc[8];
                    std::snprintf(esc, sizeof(esc), "\\u%04x", (unsigned) c);
                    out += esc;
                } else {
                    out += ch;
                }
        }
    }
    out += '"';
}

// An empty string is written as null, not as "": "no card" and "a card with an
// empty name" are different answers, and a caller should not have to guess.
inline void plan_json_optional(std::string & out, const std::string & s) {
    if (s.empty()) {
        out += "null";
    } else {
        plan_json_string(out, s);
    }
}

inline void plan_json_num(std::string & out, const char * key, long long value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "\"%s\": %lld", key, value);
    out += buf;
}

inline void plan_json_bool(std::string & out, const char * key, bool value) {
    out += '"';
    out += key;
    out += "\": ";
    out += value ? "true" : "false";
}

// One JSON object, newline-terminated, ready to be written to stdout.
inline std::string plan_json(const plan_report & r) {
    std::string out = "{\"schema\": ";
    out += std::to_string(PLAN_JSON_SCHEMA);

    out += ", \"model\": {";
    plan_json_num(out, "size_mib", (long long) r.model_mib);
    out += ", ";
    plan_json_num(out, "n_layer", r.n_layer);
    out += ", ";
    plan_json_num(out, "n_head_kv", r.n_head_kv);
    out += ", ";
    plan_json_num(out, "head_dim", r.head_dim);
    out += ", ";
    plan_json_num(out, "train_ctx", r.train_ctx);
    out += ", \"path\": ";
    plan_json_string(out, r.path);
    out += "}";

    out += ", \"device\": ";
    if (r.has_gpu) {
        out += "{";
        plan_json_num(out, "total_mib", (long long) r.gpu_total_mib);
        out += ", ";
        plan_json_num(out, "free_mib", (long long) r.gpu_free_mib);
        out += ", ";
        plan_json_num(out, "headroom_mib", (long long) r.headroom_mib);
        out += ", \"name\": ";
        plan_json_string(out, r.gpu_name);
        out += "}";
    } else {
        out += "null";
    }

    out += ", \"placement\": {";
    plan_json_bool(out, "auto", r.auto_placement);
    out += ", ";
    plan_json_bool(out, "fit_failed", r.fit_failed);
    out += ", ";
    plan_json_bool(out, "ffn_split", r.ffn_split);
    out += ", ";
    plan_json_num(out, "gpu_layers", r.gpu_layers);
    out += ", ";
    // No n_layer here: it is a property of the model, already reported under
    // "model". A second copy could only disagree with the first, and a caller
    // would then have to decide which of the two to believe.
    plan_json_num(out, "ffn_on_cpu", r.ffn_on_cpu);
    out += "}";

    out += ", \"kv\": {";
    plan_json_num(out, "mib", (long long) r.kv_mib);
    out += ", ";
    plan_json_bool(out, "lowered", r.kv_lowered);
    out += ", ";
    plan_json_bool(out, "in_ram", r.kv_in_ram);
    out += ", \"k\": ";
    plan_json_string(out, r.kv_k);
    out += ", \"v\": ";
    plan_json_string(out, r.kv_v);
    out += "}";

    out += ", \"memory\": {";
    plan_json_num(out, "resident_mib", (long long) r.resident_mib);
    out += ", ";
    plan_json_bool(out, "available_known", r.available_known);
    out += ", \"available_mib\": ";
    // Unknown is null, not 0: a caller that compares 0 would read "no RAM left"
    // where the truth is "the number was not available".
    if (r.available_known) {
        out += std::to_string(r.available_mib);
    } else {
        out += "null";
    }
    out += "}";

    out += ", \"context\": {";
    plan_json_num(out, "requested", r.ctx_requested);
    out += ", ";
    plan_json_num(out, "used", r.ctx_used);
    out += ", ";
    plan_json_num(out, "train_limit", r.train_ctx);
    out += ", ";
    plan_json_bool(out, "clamped", r.clamped);
    out += ", ";
    plan_json_bool(out, "limited_by_train", r.limited_by_train);
    out += ", \"reason\": ";
    plan_json_optional(out, r.clamp_reason);
    out += "}";

    out += ", ";
    plan_json_num(out, "threads", r.threads);
    out += "}\n";
    return out;
}
