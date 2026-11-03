# `vipera` — the placement tool

Loads a GGUF, works out where it fits (VRAM, RAM, KV cache, context), and either
prints that plan or runs the model from it. The numbers it prints are measured
on the reference 4 GB card and explained next to the flag they belong to; the
comment above each decision in `vipera.cpp` is the record of why it exists.

```bash
vipera -m model.gguf --plan          # the plan, written for a person
vipera -m model.gguf --plan --json   # the same plan, as one JSON object
vipera -m model.gguf -p "hello"      # run it
```

## The plan as data

`--json` only applies to `--plan`; without it the tool exits `2` rather than
printing an object and then running, which would corrupt stdout for a caller
that pipes it. With it, **stdout is exactly one JSON object followed by a
newline** — llama.cpp's own logging goes to stderr, so a caller can read the
whole of stdout and parse it without filtering.

```bash
vipera -m qwen2.5-1.5b.gguf --plan --json | jq .placement
```

```json
{
  "schema": 1,
  "model": {"size_mib": 1059, "n_layer": 28, "n_head_kv": 2, "head_dim": 128,
            "train_ctx": 32768, "path": "/models/qwen2.5-1.5b.gguf"},
  "device": {"total_mib": 3730, "free_mib": 3674, "headroom_mib": 233,
             "name": "NVIDIA GeForce GTX 1650"},
  "placement": {"auto": true, "fit_failed": false, "ffn_split": false,
                "gpu_layers": 28, "ffn_on_cpu": 0},
  "kv": {"mib": 896, "lowered": false, "in_ram": false, "k": "f16", "v": "f16"},
  "memory": {"resident_mib": 0, "available_known": true, "available_mib": 13345},
  "context": {"requested": 32768, "used": 32768, "train_limit": 32768,
              "clamped": false, "limited_by_train": false, "reason": null},
  "threads": 4
}
```

(Formatting only: the real output is one line with no spaces after commas.)

`n_layer` appears once, under `model`: a second copy under `placement` could
only disagree with it.

**Null means "not known", never "zero".** `device` is `null` when there is no
card. `memory.available_mib` is `null` when `/proc/meminfo` could not be read —
in that case the RAM clamp was skipped, and a caller that reads `0` would
conclude "no RAM left" where the truth is "the number was not available".
`context.reason` is `null` when nothing was clamped. `available_known` is the
explicit form of the same fact, so a caller does not have to infer it from a
null.

### Versioning

`schema` is a number, currently `1`. It is bumped whenever a field changes
meaning, and a caller **must** refuse a version it does not know instead of
reading the same keys and guessing:

```python
plan = json.loads(subprocess.run([...], capture_output=True, text=True).stdout)
if plan["schema"] != 1:
    raise SystemExit(f"vipera plan schema {plan['schema']} is newer than this caller")
```

`tests/test-vipera-plan-json.cpp` freezes the entire object as a golden string,
so adding, renaming or retyping a field fails the test until the fixture, the
schema number and this page are updated together.

## Placement ladders

`-ncffn auto` and a `-ngl` split that does not load are searched by bisection,
one child process per rung (`place.h` holds the rule, `test-vipera-place.cpp`
the test). A rung that fails for a reason unrelated to placement — a bad model
path, a build without the flag — stops the search instead of being read as "does
not fit".
