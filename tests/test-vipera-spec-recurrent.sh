#!/usr/bin/env bash
# Does --spec refuse a model whose attention cannot roll the KV cache back?
#
# A recurrent or hybrid stack cannot drop part of a sequence, so the tool has to
# notice before it decodes, print its note, and finish the run plainly instead of
# dying half way through it. That guard is the only code in front of
# llama_memory_seq_rm()'s failure branch, so it is what this test covers.
#
# There is no synthetic substitute for the fixture: a vocab-only GGUF has no KV
# cache at all (llama-context.cpp skips memory.reset), and seq_rm on a null
# memory returns true - the branch is unreachable without a real model.
#
# Fixture: point VIPERA_SPEC_RECURRENT_MODEL at a GGUF whose attention is
# recurrent or hybrid (llama_model_is_recurrent() / llama_model_is_hybrid()).
# Without it the test skips, so a default build stays green.
#
#   VIPERA_SPEC_RECURRENT_MODEL=/models/mamba-130m.gguf \
#       ctest -R test-vipera-spec-recurrent --output-on-failure

set -u

BIN=${1:-}
MODEL=${VIPERA_SPEC_RECURRENT_MODEL:-}

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "skip: set VIPERA_SPEC_RECURRENT_MODEL to a recurrent/hybrid GGUF"
    exit 77
fi

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "skip: no vipera binary at '${BIN}'"
    exit 77
fi

ERR=$(mktemp) || exit 1
trap 'rm -f "$ERR"' EXIT

OUT=$("$BIN" -m "$MODEL" -c 256 -n 8 -p "The" --spec ngram-simple 2>"$ERR")
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "FAIL: --spec run exited $STATUS on a recurrent model"
    cat "$ERR"
    exit 1
fi

if ! grep -q "run decodes plainly" "$ERR"; then
    echo "FAIL: the recurrent guard did not fire, so --spec ran on a model whose"
    echo "      cache cannot be rolled back. stderr was:"
    cat "$ERR"
    exit 1
fi

if ! printf '%s' "$OUT" | grep -q "prompt"; then
    echo "FAIL: the run produced no report - it did not decode plainly"
    exit 1
fi

echo "ok: --spec disabled itself on a recurrent model and the run completed"
