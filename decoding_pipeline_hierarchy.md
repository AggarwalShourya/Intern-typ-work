# Decoding Pipeline Feature Hierarchy — `nemo_hybrid/1/model.py`

## Overview

The model supports four optional per-request features that modify decoding behaviour:

| Input | Feature | Stage |
|-------|---------|-------|
| `LANG_TAGS` | Language Constraining | During decoding |
| `BLANK_PENALTY` | CTC Blank Penalty | During CTC decoding |
| `WORD_BOOST_PARAMS` | Word-level Language Boost | Post-decoding (step 1) |
| `CB_PARAMS` + `CONTEXT_GRAPH` | Keyword Boosting | Post-decoding (step 2) |

All features are **optional**. When none are supplied the pipeline is a plain RNNT/TDT/CTC decode.

---

## Stage-by-Stage Execution Order

### Stage 0 — Audio Preprocessing & Encoding
- Raw PCM → mel-filterbank features (preprocessor)
- Features → encoded representation (TRT or PyTorch encoder)
- Encoded → `log_probs` via CTC decoder head `(B, T_enc, vocab+1)`
- This stage is **unaffected** by any of the four features

---

### Stage 1 — Language Constraining (`LANG_TAGS`)

**What it does:**
Builds a hard additive mask of shape `(vocab,)`:
- `0.0` for every token that belongs to the requested languages + shared tokens + blank
- `-inf` for every other token (completely blocked)

**Where it's applied:**
- **RNNT / TDT path:** The `joint_after_projection` function is monkey-patched via `_masked_joint` context manager. The mask is added to the joint logits before the decoder runs. Out-of-language tokens have probability → 0 so they are *never emitted*.
- **CTC path (`_decode_ctc_batch`):** The mask is added to the CTC log-probs tensor before `ctc_decoder_predictions_tensor` runs.

**Relationship to other features:**
- Language constraining is applied **first**, before any text is produced.
- It does **not override** blank penalty — both are applied independently during CTC decoding.
- It **feeds into keyword boost**: after decoding, if lang_tags were active, keywords whose tokens are fully masked are silently dropped (they cannot have been in the acoustic signal).

---

### Stage 1b — CTC Blank Penalty (`BLANK_PENALTY`)

**What it does:**
Subtracts the given float value from the blank token log-prob at every frame, discouraging the CTC decoder from emitting blanks and encouraging non-blank token emission.

**Where it's applied:**
- **CTC path only** (`_decode_ctc_batch`): `masked_lp[i, :, blank_id] -= bp`
- **RNNT / TDT path:** This input is **ignored** for RNNT/TDT decoding. (For RNNT, blank penalty from `WORD_BOOST_PARAMS` is the relevant lever.)

**Relationship to other features:**
- Applied at the same CTC stage as language constraining; the two compose additively on the log-probs.
- **Distinct** from `WORD_BOOST_PARAMS.blank_penalty` (which serves the same conceptual purpose but only inside the word-level boost CTC re-decode step).

---

### Stage 2 — RNNT / CTC Decoding

Depending on `DECODING_STRATEGY` env var:

| Strategy | Decoder | Lang constraint applied via |
|----------|---------|---------------------------|
| `malsd_batch` (default) | RNNT beam | `_masked_joint` patch |
| `greedy_batch` | RNNT greedy | `_masked_joint` patch |
| `tdt` | TDT | `_masked_joint` patch |
| `ctc` | CTC greedy | log-prob mask in `_decode_ctc_batch` |

Output: `texts` (list of strings), `scores`, `log_probs_np` (raw CTC log-probs, always produced regardless of strategy).

---

### Stage 3 — Word-level Language Boost (`WORD_BOOST_PARAMS`)

**What it does (per-item, CPU):**
Runs an independent CTC greedy re-decode from the raw CTC log-probs with language-aware re-scoring:

1. Apply `blank_penalty` (from `WORD_BOOST_PARAMS`, default 0.5)
2. First-pass greedy argmax + CTC collapse → token sequence
3. Split collapsed tokens into word groups on the `▁` (word boundary) character
4. For each word, vote on the **dominant language** from its constituent tokens using the `token→langs` reverse lookup
5. Re-score that word's frame log-probs:
   - Add `lang_boost` (default 2.0) to tokens belonging to the dominant language
   - Subtract `lang_penalty` (default 0.0) from tokens belonging to *other* languages (shared/numerals/punctuation are never penalised)
6. Final CTC collapse on re-scored frames → new transcript

**Outputs two things:**
- `wb_text`: a new transcript string that **replaces** the RNNT/CTC decoded text
- `item_lp`: the **modified** log-prob matrix (with per-word boosts applied), passed on to keyword boost

**Relationship to other features:**
- Runs **after** main decoding (Stage 2) — it is pure post-processing on the CTC head output.
- **Does NOT use** the RNNT output; it re-decodes from CTC log-probs every time it is active.
- **Overrides** the Stage 2 transcript for that item.
- Uses the **same** `LANG_TOKEN_MAP` as language constraining (for `token→langs` lookup).
- The modified log-probs it produces become the input to keyword boost — they compose, not compete.
- Its `blank_penalty` parameter is **independent** of the `BLANK_PENALTY` input (different stage, different purpose).

---

### Stage 4 — Keyword Boosting (`CB_PARAMS` + `CONTEXT_GRAPH`)

**What it does:**
Runs a context-graph word-spotter over the CTC log-probs (possibly already modified by word-level boost) to detect pre-loaded keywords, then replaces the aligned spans in the transcript.

1. `run_word_spotter`: beam search over the `ContextGraphCTC` trie, returns spotted keyword hypotheses with time alignments and scores
2. Language guard: if `LANG_TAGS` were active, keywords whose tokens are fully blocked by the language mask are dropped
3. `merge_alignment_with_ws_hyps`: replaces matching spans in the CTC argmax alignment with the spotted keyword text

**Relationship to other features:**
- Runs **last**, after word-level boost.
- If word-level boost was active, keyword boost operates on the **language-biased log-probs** (not the raw encoder output), so keyword detection benefits from per-word language re-scoring.
- If language constraining is active, the keyword boost respects it: only keywords whose tokens survive the language mask are allowed through.
- Keyword boost **does not interfere** with blank penalty or language constraining — it is purely a post-processing transcript edit.
- Both `CB_PARAMS` **and** `CONTEXT_GRAPH` must be non-empty; if either is absent the step is skipped.

---

## Combined Interaction Summary

```
AUDIO
  │
  ▼
[Preprocessor → Encoder → CTC Decoder]       ← no feature influence here
  │
  ├─ log_probs_gpu (B, T, vocab+1)
  │
  ▼
[Stage 1: Language Constraining + Blank Penalty]
  │  LANG_TAGS → -inf mask on out-of-language tokens
  │  BLANK_PENALTY → subtract from blank log-prob
  │  (CTC path: both on log_probs; RNNT path: mask on joint logits only)
  ▼
[Stage 2: RNNT / CTC Decoding]
  │  → texts[], scores[], log_probs_np (raw CTC head output, always)
  ▼
[Stage 3: Word-level Language Boost]  ← if WORD_BOOST_PARAMS.enabled=true
  │  Input:  raw log_probs_np[i]
  │  Output: wb_text (replaces texts[i])
  │          item_lp (modified log-probs, passed to Stage 4)
  ▼
[Stage 4: Keyword Boosting]           ← if CB_PARAMS + CONTEXT_GRAPH present
  │  Input:  item_lp (word-boosted if Stage 3 ran, else raw)
  │          current texts[i] (word-boosted text if Stage 3 ran, else RNNT/CTC text)
  │  Output: boosted_text (replaces texts[i] where keywords are spotted)
  ▼
TRANSCRIPT output
```

---

## Override vs. Compose

| Pair | Interaction |
|------|------------|
| Language Constraining + Blank Penalty | **Compose** — both modify log-probs additively at the same stage |
| Language Constraining + Word Boost | **Compose** — both use the same `LANG_TOKEN_MAP`; word boost runs on log-probs that were already language-masked for CTC path |
| Language Constraining + Keyword Boost | **Compose with guard** — keywords that are fully blocked by lang mask are dropped before merge |
| Word Boost + Keyword Boost | **Pipeline** — word boost's modified log-probs are fed directly into keyword boost; they are additive rather than competing |
| Blank Penalty (`BLANK_PENALTY`) + Word Boost's `blank_penalty` | **Independent** — applied at different stages; both reduce blank emissions but act on different log-prob tensors |
| RNNT/CTC decode output vs. Word Boost output | **Override** — when word boost is active it re-decodes from CTC log-probs and fully replaces the RNNT/CTC transcript |

---

## Key Design Notes

- **Language token map** (`LANG_TOKEN_MAP`) is shared by all three language-aware features (constraining, word boost, keyword boost guard). It maps `{lang_name: [token_ids]}` and is built once lazily.
- **CTC log-probs are always computed**, even for RNNT/TDT strategies, because word boost and keyword boost always operate on CTC probabilities.
- **Mixed batches** (items with different `LANG_TAGS`) are handled correctly: RNNT/TDT runs item-by-item in that case; CTC and post-processing are always per-item.
- All four features default to "disabled" when the corresponding input tensor is absent or empty — the pipeline gracefully degrades to a plain decode.
