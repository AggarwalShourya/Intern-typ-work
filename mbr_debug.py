from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import torch

try:
    from rapidfuzz.distance import Levenshtein as _lev
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


class MBRLossFunction(str, Enum):
    EDIT_DISTANCE   = "edit_distance"
    TOKEN_OVERLAP   = "token_overlap"
    WORD_ERROR_RATE = "word_error_rate"


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance_normalised(hyp, ref, hyp_len, ref_len) -> float:
    if hyp_len == 0 and ref_len == 0:
        return 0.0
    if hyp_len == 0 or ref_len == 0:
        return 1.0
    if _HAS_RAPIDFUZZ:
        return _lev.distance(hyp[:hyp_len].tolist(),
                             ref[:ref_len].tolist()) / max(hyp_len, ref_len)
    h_list = hyp[:hyp_len].tolist()
    r_list = ref[:ref_len].tolist()
    dp = list(range(ref_len + 1))
    for i in range(1, hyp_len + 1):
        prev  = dp[0]; dp[0] = i
        for j in range(1, ref_len + 1):
            temp = dp[j]
            dp[j] = prev if h_list[i-1] == r_list[j-1] \
                    else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[ref_len] / max(hyp_len, ref_len)


# ---------------------------------------------------------------------------
# FIX 1: Vectorised sequence extraction
#
# Problem: _resolve_tree() walked backwards token-by-token in Python.
# For seq_len=82, beam=8, batch=8 → 5248 Python loop iterations just
# for extraction, before any edit distance work.
#
# Solution: reconstruct all K beams for one batch item in one tensor op.
#
# How it works:
#   transcript_wb[b, k, t]          = token emitted at step t on beam k's path
#   transcript_wb_prev_ptr[b, k, t] = which beam was the ancestor at step t
#
# We follow ALL K pointer chains simultaneously:
#   Step 1: start with cur_beam = [0, 1, 2, ..., K-1]
#   Step 2: at each timestep t (going backwards), gather the token that
#           cur_beam[k] emitted at step t → that's the token for beam k at t
#   Step 3: update cur_beam[k] = prev_ptr[cur_beam[k], t]
#   Step 4: collect tokens, reverse, strip blanks
#
# This replaces K separate Python loops with T vectorised gather ops.
# T is typically 30-100, K=8 → ~8x fewer Python iterations, and each
# "iteration" is a GPU/CPU tensor op instead of a Python function call.
# ---------------------------------------------------------------------------

def _extract_sequences_vectorised(batched_hyps):
    """
    Vectorised replacement for the nested _resolve_tree loop.

    For each batch item, reconstructs all K beam sequences in parallel
    using tensor gather ops instead of K separate Python pointer walks.

    Returns:
        sequences : list[B] of list[K] of 1-D LongTensors (clean, no blanks)
        lengths   : list[B] of list[K] of int
    """
    B     = batched_hyps.batch_size
    K     = batched_hyps.beam_size
    blank = batched_hyps.blank_index

    # transcript_wb      : [B, K, T]  — token emitted at each step
    # transcript_wb_prev_ptr : [B, K, T]  — ancestor beam pointer
    twb  = batched_hyps.transcript_wb           # [B, K, T]
    tptr = batched_hyps.transcript_wb_prev_ptr  # [B, K, T]

    sequences = []
    lengths   = []

    for b in range(B):
        # Max steps across all beams for this batch item
        wb_lengths = batched_hyps.current_lengths_wb[b]   # [K]
        max_len    = int(wb_lengths.max().item())

        if max_len == 0:
            sequences.append([torch.zeros(0, dtype=torch.long)] * K)
            lengths.append([0] * K)
            continue

        # cur_beam[k] = which beam we're currently following for hypothesis k
        # Starts as [0, 1, 2, ..., K-1]
        cur_beam = torch.arange(K, device=twb.device)     # [K]

        # collected[k, t] will hold the token for beam k at position t
        collected = torch.zeros(K, max_len, dtype=torch.long)

        # Walk backwards from max_len-1 to 0
        for t in range(max_len - 1, -1, -1):
            # Gather token: for each beam k, what did cur_beam[k] emit at step t?
            # twb[b, :, t] shape: [K]  — tokens at step t for every beam
            tokens_at_t = twb[b, :, t]                    # [K]
            gathered_tokens = tokens_at_t[cur_beam]       # [K] — follow cur_beam ptr

            collected[:, t] = gathered_tokens

            # Update cur_beam: follow the pointer backwards
            ptrs_at_t = tptr[b, :, t]                     # [K]
            cur_beam  = ptrs_at_t[cur_beam].clamp(min=0)  # [K]

        # Now strip blanks and sentinels per beam
        batch_seqs    = []
        batch_lengths = []

        for k in range(K):
            actual_len = int(wb_lengths[k].item())
            if actual_len == 0:
                batch_seqs.append(torch.zeros(0, dtype=torch.long))
                batch_lengths.append(0)
                continue

            seq  = collected[k, :actual_len]
            mask = (seq >= 0) & (seq != blank)
            clean = seq[mask]
            batch_seqs.append(clean)
            batch_lengths.append(int(clean.shape[0]))

        sequences.append(batch_seqs)
        lengths.append(batch_lengths)

    return sequences, lengths


# ---------------------------------------------------------------------------
# FIX 2: Adaptive temperature — based on actual observed spread
#
# Problem: a fixed temperature (or none) fails to handle the wide variance
# in score spreads across batches (31 → 87 in the diagnostic output).
# Batches with spread=31 already collapse to 0.787 on beam 0.
#
# Solution: compute temperature per batch item from the actual active score
# spread, targeting a max softmax weight of ~0.4 (prevents any single beam
# from dominating while still respecting the model's confidence ordering).
#
# target_max_weight = 0.4 means no beam gets more than 40% of the vote.
# Formula derived from: softmax_max ≈ exp(spread/T) / K
# Solving for T: T = spread / (ln(target * K) ... bounded to sane range.
# ---------------------------------------------------------------------------

def _compute_scores_adaptive_temperature(
    batched_hyps,
    score_norm:       bool  = True,    # length-normalise first (recommended)
    target_max_weight: float = 0.40,   # no beam should get more than this
    min_temp:         float = 1.0,
    max_temp:         float = 20.0,
) -> torch.Tensor:
    """
    Per-batch-item temperature scaling.

    Rather than a global temperature, we compute the temperature each batch
    item needs so that the dominant beam's softmax weight stays at or below
    target_max_weight. This handles the wide spread variance automatically.
    """
    import math

    scores   = batched_hyps.scores.float().clone()        # [B, K]
    inactive = scores < -1e8

    if score_norm:
        lengths = batched_hyps.current_lengths_nb.float() + 1.0
        scores  = torch.where(inactive, scores, scores / lengths)

    B, K = scores.shape
    result = scores.clone()

    for b in range(B):
        active_mask   = ~inactive[b]
        active_scores = scores[b][active_mask]
        if active_scores.numel() < 2:
            continue

        spread = float((active_scores.max() - active_scores.min()).item())
        if spread < 1e-6:
            continue

        # Temperature needed so softmax_max ≈ target_max_weight
        # Approximate: softmax_max ≈ exp(spread/T) / (1 + (K_active-1)*exp(0))
        # Simpler bound: T = spread / log(1/target - 1 + 1/K_active) clipped
        K_active = int(active_mask.sum().item())
        try:
            # From logit spread: if top score is `spread` above others,
            # temp needed = spread / ln((1-p)/p * K_active ... approx:
            temp = spread / math.log(max(2.0, (1.0 / target_max_weight - 1.0) * K_active))
        except (ValueError, ZeroDivisionError):
            temp = 1.0

        temp = max(min_temp, min(max_temp, temp))

        # Apply temperature only to active scores
        result[b] = torch.where(inactive[b], scores[b], scores[b] / temp)

    return result


# ---------------------------------------------------------------------------
# Loss matrix builders
# ---------------------------------------------------------------------------

def _edit_distance_loss_matrix(
    sequences_b: List[torch.Tensor],
    lengths_b:   List[int],
    inactive_b:  torch.Tensor,
) -> torch.Tensor:
    K = len(sequences_b)
    loss_matrix = torch.zeros(K, K)

    for i in range(K):
        if inactive_b[i]:
            loss_matrix[i, :] = 1.0
            loss_matrix[:, i] = 1.0
            continue
        for j in range(i + 1, K):
            if inactive_b[j]:
                loss_matrix[i, j] = loss_matrix[j, i] = 1.0
                continue
            hyp_i, len_i = sequences_b[i], lengths_b[i]
            hyp_j, len_j = sequences_b[j], lengths_b[j]
            # Fast path: identical sequences
            if (len_i == len_j and len_i > 0 and
                    torch.equal(hyp_i[:len_i], hyp_j[:len_j])):
                continue
            val = _edit_distance_normalised(hyp_i, hyp_j, len_i, len_j)
            loss_matrix[i, j] = val
            loss_matrix[j, i] = val
    return loss_matrix


def _token_overlap_loss_matrix_vectorised(
    sequences: List[torch.Tensor],
    lengths:   List[int],
    inactive:  torch.Tensor,
    device:    torch.device,
) -> torch.Tensor:
    K = len(sequences)
    if K == 0:
        return torch.zeros(K, K)
    max_tok = 0
    for seq, length in zip(sequences, lengths):
        if length > 0:
            max_tok = max(max_tok, int(seq[:length].max().item()))
    vocab_size = max_tok + 1
    if vocab_size == 0:
        return torch.zeros(K, K)
    multi_hot = torch.zeros(K, vocab_size, device=device)
    for k, (seq, length) in enumerate(zip(sequences, lengths)):
        if length > 0 and not inactive[k]:
            indices = seq[:length].clamp(0, vocab_size - 1)
            multi_hot[k].scatter_(0, indices, 1.0)
    intersection = multi_hot @ multi_hot.t()
    set_sizes    = multi_hot.sum(dim=1)
    union        = set_sizes.unsqueeze(1) + set_sizes.unsqueeze(0) - intersection
    return torch.where(union > 0,
                       1.0 - intersection / union.clamp(min=1e-9),
                       torch.zeros_like(union))


# ---------------------------------------------------------------------------
# MBR selection
# ---------------------------------------------------------------------------

def _mbr_select(batched_hyps, loss_fn, scores_for_weights) -> torch.Tensor:
    batch_size = batched_hyps.batch_size
    beam_size  = batched_hyps.beam_size
    device     = batched_hyps.device

    inactive    = batched_hyps.scores.float() < -1e8
    masked      = scores_for_weights.masked_fill(inactive, float('-inf'))
    ref_weights = torch.softmax(masked, dim=-1)

    sequences, lengths = _extract_sequences_vectorised(batched_hyps)

    best_beam_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        inactive_b = inactive[b]

        if loss_fn == MBRLossFunction.TOKEN_OVERLAP:
            loss_matrix = _token_overlap_loss_matrix_vectorised(
                sequences[b], lengths[b], inactive_b, device
            )
            inactive_cpu = inactive_b.cpu()
            loss_matrix[inactive_cpu, :] = 1.0
            loss_matrix[:, inactive_cpu] = 1.0
            loss_matrix.fill_diagonal_(0.0)
        else:
            loss_matrix = _edit_distance_loss_matrix(
                sequences[b], lengths[b], inactive_b
            )

        weights_b = ref_weights[b].cpu()
        expected  = (loss_matrix * weights_b.unsqueeze(0)).sum(dim=-1)
        expected[inactive_b.cpu()] = float('inf')
        best_beam_indices[b] = expected.argmin()

    return best_beam_indices


# ---------------------------------------------------------------------------
# Permutation building — vectorised
# ---------------------------------------------------------------------------

def _build_permutation(best_indices, beam_size, device):
    B    = best_indices.shape[0]
    K    = beam_size
    all_k  = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
    winner = best_indices.unsqueeze(1)
    sort_key = (~(all_k == winner)).long() * K + all_k
    _, perm  = sort_key.sort(dim=1, stable=True)
    inv_perm = torch.zeros(B, K, dtype=torch.long, device=device)
    inv_perm.scatter_(1, perm,
                      torch.arange(K, device=device).unsqueeze(0).expand(B, -1))
    return perm, inv_perm


# ---------------------------------------------------------------------------
# Diagnostic (kept, but threshold fixed to 30 based on observed data)
# ---------------------------------------------------------------------------

def diagnose_mbr_inputs(batched_hyps, score_norm: bool = True):
    print("\n" + "="*60)
    print("MBR DIAGNOSTIC REPORT")
    print("="*60)

    B, K   = batched_hyps.batch_size, batched_hyps.beam_size
    blank  = batched_hyps.blank_index
    scores_raw = batched_hyps.scores.float()
    inactive   = scores_raw < -1e8

    print(f"\n[CONFIG] batch={B}, beam={K}, blank_index={blank}")
    print(f"[CONFIG] rapidfuzz available: {_HAS_RAPIDFUZZ}")

    print("\n[SCORES] Raw score range per batch item:")
    for b in range(B):
        active = scores_raw[b][~inactive[b]]
        if active.numel() == 0:
            print(f"  batch {b}: ALL INACTIVE")
            continue
        spread = float((active.max() - active.min()).item())
        flag   = "  ⚠️  WEIGHT COLLAPSE RISK" if spread > 30 else ""  # fixed threshold
        print(f"  batch {b}: min={active.min():.2f}  max={active.max():.2f}"
              f"  spread={spread:.2f}{flag}")

    scores_for_w = _compute_scores_adaptive_temperature(batched_hyps, score_norm)
    masked   = scores_for_w.masked_fill(inactive, float('-inf'))
    weights  = torch.softmax(masked, dim=-1)

    print("\n[WEIGHTS] Softmax weights after adaptive temperature:")
    for b in range(B):
        w     = weights[b]
        max_w = w.max().item()
        flag  = f"  ⚠️  still dominated ({max_w:.1%})" if max_w > 0.6 else ""
        print(f"  batch {b}: {[f'{x:.3f}' for x in w.tolist()]}{flag}")

    sequences, lengths = _extract_sequences_vectorised(batched_hyps)
    print("\n[SEQUENCES] Extracted (first 10 tokens shown):")
    for b in range(B):
        print(f"  batch {b}:")
        for k in range(K):
            preview = sequences[b][k][:10].tolist() if lengths[b][k] > 0 else []
            tag = "  [INACTIVE]" if inactive[b, k] else ""
            print(f"    beam {k}: len={lengths[b][k]}  tokens={preview}"
                  f"{'...' if lengths[b][k] > 10 else ''}{tag}")

    print("\n[LOSS MATRIX] Edit distance for batch 0:")
    b = 0
    rows = []
    for i in range(K):
        row = []
        for j in range(K):
            if i == j:
                row.append(0.0)
            elif inactive[b,i] or inactive[b,j]:
                row.append(-1.0)
            else:
                row.append(round(_edit_distance_normalised(
                    sequences[b][i], sequences[b][j],
                    lengths[b][i],   lengths[b][j]), 3))
        rows.append(row)
    for i, row in enumerate(rows):
        print(f"  beam {i}: {row}")

    print("\n[EXPECTED LOSS] Per beam for batch 0:")
    lm = torch.tensor(rows)
    w0 = weights[0].cpu()
    exp = (lm * w0.unsqueeze(0)).sum(dim=-1)
    exp[inactive[0].cpu()] = float('inf')
    winner = int(exp.argmin().item())
    for k in range(K):
        tag = " ← MBR winner" if k == winner else ""
        tag += " [INACTIVE]" if inactive[0,k] else ""
        print(f"  beam {k}: expected_loss={exp[k]:.4f}{tag}")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# MBRDecoder
# ---------------------------------------------------------------------------

class MBRDecoder:
    def __init__(
        self,
        beam_decoder,
        loss_fn:           str | MBRLossFunction = MBRLossFunction.EDIT_DISTANCE,
        score_norm:        bool  = True,    # recommended True based on diagnostic
        target_max_weight: float = 0.25,    # adaptive temperature target
        diagnose:          bool  = False,
    ):
        self.beam_decoder      = beam_decoder
        self.loss_fn           = MBRLossFunction(loss_fn)
        self.score_norm        = score_norm
        self.target_max_weight = target_max_weight
        self.diagnose          = diagnose

    def _reorder_beam_to_front(self, batched_hyps, best_indices):
        beam_size = batched_hyps.beam_size
        device    = batched_hyps.device
        perm, inv_perm = _build_permutation(best_indices, beam_size, device)

        def _reorder(t):
            if t.dim() == 2:
                return torch.gather(t, 1, perm)
            if t.dim() == 3:
                return torch.gather(t, 1, perm[:, :, None].expand_as(t))
            return t

        batched_hyps.scores        = _reorder(batched_hyps.scores)
        batched_hyps.transcript_wb = _reorder(batched_hyps.transcript_wb)

        prev_ptr = _reorder(batched_hyps.transcript_wb_prev_ptr)
        if prev_ptr.numel() > 0:
            T          = prev_ptr.shape[2]
            valid_mask = prev_ptr >= 0
            lookup     = inv_perm.unsqueeze(-1).expand(-1, -1, T)
            remapped   = torch.gather(lookup, 1, prev_ptr.clamp(min=0))
            batched_hyps.transcript_wb_prev_ptr = torch.where(
                valid_mask, remapped, prev_ptr)
        else:
            batched_hyps.transcript_wb_prev_ptr = prev_ptr

        batched_hyps.timestamps           = _reorder(batched_hyps.timestamps)
        batched_hyps.current_lengths_nb   = _reorder(batched_hyps.current_lengths_nb)
        batched_hyps.current_lengths_wb   = _reorder(batched_hyps.current_lengths_wb)
        batched_hyps.last_label           = _reorder(batched_hyps.last_label)
        batched_hyps.transcript_hash      = _reorder(batched_hyps.transcript_hash)
        batched_hyps.next_timestamp       = _reorder(batched_hyps.next_timestamp)
        batched_hyps.last_timestamp_lasts = _reorder(batched_hyps.last_timestamp_lasts)
        if batched_hyps.store_prefix_hashes:
            batched_hyps.transcript_prefix_hash = _reorder(
                batched_hyps.transcript_prefix_hash)

    def run_mbr(self, batched_hyps):
        if self.diagnose:
            diagnose_mbr_inputs(batched_hyps, self.score_norm)
        scores_for_weights = _compute_scores_adaptive_temperature(
            batched_hyps, self.score_norm, self.target_max_weight
        )
        best_indices = _mbr_select(batched_hyps, self.loss_fn, scores_for_weights)
        print("MBR winners:", best_indices.tolist())
        self._reorder_beam_to_front(batched_hyps, best_indices)
        return batched_hyps

    def __call__(self, x, out_len):
        batched_hyps = self.beam_decoder(x, out_len)
        if self.diagnose:
            diagnose_mbr_inputs(batched_hyps, self.score_norm)
        scores_for_weights = _compute_scores_adaptive_temperature(
            batched_hyps, self.score_norm, self.target_max_weight
        )
        best_indices = _mbr_select(batched_hyps, self.loss_fn, scores_for_weights)
        self._reorder_beam_to_front(batched_hyps, best_indices)
        return batched_hyps

    def __getattr__(self, name):
        if name in ("beam_decoder", "loss_fn", "score_norm",
                    "target_max_weight", "diagnose"):
            raise AttributeError(name)
        return getattr(self.beam_decoder, name)