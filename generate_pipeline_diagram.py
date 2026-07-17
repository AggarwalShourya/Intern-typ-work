"""
Generate a visual diagram of the nemo_hybrid decoding pipeline feature hierarchy.
Saves: decoding_pipeline_hierarchy.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(16, 22))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── Color palette ──────────────────────────────────────────────────────────────
C = {
    "bg":       "#0f1117",
    "card":     "#1e2130",
    "border":   "#2e3250",
    "audio":    "#3b82f6",   # blue
    "encode":   "#6366f1",   # indigo
    "lang":     "#f59e0b",   # amber  — language constraining
    "blank":    "#f97316",   # orange — blank penalty
    "decode":   "#10b981",   # emerald — RNNT/CTC decode
    "wb":       "#8b5cf6",   # violet — word boost
    "kb":       "#ec4899",   # pink   — keyword boost
    "output":   "#22d3ee",   # cyan
    "arrow":    "#94a3b8",
    "text":     "#f1f5f9",
    "subtext":  "#94a3b8",
    "tag":      "#334155",
}


def rounded_box(ax, x, y, w, h, color, border_color=None, radius=0.4, alpha=1.0):
    bdr = border_color or color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=2, edgecolor=bdr,
        facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    return box


def label(ax, x, y, text, size=11, color="#f1f5f9", bold=False, ha="center", va="center"):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=size, color=color,
            ha=ha, va=va, fontweight=weight, zorder=5,
            fontfamily="monospace")


def arrow(ax, x1, y1, x2, y2, color="#94a3b8", lw=2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=lw,
            mutation_scale=18,
        ),
        zorder=4,
    )


def dashed_arrow(ax, x1, y1, x2, y2, color="#94a3b8", lw=1.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=lw,
            linestyle="dashed",
            mutation_scale=14,
        ),
        zorder=4,
    )


def tag_box(ax, x, y, text, color):
    rounded_box(ax, x, y, 3.4, 0.55, color, border_color=color, radius=0.2, alpha=0.25)
    label(ax, x + 1.7, y + 0.275, text, size=9, color=color, bold=True)


# ── Title ──────────────────────────────────────────────────────────────────────
label(ax, 8, 21.4, "nemo_hybrid Decoding Pipeline — Feature Hierarchy",
      size=15, bold=True, color=C["text"])
label(ax, 8, 21.0, "model_repo/nemo_hybrid/1/model.py",
      size=9, color=C["subtext"])

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN LAYOUT
#   Left column  (x 0.4 .. 7.6):  main pipeline boxes
#   Right column (x 8.4 .. 15.6): feature detail / legend
# ══════════════════════════════════════════════════════════════════════════════

LEFT_X = 0.4
LEFT_W = 7.2
MID_X  = LEFT_X + LEFT_W / 2   # 4.0

RIGHT_X = 8.4
RIGHT_W = 7.0

# ── Stage boxes (top → bottom) ─────────────────────────────────────────────────
#  y positions (top of box):
Y_AUDIO   = 19.8
Y_ENCODE  = 17.8
Y_LANG    = 15.4
Y_DECODE  = 13.0
Y_WB      = 10.4
Y_KB      = 7.6
Y_OUT     = 5.2

BOX_H     = 1.5

# ─── AUDIO ───────────────────────────────────────────────────────────────────
rounded_box(ax, LEFT_X, Y_AUDIO, LEFT_W, BOX_H, C["card"], C["audio"])
label(ax, MID_X, Y_AUDIO + 1.0, "AUDIO INPUT", size=12, bold=True, color=C["audio"])
label(ax, MID_X, Y_AUDIO + 0.5, "(B, T)  INT16  raw PCM", size=9, color=C["subtext"])

# ─── ENCODE ──────────────────────────────────────────────────────────────────
rounded_box(ax, LEFT_X, Y_ENCODE, LEFT_W, BOX_H, C["card"], C["encode"])
label(ax, MID_X, Y_ENCODE + 1.0, "STAGE 0 — Preprocessing + Encoding", size=11, bold=True, color=C["encode"])
label(ax, MID_X, Y_ENCODE + 0.58, "Preprocessor  →  Encoder (TRT-FP16 or PyTorch AMP)", size=9, color=C["subtext"])
label(ax, MID_X, Y_ENCODE + 0.22, "→  CTC decoder head  →  log_probs (B, T_enc, vocab+1)", size=9, color=C["subtext"])

# ─── LANG CONSTRAINT ─────────────────────────────────────────────────────────
rounded_box(ax, LEFT_X, Y_LANG, LEFT_W, 1.9, C["card"], C["lang"])
label(ax, MID_X, Y_LANG + 1.55, "STAGE 1 — Language Constraining + Blank Penalty", size=11, bold=True, color=C["lang"])
label(ax, MID_X, Y_LANG + 1.1, "LANG_TAGS → -inf mask on out-of-language tokens", size=9, color=C["subtext"])
label(ax, MID_X, Y_LANG + 0.72, "BLANK_PENALTY → subtract from blank log-prob (CTC only)", size=9, color=C["subtext"])
label(ax, MID_X, Y_LANG + 0.34, "RNNT/TDT: mask patched into joint_after_projection", size=9, color=C["subtext"])

# ─── DECODE ──────────────────────────────────────────────────────────────────
rounded_box(ax, LEFT_X, Y_DECODE, LEFT_W, 1.7, C["card"], C["decode"])
label(ax, MID_X, Y_DECODE + 1.35, "STAGE 2 — RNNT / TDT / CTC Decoding", size=11, bold=True, color=C["decode"])
label(ax, MID_X, Y_DECODE + 0.9, "Outputs: texts[], scores[], log_probs_np (always CTC)", size=9, color=C["subtext"])
label(ax, MID_X, Y_DECODE + 0.52, "Strategy: malsd_batch | greedy_batch | tdt | ctc", size=9, color=C["subtext"])
label(ax, MID_X, Y_DECODE + 0.18, "CTC log-probs always produced (needed by later stages)", size=9, color=C["subtext"])

# ─── WORD BOOST ──────────────────────────────────────────────────────────────
rounded_box(ax, LEFT_X, Y_WB, LEFT_W, 2.2, C["card"], C["wb"])
label(ax, MID_X, Y_WB + 1.9, "STAGE 3 — Word-level Language Boost  [WORD_BOOST_PARAMS]", size=11, bold=True, color=C["wb"])
label(ax, MID_X, Y_WB + 1.47, "Input: raw CTC log_probs_np[i]", size=9, color=C["subtext"])
label(ax, MID_X, Y_WB + 1.1,  "1. Apply blank_penalty (wb)  2. Greedy CTC collapse", size=9, color=C["subtext"])
label(ax, MID_X, Y_WB + 0.74, "3. Split tokens into words  4. Vote dominant language/word", size=9, color=C["subtext"])
label(ax, MID_X, Y_WB + 0.38, "5. Boost dominant-lang tokens  6. Re-decode  → new text", size=9, color=C["subtext"])
label(ax, MID_X, Y_WB + 0.06, "Outputs: wb_text (replaces Stage 2 text) + modified item_lp", size=9, color=C["wb"])

# ─── KEYWORD BOOST ───────────────────────────────────────────────────────────
rounded_box(ax, LEFT_X, Y_KB, LEFT_W, 2.0, C["card"], C["kb"])
label(ax, MID_X, Y_KB + 1.7, "STAGE 4 — Keyword Boosting  [CB_PARAMS + CONTEXT_GRAPH]", size=11, bold=True, color=C["kb"])
label(ax, MID_X, Y_KB + 1.28, "Input: item_lp (word-boosted if Stage 3 ran, else raw)", size=9, color=C["subtext"])
label(ax, MID_X, Y_KB + 0.92, "1. run_word_spotter → keyword hypotheses with alignments", size=9, color=C["subtext"])
label(ax, MID_X, Y_KB + 0.56, "2. Drop keywords blocked by LANG_TAGS (if active)", size=9, color=C["subtext"])
label(ax, MID_X, Y_KB + 0.22, "3. merge_alignment_with_ws_hyps → splice keywords into text", size=9, color=C["subtext"])

# ─── OUTPUT ──────────────────────────────────────────────────────────────────
rounded_box(ax, LEFT_X, Y_OUT, LEFT_W, BOX_H, C["card"], C["output"])
label(ax, MID_X, Y_OUT + 1.0, "OUTPUT", size=12, bold=True, color=C["output"])
label(ax, MID_X, Y_OUT + 0.55, "TRANSCRIPT  (B, 1) BYTES    SCORE  (B, 1) FP32", size=9, color=C["subtext"])

# ── Vertical arrows ────────────────────────────────────────────────────────────
arrow(ax, MID_X, Y_AUDIO,                         MID_X, Y_ENCODE + BOX_H, C["encode"])
arrow(ax, MID_X, Y_ENCODE,                        MID_X, Y_LANG   + 1.9,   C["lang"])
arrow(ax, MID_X, Y_LANG,                          MID_X, Y_DECODE + 1.7,   C["decode"])
arrow(ax, MID_X, Y_DECODE,                        MID_X, Y_WB    + 2.2,    C["wb"])
arrow(ax, MID_X, Y_WB,                            MID_X, Y_KB    + 2.0,    C["kb"])
arrow(ax, MID_X, Y_KB,                            MID_X, Y_OUT   + BOX_H,  C["output"])

# ── "if active" bypass labels on arrows ───────────────────────────────────────
def bypass_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=7.5, color=color, ha="center", va="center",
            style="italic", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc=C["bg"], ec=color, lw=1, alpha=0.85))

bypass_label(ax, MID_X + 1.6, (Y_LANG + 1.9 + Y_LANG + 1.9 + 0.35) / 2,
             "if LANG_TAGS / BLANK_PENALTY", C["lang"])

bypass_label(ax, MID_X + 1.6, (Y_WB + 2.2 + Y_WB + 2.2 + 0.35) / 2,
             "if WORD_BOOST_PARAMS.enabled", C["wb"])

bypass_label(ax, MID_X + 1.6, (Y_KB + 2.0 + Y_KB + 2.0 + 0.35) / 2,
             "if CB_PARAMS + CONTEXT_GRAPH", C["kb"])

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Interaction / Compose table + legend
# ══════════════════════════════════════════════════════════════════════════════

RMX = RIGHT_X + RIGHT_W / 2   # center of right column

# ─── Section: Override vs Compose ─────────────────────────────────────────────
rounded_box(ax, RIGHT_X, 13.8, RIGHT_W, 6.8, "#14172a", C["border"], radius=0.5)
label(ax, RMX, 20.35, "Override vs. Compose", size=12, bold=True, color=C["text"])

rows = [
    ("Lang Constraint + Blank Penalty", "Compose", "both modify log-probs additively at Stage 1", C["lang"]),
    ("Lang Constraint + Word Boost",    "Compose", "share same LANG_TOKEN_MAP, different stages",  C["wb"]),
    ("Lang Constraint + Keyword Boost", "Compose\n+ Guard", "masked keywords silently dropped",    C["kb"]),
    ("Word Boost + Keyword Boost",      "Pipeline", "WB's modified log-probs fed into KB",         C["kb"]),
    ("BLANK_PENALTY vs WB blank_pen",   "Independent", "different stages; different log-prob tensors", C["blank"]),
    ("RNNT/CTC text vs Word Boost",     "Override", "WB re-decodes from CTC & replaces RNNT text", C["wb"]),
]

for idx, (pair, verdict, detail, col) in enumerate(rows):
    ry = 19.6 - idx * 1.0
    rounded_box(ax, RIGHT_X + 0.25, ry - 0.38, RIGHT_W - 0.5, 0.82, "#1a1e30", col, radius=0.25, alpha=0.5)
    label(ax, RIGHT_X + 0.55, ry + 0.09, pair, size=8.5, color=C["text"], ha="left")
    # verdict pill
    rounded_box(ax, RIGHT_X + 4.7, ry - 0.3, 1.5, 0.6, col, col, radius=0.2, alpha=0.35)
    label(ax, RIGHT_X + 5.45, ry + 0.06, verdict, size=8, color=col, bold=True)
    label(ax, RIGHT_X + 0.55, ry - 0.22, detail, size=7.5, color=C["subtext"], ha="left")

# ─── Section: Key design notes ────────────────────────────────────────────────
rounded_box(ax, RIGHT_X, 7.6, RIGHT_W, 5.8, "#14172a", C["border"], radius=0.5)
label(ax, RMX, 13.15, "Key Design Notes", size=12, bold=True, color=C["text"])

notes = [
    ("LANG_TOKEN_MAP is shared", "Language constraining, word boost, and keyword boost guard all\nuse the same {lang: [token_ids]} map, built once and cached."),
    ("CTC log-probs always produced", "Even for RNNT/TDT strategies, the CTC decoder head always runs\nso Stages 3 & 4 can operate on frame-level probabilities."),
    ("Mixed batches handled", "Items with different LANG_TAGS trigger per-item RNNT decode.\nCTC path and post-processing are always per-item."),
    ("Word Boost blank_penalty ≠ BLANK_PENALTY", "BLANK_PENALTY acts during Stage 1 (CTC path only).\nWord Boost's blank_penalty acts inside its own re-decode."),
]

ny = 12.7
for title, body in notes:
    label(ax, RIGHT_X + 0.5, ny, f"▸ {title}", size=9, color=C["text"], ha="left", bold=True)
    for line in body.split("\n"):
        ny -= 0.36
        label(ax, RIGHT_X + 0.7, ny, line, size=8, color=C["subtext"], ha="left")
    ny -= 0.5

# ─── Section: Legend ──────────────────────────────────────────────────────────
rounded_box(ax, RIGHT_X, 4.8, RIGHT_W, 2.5, "#14172a", C["border"], radius=0.5)
label(ax, RMX, 7.1, "Legend", size=12, bold=True, color=C["text"])

legend_items = [
    (C["audio"],  "AUDIO input"),
    (C["encode"], "Stage 0 — Preprocessing / Encoding"),
    (C["lang"],   "Stage 1 — Language Constraining / Blank Penalty"),
    (C["decode"], "Stage 2 — RNNT / TDT / CTC Decoding"),
    (C["wb"],     "Stage 3 — Word-level Language Boost"),
    (C["kb"],     "Stage 4 — Keyword Boosting"),
    (C["output"], "TRANSCRIPT + SCORE output"),
]

lx_col1 = RIGHT_X + 0.4
lx_col2 = RIGHT_X + RIGHT_W / 2 + 0.2
cols_split = 4  # first 4 items in col1

for li, (col, desc) in enumerate(legend_items):
    if li < cols_split:
        lx, ly = lx_col1, 6.65 - li * 0.42
    else:
        lx, ly = lx_col2, 6.65 - (li - cols_split) * 0.42

    rounded_box(ax, lx, ly - 0.12, 0.3, 0.28, col, col, radius=0.1)
    label(ax, lx + 0.5, ly + 0.02, desc, size=8, color=C["text"], ha="left")

# ─── Footer ───────────────────────────────────────────────────────────────────
label(ax, 8, 0.35, "All features are optional per-request. When none are active, the pipeline is a plain RNNT/TDT/CTC decode.",
      size=8.5, color=C["subtext"])

plt.tight_layout(pad=0.2)
out_path = "/home/nobroker-tlt415/Documents/claude/combined_implementation/decoding_pipeline_hierarchy.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"Saved: {out_path}")
