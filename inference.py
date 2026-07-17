#!/usr/bin/env python3
"""
inference.py
================================================================================
A local-VM script to dynamically scan for checkpoints under the training directory,
load them sequentially, run TDT/RNNT/CTC inference on a hardcoded audio file path, 
and save each transcript to both a consolidated text file and individual text files.

Includes:
- Dynamic support for both EncDecHybridRNNTCTCBPEModel and EncDecCTCModelBPE checkpoints
- High-speed weight swapping (reduces checkpoint load time from 25s to 0.2s)
- Dynamic self-healing SentencePiece vocab.txt builder
- Consolidated single file transcriber
================================================================================
"""

import os
# Fix FileNotFoundError for temporary directories when loading torch / dill in restricted environments
os.environ["TMPDIR"] = os.path.expanduser("~") or os.getcwd()

import sys
import time
import glob
import json
import tarfile
import numpy as np
import torch
import soundfile as sf
from typing import List, Dict, Tuple, Optional, Any
from omegaconf import OmegaConf

# Force NeMo & PyTorch Lightning to be quiet
import logging
for _log in ("nemo_logger", "pytorch_lightning", "nemo.collections.asr"):
    logging.getLogger(_log).setLevel(logging.ERROR)

from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import (
    EncDecHybridRNNTCTCBPEModel,
)

from nemo.collections.asr.models.ctc_bpe_models import (
    EncDecCTCModelBPE,
)

# ── CONFIGURATION (Edit audio path or results folder if they are different) ────

AUDIO_PATH = "mam.wav"  # Hardcoded audio file path to run inference on

RESULTS_BASE_DIR = "/home/ubuntu/Vallabh/conformer/parakeet_tdt_training/cache_clean_v1/audio/training/Parakeet_Hybrid/results/Parakeet_Hybrid"

CONSOLIDATED_FILE = "all_transcripts.txt"  # Single consolidated output file

# Tuning parameters matching model.py defaults
SAMPLE_RATE = 16000
WORD_BOUNDARY = "\u2581"
LONG_AUDIO_THRESHOLD_S = 20.0
LONG_AUDIO_CHUNK_S = 20.0
LONG_AUDIO_OVERLAP_S = 3.0
MAX_SYMBOLS = 10
BLANK_PENALTY = 0.0

# ── Helper functions for Numerical Operations ───────────────────────────────────

def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable log_softmax along the last axis of a numpy array."""
    m = np.max(x, axis=-1, keepdims=True)
    m = np.where(np.isneginf(m), 0.0, m)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


# ── Riva-style Confidence Estimation Helpers ─────────────────────────────────────

def _riva_confidence(log_probs_2d: np.ndarray, blank_id: int) -> float:
    T = log_probs_2d.shape[0]
    if T == 0:
        return 0.0

    lp = _log_softmax_np(log_probs_2d)
    ctc_argmax = np.argmax(lp, axis=-1)           # (T,)
    frame_logp = lp[np.arange(T), ctc_argmax]     # (T,)

    word_means: List[float] = []
    word_lens:  List[int]   = []
    cur_logps:  List[float] = []

    for t in range(T):
        if ctc_argmax[t] == blank_id:
            if cur_logps:
                word_means.append(float(np.mean(cur_logps)))
                word_lens.append(len(cur_logps))
                cur_logps = []
        else:
            cur_logps.append(float(frame_logp[t]))

    if cur_logps:
        word_means.append(float(np.mean(cur_logps)))
        word_lens.append(len(cur_logps))

    if word_means:
        wm = np.array(word_means, dtype=np.float64)
        wl = np.array(word_lens,  dtype=np.float64)
        return float(np.dot(wm, wl) / wl.sum())
    else:
        return float(frame_logp.mean())


def _riva_confidence_rnnt(
    log_probs_2d: np.ndarray,
    blank_id: int,
    hypothesis,
    tokenizer,
) -> float:
    T = log_probs_2d.shape[0]
    if T == 0:
        return 0.0

    lp = _log_softmax_np(log_probs_2d)
    y_seq = hypothesis.y_sequence
    if isinstance(y_seq, torch.Tensor):
        y_seq = y_seq.tolist()

    timestamps = getattr(hypothesis, "timestamp", None)
    if timestamps is None:
        return _riva_confidence(log_probs_2d, blank_id)
    if isinstance(timestamps, dict):
        timestamps = timestamps.get("timestep", [])
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.tolist()

    if len(y_seq) == 0 or len(timestamps) == 0 or len(y_seq) != len(timestamps):
        return _riva_confidence(log_probs_2d, blank_id)

    try:
        token_strs = tokenizer.ids_to_tokens(list(y_seq))
    except Exception:
        return _riva_confidence(log_probs_2d, blank_id)

    word_means: List[float] = []
    word_lens:  List[int]   = []
    cur_logps:  List[float] = []

    for tstr, tid, frame in zip(token_strs, y_seq, timestamps):
        f   = min(int(frame), T - 1)
        tlp = float(lp[f, int(tid)])
        if tstr.startswith(WORD_BOUNDARY) and cur_logps:
            word_means.append(float(np.mean(cur_logps)))
            word_lens.append(len(cur_logps))
            cur_logps = []
        cur_logps.append(tlp)

    if cur_logps:
        word_means.append(float(np.mean(cur_logps)))
        word_lens.append(len(cur_logps))

    if word_means:
        wm = np.array(word_means, dtype=np.float64)
        wl = np.array(word_lens,  dtype=np.float64)
        return float(np.dot(wm, wl) / wl.sum())

    all_lps = [float(lp[min(int(f), T - 1), int(tid)]) for tid, f in zip(y_seq, timestamps)]
    return float(np.mean(all_lps)) if all_lps else _riva_confidence(log_probs_2d, blank_id)


# ── Timestamps Extraction Helpers ────────────────────────────────────────────────

def _hypothesis_to_word_timestamps(
    hypothesis,
    tokenizer,
    frame_shift_s: float,
    log_probs_2d: Optional[np.ndarray] = None,
) -> List[Dict]:
    y_seq = hypothesis.y_sequence
    if isinstance(y_seq, torch.Tensor):
        y_seq = y_seq.tolist()

    timestamps = hypothesis.timestep if hasattr(hypothesis, "timestep") else getattr(hypothesis, "timestamp", None)
    if isinstance(timestamps, dict):
        timestamps = timestamps.get("timestep", [])
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.tolist()

    if len(y_seq) == 0 or not timestamps or len(y_seq) != len(timestamps):
        return []

    try:
        token_strs = tokenizer.ids_to_tokens(list(y_seq))
    except Exception:
        return []

    lp_norm: Optional[np.ndarray] = None
    T_lp = 0
    if log_probs_2d is not None:
        lp_norm = _log_softmax_np(log_probs_2d)
        T_lp = lp_norm.shape[0]

    words: List[Dict] = []
    cur_pieces: List[str] = []
    cur_frames: List[int] = []
    cur_token_ids: List[int] = []

    def _flush_word_rnnt():
        if not cur_pieces:
            return
        word_text = "".join(cur_pieces).lstrip(WORD_BOUNDARY)
        if not word_text:
            return
        entry: Dict = {
            "word":    word_text,
            "start_s": round(min(cur_frames) * frame_shift_s, 3),
            "end_s":   round((max(cur_frames) + 1) * frame_shift_s, 3),
        }
        if lp_norm is not None and cur_token_ids:
            token_logps = [
                float(lp_norm[f, tid])
                for f, tid in zip(cur_frames, cur_token_ids)
                if 0 <= f < T_lp
            ]
            if token_logps:
                entry["score"] = round(float(np.mean(token_logps)), 4)
        words.append(entry)

    for tstr, frame, tid in zip(token_strs, timestamps, y_seq):
        frame = int(frame)
        if tstr.startswith(WORD_BOUNDARY) and cur_pieces:
            _flush_word_rnnt()
            cur_pieces = []
            cur_frames = []
            cur_token_ids = []
        cur_pieces.append(tstr)
        cur_frames.append(frame)
        cur_token_ids.append(int(tid))

    _flush_word_rnnt()
    return words


def _compute_ctc_word_timestamps(
    log_probs_2d: np.ndarray,
    tokenizer,
    frame_shift_s: float,
    blank_id: int,
    vocab_size: int,
) -> List[Dict]:
    lp_norm = _log_softmax_np(log_probs_2d)
    raw_ids = np.argmax(lp_norm, axis=-1).tolist()

    collapsed: List[Tuple[int, int]] = []
    prev_tid = -1
    run_start = 0
    for t_idx, tid in enumerate(raw_ids):
        if tid != prev_tid:
            if prev_tid != -1 and prev_tid != blank_id and prev_tid < vocab_size:
                collapsed.append((prev_tid, run_start))
            run_start = t_idx
            prev_tid = tid
    if prev_tid != -1 and prev_tid != blank_id and prev_tid < vocab_size:
        collapsed.append((prev_tid, run_start))

    if not collapsed:
        return []

    try:
        unique_ids = list({tid for tid, _ in collapsed})
        piece_map  = {tid: (tokenizer.ids_to_tokens([tid]) or [""])[0] for tid in unique_ids}
    except Exception:
        return []

    words: List[Dict] = []
    cur_pieces: List[str] = []
    cur_token_logps: List[float] = []
    cur_start_emission: Optional[int] = None
    cur_end_emission: Optional[int] = None
    prev_word_end_emission: Optional[int] = None

    def _flush_word():
        if not cur_pieces or cur_start_emission is None:
            return
        word_text = "".join(cur_pieces).lstrip(WORD_BOUNDARY)
        if not word_text:
            return
        s_frame = (
            prev_word_end_emission
            if prev_word_end_emission is not None
            else max(0, cur_start_emission - 1)
        )
        entry: Dict = {
            "word":    word_text,
            "start_s": round(s_frame * frame_shift_s, 3),
            "end_s":   round(cur_end_emission * frame_shift_s, 3),
        }
        if cur_token_logps:
            entry["score"] = round(float(np.mean(cur_token_logps)), 4)
        words.append(entry)

    for tid, emission_frame in collapsed:
        piece = piece_map.get(tid, "")
        if piece.startswith(WORD_BOUNDARY) and cur_pieces:
            _flush_word()
            prev_word_end_emission = cur_end_emission
            cur_pieces         = []
            cur_token_logps    = []
            cur_start_emission = None
            cur_end_emission   = None
        cur_pieces.append(piece)
        cur_token_logps.append(float(lp_norm[emission_frame, tid]))
        if cur_start_emission is None:
            cur_start_emission = emission_frame
        cur_end_emission = emission_frame

    _flush_word()
    return words


# ── Long Audio Chunking & Stitching Helpers ──────────────────────────────────────

def _plan_audio_chunks(
    audio_f32: np.ndarray,
    sample_rate: int,
    chunk_s: float,
    overlap_s: float,
) -> List[Dict]:
    T               = len(audio_f32)
    chunk_samples   = int(chunk_s   * sample_rate)
    overlap_samples = int(overlap_s * sample_rate)

    chunks: List[Dict] = []
    pos = 0

    while pos < T:
        chunk_end = min(pos + chunk_samples, T)
        is_last   = chunk_end >= T

        if is_last:
            chunks.append({
                'start':   pos,
                'end':     T,
                'start_s': pos / sample_rate,
                'join_s':  None,
            })
            break

        chunks.append({
            'start':   pos,
            'end':     chunk_end,
            'start_s': pos / sample_rate,
            'join_s':  chunk_end / sample_rate,
        })

        next_start = max(0, chunk_end - overlap_samples)
        if next_start <= pos:
            next_start = pos + max(1, chunk_samples // 2)
        pos = next_start

    return chunks


def _stitch_chunk_results(chunk_results: List[Dict]) -> Tuple[str, List[Dict]]:
    all_words: List[Dict] = []
    prev_join = 0.0

    for k, chunk in enumerate(chunk_results):
        off  = chunk['start_s']
        join = chunk.get('join_s')

        c_words_abs = []
        for w in chunk['words']:
            abs_start = w['start_s'] + off
            abs_end   = w['end_s']   + off
            entry = {
                'word':    w['word'],
                'start_s': round(abs_start, 3),
                'end_s':   round(abs_end,   3),
            }
            if 'score' in w:
                entry['score'] = w['score']
            c_words_abs.append(entry)

        if k == 0:
            for entry in c_words_abs:
                if join is not None and entry['start_s'] >= join:
                    continue
                all_words.append(entry)
            if join is not None:
                prev_join = join
            continue

        best_match = None
        search_all_start = max(0, len(all_words) - 15)
        search_c_end = min(len(c_words_abs), 15)

        for i in range(search_all_start, len(all_words)):
            w1 = all_words[i]
            w1_word = w1['word'].lower().strip()
            w1_time = w1['start_s']

            for j in range(search_c_end):
                w2 = c_words_abs[j]
                w2_word = w2['word'].lower().strip()
                w2_time = w2['start_s']

                if w1_word == w2_word and abs(w1_time - w2_time) < 1.5:
                    if best_match is None or i > best_match[0]:
                        best_match = (i, j)

        if best_match is not None:
            i_match, j_match = best_match
            all_words = all_words[:i_match + 1]
            for entry in c_words_abs[j_match + 1:]:
                if join is not None and entry['start_s'] >= join:
                    continue
                all_words.append(entry)
        else:
            for entry in c_words_abs:
                if entry['start_s'] < prev_join:
                    continue
                if join is not None and entry['start_s'] >= join:
                    continue
                all_words.append(entry)

        if join is not None:
            prev_join = join

    final_text = ' '.join(w['word'] for w in all_words).strip()
    return final_text, all_words


def _hyp_to_text(h) -> str:
    if h is None:
        return ""
    if isinstance(h, (bytes, bytearray)):
        return h.decode("utf-8", errors="replace")
    if hasattr(h, "text"):
        t = h.text
        return t.decode("utf-8", errors="replace") if isinstance(t, (bytes, bytearray)) else (t or "")
    return str(h)


# ── Single Audio Segment Decoder ─────────────────────────────────────────────────

def _decode_single_signal(
    model,
    signal_f32: np.ndarray,
    device: torch.device,
    blank_penalty: float,
    blank_id: int,
) -> Tuple[Any, np.ndarray]:
    batch_1 = signal_f32[np.newaxis, :]
    signal_t = torch.from_numpy(batch_1).to(device)
    lengths = torch.full((1,), signal_f32.shape[0], dtype=torch.long, device=device)

    # 1. Feature extraction
    features, feat_lengths = model.preprocessor(input_signal=signal_t, length=lengths)

    # 2. Encoder forward pass
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
        encoded, encoded_lengths = model.encoder(audio_signal=features, length=feat_lengths)
    encoded_f32 = encoded.float()

    # Dynamic CTC-only vs Hybrid RNNT paths
    is_ctc_only = not hasattr(model, "joint")

    if is_ctc_only:
        # CTC-only decoding path
        if hasattr(model, "decoder"):
            log_probs = model.decoder(encoder_output=encoded_f32)
        else:
            log_probs = model.ctc_decoder(encoder_output=encoded_f32)

        lp_gpu = log_probs[0].clone()
        if blank_penalty != 0.0:
            lp_gpu[:, blank_id] -= blank_penalty
        lp_final = lp_gpu.cpu().numpy()

        result = model.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs,
            decoder_lengths=encoded_lengths,
            return_hypotheses=True,
        )
        hyps = result[0] if (isinstance(result, tuple) and not isinstance(result[0], str)) else result
        hyp = hyps[0]
        return hyp, lp_final
    else:
        # Hybrid RNNT/CTC paths
        log_probs_gpu = model.ctc_decoder(encoder_output=encoded_f32)
        lp_gpu = log_probs_gpu[0].clone()
        if blank_penalty != 0.0:
            lp_gpu[:, blank_id] -= blank_penalty
        lp_final = lp_gpu.cpu().numpy()

        # Transducer (RNNT/TDT) greedy decoding
        joint_out_layer = model.joint.joint_net[-1]
        bias = joint_out_layer.bias.data

        if blank_penalty != 0.0:
            bias[blank_id] -= blank_penalty

        try:
            result = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded_f32,
                encoded_lengths=encoded_lengths,
                return_hypotheses=True,
            )
        finally:
            if blank_penalty != 0.0:
                bias[blank_id] += blank_penalty

        hyps = result[0] if (isinstance(result, tuple) and not isinstance(result[0], str)) else result
        hyp = hyps[0]

        return hyp, lp_final


# ── Load Audio utility ───────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Loads an audio file and resamples to target sample rate, returning float32 mono."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(
                audio, 
                orig_sr=sr, 
                target_sr=target_sr
            )
        except ImportError:
            raise RuntimeError(
                f"Audio sample rate {sr} != target {target_sr}. "
                f"Please install 'librosa' for dynamic resampling."
            )
    return audio


# ── Tokenizer Self-Healing Dynamic Extraction / Generation ───────────────────────

def check_and_fix_tokenizer(results_base_dir: str) -> bool:
    """
    Checks if /home/ubuntu/Vallabh/conformer/extracted_tokenizer/vocab.txt exists.
    If vocab.txt is missing but tokenizer.model is present, automatically
    reconstructs vocab.txt using SentencePiece.
    Otherwise, extracts from any valid local .nemo package on the VM.
    """
    target_vocab = "/home/ubuntu/Vallabh/conformer/extracted_tokenizer/vocab.txt"
    target_model = "/home/ubuntu/Vallabh/conformer/extracted_tokenizer/tokenizer.model"
    target_dir = os.path.dirname(target_vocab)
    
    if os.path.exists(target_vocab):
        print(f"[evaluator] Tokenizer vocabulary found locally at '{target_vocab}'")
        return True
        
    # Case 1: Model file exists, but vocab.txt does not. Rebuild using SentencePiece!
    if os.path.exists(target_model):
        print(f"[evaluator] Found '{target_model}'. Generating '{target_vocab}' dynamically using SentencePiece...", flush=True)
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(target_model)
            with open(target_vocab, "w", encoding="utf-8") as f:
                for i in range(sp.get_piece_size()):
                    piece = sp.id_to_piece(i)
                    score = sp.get_score(i)
                    f.write(f"{piece}\t{score}\n")
            print(f"[evaluator] SUCCESS: Generated vocabulary file at '{target_vocab}'!")
            return True
        except Exception as e:
            print(f"[evaluator] Error generating vocab.txt from tokenizer.model: {e}", file=sys.stderr)

    # Case 2: Standard .nemo tar extraction
    print(f"[evaluator] WARNING: Tokenizer vocabulary missing at '{target_vocab}'. Attempting auto-extraction...", flush=True)
    
    search_paths = [
        ".",
        "..",
        results_base_dir,
        os.path.dirname(results_base_dir),
        "/home/ubuntu/Vallabh/conformer"
    ]
    
    nemo_files = []
    for p in search_paths:
        if os.path.exists(p):
            nemo_files.extend(glob.glob(os.path.join(p, "*.nemo")))
            nemo_files.extend(glob.glob(os.path.join(p, "*/*.nemo")))
            
    nemo_files = list(set([os.path.abspath(f) for f in nemo_files]))
    
    if not nemo_files:
        print("[evaluator] ERROR: No .nemo file found in the search paths to extract the tokenizer from.", file=sys.stderr)
        return False
        
    # Loop over nemo files until one successfully yields tokenizer files
    for nemo_path in nemo_files:
        print(f"[evaluator] Trying .nemo file: '{nemo_path}'...", flush=True)
        try:
            os.makedirs(target_dir, exist_ok=True)
            with tarfile.open(nemo_path, "r") as tar:
                extracted_any = False
                for member in tar.getmembers():
                    basename = os.path.basename(member.name)
                    if "tokenizer" in member.name or "vocab" in member.name:
                        member.name = basename
                        tar.extract(member, path=target_dir)
                        print(f"[evaluator] Extracted '{basename}' -> '{target_dir}'", flush=True)
                        extracted_any = True
                
                # If we successfully extracted tokenizer.model but not vocab.txt, trigger Case 1
                if os.path.exists(target_model) and not os.path.exists(target_vocab):
                    import sentencepiece as spm
                    sp = spm.SentencePieceProcessor()
                    sp.load(target_model)
                    with open(target_vocab, "w", encoding="utf-8") as f:
                        for i in range(sp.get_piece_size()):
                            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
                    print(f"[evaluator] Generated missing vocab.txt from extracted tokenizer.model")
                    
                if os.path.exists(target_vocab):
                    print(f"[evaluator] SUCCESS: Tokenizer files configured at '{target_vocab}'!", flush=True)
                    return True
        except Exception as e:
            print(f"[evaluator] Skipped '{nemo_path}' due to error: {e}", file=sys.stderr)
            
    print("[evaluator] ERROR: Could not extract tokenizer from any available .nemo files.", file=sys.stderr)
    return False


# ── High-Speed Weight Swapping Helper ────────────────────────────────────────────

def fast_load_state_dict(model: Any, ckpt_path: str, device: torch.device):
    """Loads state dict directly from checkpoints to bypass slow full initialization."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    model_state = model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k in model_state:
            new_state_dict[k] = v
        elif k.startswith("model.") and k[6:] in model_state:
            new_state_dict[k[6:]] = v
        elif f"model.{k}" in model_state:
            new_state_dict[f"model.{k}"] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)


# ── Main Evaluator ───────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluator] Using device: {device}", flush=True)

    # Validate audio file
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: Audio file not found at '{AUDIO_PATH}'", file=sys.stderr)
        sys.exit(1)

    print(f"[evaluator] Pre-loading audio '{AUDIO_PATH}' ...", flush=True)
    signal_f32 = load_audio(AUDIO_PATH, SAMPLE_RATE)
    duration_s = len(signal_f32) / SAMPLE_RATE
    print(f"[evaluator] Audio loaded successfully: {duration_s:.2f}s (mono, 16000Hz)", flush=True)

    # Automatically check and self-heal the tokenizer vocabulary files if they are missing
    check_and_fix_tokenizer(RESULTS_BASE_DIR)

    # Initialize Consolidated Single File
    with open(CONSOLIDATED_FILE, "w", encoding="utf-8") as ch:
        ch.write(f"================================================================================\n")
        ch.write(f"CONSOLIDATED TRANSCRIPTS FOR AUDIO: {AUDIO_PATH}\n")
        ch.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        ch.write(f"================================================================================\n\n")

    # Dynamic Scan for Checkpoints across every timestamp/date subfolder
    glob_pattern = os.path.join(RESULTS_BASE_DIR, "*/checkpoints/*.ckpt")
    print(f"[evaluator] Scanning folder path: '{glob_pattern}' ...", flush=True)
    checkpoints = sorted(glob.glob(glob_pattern))

    if not checkpoints:
        print(f"[evaluator] ERROR: No .ckpt files found in '{RESULTS_BASE_DIR}/*/checkpoints/'.", file=sys.stderr)
        sys.exit(1)

    print(f"[evaluator] Found {len(checkpoints)} checkpoints to evaluate.", flush=True)

    model = None
    model_cls = None
    cfg_dec = None
    is_ctc_only = False

    for idx, ckpt_path in enumerate(checkpoints):
        print("\n" + "="*80)
        print(f"[evaluator] [{idx+1}/{len(checkpoints)}] Evaluating: '{ckpt_path}' ...")
        t_load0 = time.perf_counter()

        path_parts = ckpt_path.split(os.sep)
        date_folder = path_parts[-3] if len(path_parts) >= 3 else "unknown_date"
        ckpt_filename = os.path.basename(ckpt_path)

        if model is None:
            # First checkpoint: Full load to initialize the architecture, tokenizer, preprocessor
            print(f"[evaluator] Initializing model architecture from first checkpoint (slow step)...")
            try:
                # Dynamic model class detection
                ckpt_dict = torch.load(ckpt_path, map_location="cpu")
                target_cls = ckpt_dict.get("hyper_parameters", {}).get("cfg", {}).get("_target_", "")
                
                if "EncDecCTCBPEModel" in target_cls or "EncDecCTCModelBPE" in target_cls:
                    model_cls = EncDecCTCModelBPE
                    is_ctc_only = True
                    print(f"[evaluator] Detected CTC model checkpoint target class: {target_cls}")
                else:
                    model_cls = EncDecHybridRNNTCTCBPEModel
                    is_ctc_only = False
                    print(f"[evaluator] Detected Hybrid RNNT/CTC model checkpoint target class: {target_cls}")

                model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)
                model.eval().to(device)
                model.requires_grad_(False)
                print(f"[evaluator] Model initialized fully in {time.perf_counter() - t_load0:.2f}s")
            except Exception as e:
                print(f"[evaluator] Fatal: Failed to initialize model structure: {e}", file=sys.stderr)
                sys.exit(1)

            # Auto-configure decoding strategy
            if is_ctc_only:
                cfg_dec = OmegaConf.create({
                    "strategy": "greedy",
                })
                if hasattr(model, "change_decoding_strategy"):
                    model.change_decoding_strategy(decoding_cfg=cfg_dec)
                print("[evaluator] Decoder strategy: CTC Greedy")
            else:
                is_tdt_supported = getattr(model.joint, "num_extra_outputs", 0) > 0
                if is_tdt_supported:
                    cfg_dec = OmegaConf.create({
                        "strategy":           "greedy_batch",
                        "model_type":         "tdt",
                        "durations":          [0, 1, 2, 3, 4],
                        "num_tdt_durations":  5,
                        "greedy": {
                            "max_symbols":              MAX_SYMBOLS,
                            "use_cuda_graph_decoder":   False,
                        },
                    })
                    model.change_decoding_strategy(decoder_type="rnnt", decoding_cfg=cfg_dec)
                    print("[evaluator] Decoder strategy: TDT")
                else:
                    cfg_dec = OmegaConf.create({
                        "strategy": "greedy_batch",
                        "greedy": {
                            "max_symbols":              MAX_SYMBOLS,
                            "use_cuda_graph_decoder":   False,
                        },
                    })
                    model.change_decoding_strategy(decoder_type="rnnt", decoding_cfg=cfg_dec)
                    print("[evaluator] Decoder strategy: Standard RNNT")
        else:
            # Subsequent checkpoints: Ultra-fast state-dict weight swap!
            try:
                fast_load_state_dict(model, ckpt_path, device)
                print(f"[evaluator] Swapped weights dynamically in {time.perf_counter() - t_load0:.2f}s")
            except Exception as e:
                print(f"[evaluator] Swapping weights failed. Falling back to slow load. Error: {e}")
                try:
                    model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)
                    model.eval().to(device)
                    model.requires_grad_(False)
                    if is_ctc_only:
                        if hasattr(model, "change_decoding_strategy"):
                            model.change_decoding_strategy(decoding_cfg=cfg_dec)
                    else:
                        model.change_decoding_strategy(decoder_type="rnnt", decoding_cfg=cfg_dec)
                    print(f"[evaluator] Fallback slow load completed in {time.perf_counter() - t_load0:.2f}s")
                except Exception as ex:
                    print(f"[evaluator] Error: Checkpoint could not be loaded: {ex}", file=sys.stderr)
                    continue

        frame_shift_s = model.cfg.preprocessor.window_stride * model.cfg.encoder.subsampling_factor
        
        # Extract blank_id safely
        if hasattr(model, "ctc_decoding"):
            blank_id = getattr(model.ctc_decoding, 'blank_id', model.tokenizer.vocab_size)
        elif hasattr(model, "decoding") and hasattr(model.decoding, "blank_id"):
            blank_id = getattr(model.decoding, 'blank_id', model.tokenizer.vocab_size)
        else:
            blank_id = model.tokenizer.vocab_size

        # Transcribe
        t_dec0 = time.perf_counter()
        if duration_s > LONG_AUDIO_THRESHOLD_S:
            chunks = _plan_audio_chunks(
                signal_f32,
                SAMPLE_RATE,
                chunk_s=LONG_AUDIO_CHUNK_S,
                overlap_s=LONG_AUDIO_OVERLAP_S
            )

            chunk_results = []
            for k, chunk in enumerate(chunks):
                c_audio   = signal_f32[chunk['start']:chunk['end']]
                c_start_s = chunk['start_s']
                c_dur_s   = len(c_audio) / SAMPLE_RATE

                c_hyp, c_lp = _decode_single_signal(
                    model=model,
                    signal_f32=c_audio,
                    device=device,
                    blank_penalty=BLANK_PENALTY,
                    blank_id=blank_id
                )

                if is_ctc_only:
                    c_words = _compute_ctc_word_timestamps(
                        c_lp, model.tokenizer, frame_shift_s, blank_id, model.tokenizer.vocab_size
                    )
                    c_score = _riva_confidence(c_lp, blank_id)
                else:
                    c_words = _hypothesis_to_word_timestamps(
                        c_hyp, model.tokenizer, frame_shift_s, log_probs_2d=c_lp
                    )
                    if not c_words:
                        c_words = _compute_ctc_word_timestamps(
                            c_lp, model.tokenizer, frame_shift_s, blank_id, model.tokenizer.vocab_size
                        )
                    c_score = _riva_confidence_rnnt(c_lp, blank_id, c_hyp, model.tokenizer)

                chunk_results.append({
                    'words':   c_words,
                    'start_s': c_start_s,
                    'join_s':  chunk['join_s'],
                    'score':   c_score,
                    'dur_s':   c_dur_s,
                })

            final_text, _ = _stitch_chunk_results(chunk_results)
        else:
            c_hyp, c_lp = _decode_single_signal(
                model=model,
                signal_f32=signal_f32,
                device=device,
                blank_penalty=BLANK_PENALTY,
                blank_id=blank_id
            )
            final_text = _hyp_to_text(c_hyp)

        dec_elapsed = time.perf_counter() - t_dec0
        print(f"[evaluator] Decoding finished in {dec_elapsed:.2f}s (RTF: {dec_elapsed/duration_s:.3f})")

        # Save to individual text file (as backup)
        out_filename = f"{date_folder}_{os.path.splitext(ckpt_filename)[0]}_transcript.txt"
        with open(out_filename, "w", encoding="utf-8") as fh:
            fh.write(final_text + "\n")

        # Save to consolidated single file
        with open(CONSOLIDATED_FILE, "a", encoding="utf-8") as ch:
            ch.write(f"Checkpoint: {date_folder}/{ckpt_filename}\n")
            ch.write(f"Transcript: {final_text}\n")
            ch.write(f"{'-'*80}\n\n")

        print(f"[evaluator] Saved individual: '{out_filename}'")
        print(f"[evaluator] Appended to consolidated file: '{CONSOLIDATED_FILE}'")
        print("="*80)

    print(f"\n[evaluator] All checkpoint evaluations completed.")
    print(f"[evaluator] Consolidated transcripts saved to: '{os.path.abspath(CONSOLIDATED_FILE)}'")


if __name__ == "__main__":
    main()
