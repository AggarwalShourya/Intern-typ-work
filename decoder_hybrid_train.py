"""
Author: Sri Vallabh Tammiredd
Date: 12th February 2026

Modified: Replaced EncDecRNNTBPEModel → EncDecHybridRNNTCTCBPEModel.
          InterCTC loss enabled at layer 8 (midpoint of 17 layers), weight=0.3.
          Optimizer now covers encoder + decoder + joint + aux_ctc decoder params.
          --dataset_name now accepts multiple comma-separated CSV/JSON paths.

          # Two decoders: RNNT (Transducer) + auxiliary CTC
# InterCTC enabled at encoder layer 8 (of 17 total layers), weight=0.3
#
# Loss breakdown:
#   - Final layer RNNT loss    (weight = 0.7 × (1 - aux_ctc.ctc_loss_weight))
#   - Final layer aux CTC loss (weight = 0.7 × aux_ctc.ctc_loss_weight = 0.21)
#   - InterCTC at layer 8      (weight = 0.3, per interctc.loss_weights)
#
# Reference: InterCTC paper https://arxiv.org/abs/2102.03216
"""


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# os.environ["NEMO_RNNT_USE_NUMBA"] = "0"          
# os.environ["NEMO_FORCE_RNNT_LOSS_TORCH"] = "1"   
# os.environ["NUMBA_DISABLE_JIT"] = "1"            
# os.environ["NEMO_ENABLE_CUDA_GRAPHS"] = "0"
import soundfile as sf
import re
import glob
import sox
import json
from lightning.pytorch.callbacks import LearningRateMonitor
import logging
import tokenizers
import pandas as pd
from tqdm import tqdm
import torch
torch.set_float32_matmul_precision("high")
import lightning.pytorch as pl
import multiprocessing as mp
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

# from gcs_bucket import download_file, gcs_path_exists
import CONFORMER_CTC_DEFAULT_TRAINING_PARAMS

from CONFORMER_CTC_DEFAULT_TRAINING_PARAMS import CONFORMER_CTC_DEFAULT_TRAINING_PARAMS


from lightning.pytorch.callbacks import ModelCheckpoint
from nemo.utils import logging
from nemo.collections import asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
# ── CHANGED: import the Hybrid RNNT-CTC model instead of plain RNNT ──────────
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel

# ── Monkey patch NeMo's RNNT Loss resolver to support robust transducer losses ──
import nemo.collections.asr.losses.rnnt as nemo_rnnt
import torch

original_resolve_rnnt_loss = nemo_rnnt.resolve_rnnt_loss

def robust_resolve_rnnt_loss(loss_name: str, blank_idx: int, loss_kwargs: dict = None) -> torch.nn.Module:
    if loss_name == 'star_transducer':
        from target_robust_transducer import GraphStarTransducerLoss
        cleaned_kwargs = nemo_rnnt._clean_kwargs(loss_name, loss_kwargs, GraphStarTransducerLoss.__init__, ignore_params={"blank"})
        return GraphStarTransducerLoss(blank=blank_idx, **cleaned_kwargs)
    elif loss_name == 'bypass_transducer':
        from target_robust_transducer import GraphBypassTransducerLoss
        cleaned_kwargs = nemo_rnnt._clean_kwargs(loss_name, loss_kwargs, GraphBypassTransducerLoss.__init__, ignore_params={"blank"})
        return GraphBypassTransducerLoss(blank=blank_idx, **cleaned_kwargs)
    elif loss_name == 'target_robust_transducer':
        from target_robust_transducer import GraphTargetRobustTransducerLoss
        cleaned_kwargs = nemo_rnnt._clean_kwargs(loss_name, loss_kwargs, GraphTargetRobustTransducerLoss.__init__, ignore_params={"blank"})
        return GraphTargetRobustTransducerLoss(blank=blank_idx, **cleaned_kwargs)
    else:
        return original_resolve_rnnt_loss(loss_name, blank_idx, loss_kwargs)

nemo_rnnt.resolve_rnnt_loss = robust_resolve_rnnt_loss

# Register new loss names in resolver
for name in ['star_transducer', 'bypass_transducer', 'target_robust_transducer']:
    nemo_rnnt.RNNT_LOSS_RESOLVER[name] = nemo_rnnt.RNNTLossConfig(
        loss_name=name,
        lib_name="k2",
        is_available=True,  # Override to True to bypass environment checks
        installation_msg=nemo_rnnt.K2_INSTALLATION_MESSAGE,
        force_float32=False,
    )

def robust_training_step(self, batch, batch_nb):
    # Reset access registry
    from nemo.core.classes.mixins import AccessMixin
    import torch
    
    if AccessMixin.is_access_enabled(self.model_guid):
        AccessMixin.reset_registry(self)

    if self.is_interctc_enabled():
        AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

    if len(batch) == 5:
        signal, signal_len, transcript, transcript_len, lang = batch
        self._current_lang = lang
        if hasattr(self.joint, 'joint_net') and len(self.joint.joint_net) > 2:
            self.joint.joint_net[2]._current_lang = lang
        if hasattr(self, 'ctc_decoder'):
            self.ctc_decoder._current_lang = lang
    else:
        signal, signal_len, transcript, transcript_len = batch

    # forward() only performs encoder forward
    from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
    if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
        encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
    else:
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
    del signal

    # During training, loss must be computed, so decoder forward is necessary
    decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

    if hasattr(self, '_trainer') and self._trainer is not None:
        log_every_n_steps = self._trainer.log_every_n_steps
        sample_id = self._trainer.global_step
    else:
        log_every_n_steps = 1
        sample_id = batch_nb

    if (sample_id + 1) % log_every_n_steps == 0:
        compute_wer = True
    else:
        compute_wer = False

    # Compute full joint and loss
    joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
    
    main_loss = self.loss(
        log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
    )

    # Add auxiliary losses, if registered
    main_loss = self.add_auxiliary_losses(main_loss)

    # Calculate robust transducer loss if weight > 0
    robust_weight = getattr(self, "robust_loss_weight", 0.0)
    robust_loss_value = 0.0
    if robust_weight > 0.0 and hasattr(self, "robust_loss_module"):
        num_extra = getattr(self.joint, "num_extra_outputs", 0)
        vocab_size_with_blank = joint.shape[-1] - num_extra
        sliced_joint = joint[..., :vocab_size_with_blank]
        
        robust_loss_value = self.robust_loss_module(
            acts=sliced_joint,
            labels=transcript,
            act_lens=encoded_len,
            label_lens=target_length
        )
        
        if self.loss.reduction == 'mean_batch':
            robust_loss_value = robust_loss_value.mean()
        elif self.loss.reduction == 'mean':
            robust_loss_value = torch.div(robust_loss_value, target_length).mean()
        elif self.loss.reduction == 'sum':
            robust_loss_value = robust_loss_value.sum()
        elif self.loss.reduction == 'mean_volume':
            robust_loss_value = robust_loss_value.sum() / target_length.sum()
            
        loss_value = (1 - robust_weight) * main_loss + robust_weight * robust_loss_value
    else:
        loss_value = main_loss

    tensorboard_logs = {
        'learning_rate': self._optimizer.param_groups[0]['lr'],
        'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
    }

    if compute_wer:
        if hasattr(self, '_current_lang') and isinstance(self._current_lang, list) and hasattr(self, 'joint') and hasattr(self.joint, 'joint_net') and len(self.joint.joint_net) > 2:
            for b in range(encoded.shape[0]):
                self.joint.joint_net[2]._current_lang = self._current_lang[b]
                self.wer.update(
                    predictions=encoded[b:b+1],
                    predictions_lengths=encoded_len[b:b+1],
                    targets=transcript[b:b+1],
                    targets_lengths=transcript_len[b:b+1],
                )
            self.joint.joint_net[2]._current_lang = self._current_lang
        else:
            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
        _, scores, words = self.wer.compute()
        self.wer.reset()
        tensorboard_logs.update({'training_batch_wer': scores.float() / words})

    if self.ctc_loss_weight > 0:
        log_probs = self.ctc_decoder(encoder_output=encoded)
        ctc_loss = self.ctc_loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        tensorboard_logs['train_rnnt_loss'] = loss_value
        tensorboard_logs['train_ctc_loss'] = ctc_loss
        if robust_weight > 0.0:
            tensorboard_logs['train_tdt_loss'] = main_loss
            tensorboard_logs['train_robust_loss'] = robust_loss_value
            
        loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
        
        if compute_wer:
            self.ctc_wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            ctc_wer, _, _ = self.ctc_wer.compute()
            self.ctc_wer.reset()
            tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

    loss_value, additional_logs = self.add_interctc_losses(
        loss_value, transcript, transcript_len, compute_wer=compute_wer
    )
    tensorboard_logs.update(additional_logs)
    tensorboard_logs['train_loss'] = loss_value
    
    if AccessMixin.is_access_enabled(self.model_guid):
        AccessMixin.reset_registry(self)

    self.log_dict(tensorboard_logs)

    if self._optim_normalize_joint_txu:
        self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

    return {'loss': loss_value}


def robust_validation_pass(self, batch, batch_idx, dataloader_idx):
    from nemo.core.classes.mixins import AccessMixin
    import torch
    
    if self.is_interctc_enabled():
        AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

    if len(batch) == 5:
        signal, signal_len, transcript, transcript_len, lang = batch
        self._current_lang = lang
        if hasattr(self.joint, 'joint_net') and len(self.joint.joint_net) > 2:
            self.joint.joint_net[2]._current_lang = lang
        if hasattr(self, 'ctc_decoder'):
            self.ctc_decoder._current_lang = lang
    else:
        signal, signal_len, transcript, transcript_len = batch

    # forward() only performs encoder forward
    from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
    if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
        encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
    else:
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
    del signal

    tensorboard_logs = {}
    loss_value = None

    if self.compute_eval_loss:
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

        main_loss = self.loss(
            log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
        )
        
        robust_weight = getattr(self, "robust_loss_weight", 0.0)
        robust_loss_value = 0.0
        if robust_weight > 0.0 and hasattr(self, "robust_loss_module"):
            num_extra = getattr(self.joint, "num_extra_outputs", 0)
            vocab_size_with_blank = joint.shape[-1] - num_extra
            sliced_joint = joint[..., :vocab_size_with_blank]
            
            robust_loss_value = self.robust_loss_module(
                acts=sliced_joint,
                labels=transcript,
                act_lens=encoded_len,
                label_lens=target_length
            )
            
            if self.loss.reduction == 'mean_batch':
                robust_loss_value = robust_loss_value.mean()
            elif self.loss.reduction == 'mean':
                robust_loss_value = torch.div(robust_loss_value, target_length).mean()
            elif self.loss.reduction == 'sum':
                robust_loss_value = robust_loss_value.sum()
            elif self.loss.reduction == 'mean_volume':
                robust_loss_value = robust_loss_value.sum() / target_length.sum()
                
            loss_value = (1 - robust_weight) * main_loss + robust_weight * robust_loss_value
            tensorboard_logs['val_robust_loss'] = robust_loss_value
            tensorboard_logs['val_tdt_loss'] = main_loss
        else:
            loss_value = main_loss
            
        tensorboard_logs['val_loss'] = loss_value

    if hasattr(self, '_current_lang') and isinstance(self._current_lang, list) and hasattr(self, 'joint') and hasattr(self.joint, 'joint_net') and len(self.joint.joint_net) > 2:
        for b in range(encoded.shape[0]):
            self.joint.joint_net[2]._current_lang = self._current_lang[b]
            self.wer.update(
                predictions=encoded[b:b+1],
                predictions_lengths=encoded_len[b:b+1],
                targets=transcript[b:b+1],
                targets_lengths=transcript_len[b:b+1],
            )
        self.joint.joint_net[2]._current_lang = self._current_lang
    else:
        self.wer.update(
            predictions=encoded,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len,
        )
    wer, wer_num, wer_denom = self.wer.compute()
    self.wer.reset()

    tensorboard_logs['val_wer_num'] = wer_num
    tensorboard_logs['val_wer_denom'] = wer_denom
    tensorboard_logs['val_wer'] = wer

    log_probs = self.ctc_decoder(encoder_output=encoded)
    if self.compute_eval_loss:
        ctc_loss = self.ctc_loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        tensorboard_logs['val_ctc_loss'] = ctc_loss
        tensorboard_logs['val_rnnt_loss'] = loss_value
        loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
        tensorboard_logs['val_loss'] = loss_value
        
    self.ctc_wer.update(
        predictions=log_probs,
        targets=transcript,
        targets_lengths=transcript_len,
        predictions_lengths=encoded_len,
    )
    ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
    self.ctc_wer.reset()
    tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
    tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
    tensorboard_logs['val_wer_ctc'] = ctc_wer

    self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

    loss_value, additional_logs = self.add_interctc_losses(
        loss_value,
        transcript,
        transcript_len,
        compute_wer=True,
        compute_loss=self.compute_eval_loss,
        log_wer_num_denom=True,
        log_prefix="val_",
    )
    if self.compute_eval_loss:
        tensorboard_logs['val_loss'] = loss_value
    tensorboard_logs.update(additional_logs)
    
    if AccessMixin.is_access_enabled(self.model_guid):
        AccessMixin.reset_registry(self)

    return tensorboard_logs


# ── HAINAN Decoding and inference implementation ──
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding

class HainanDecoding(AbstractRNNTDecoding):
    def __init__(self, decoding_cfg, decoder, joint, blank_id: int, supported_punctuation = None, mode="sar"):
        super().__init__(decoding_cfg, decoder, joint, blank_id, supported_punctuation)
        self.mode = mode  # "nar" or "sar"
        self.durations = self.cfg.get("durations", [0, 1, 2, 3, 4])  # Durations for TDT

    def rnnt_decoder_predictions_tensor(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        return_hypotheses: bool = False,
        partial_hypotheses = None,
    ):
        import torch
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        # encoder_output shape: [B, D, T]
        enc = encoder_output.transpose(1, 2)  # [B, T, D]
        batch_size = enc.shape[0]
        device = enc.device

        # 1. Non-Autoregressive (NAR) pass
        pred_zeros = torch.zeros((batch_size, 1, self.decoder.pred_hidden), device=device, dtype=enc.dtype)
        pred_projected = self.joint.project_prednet(pred_zeros)  # [B, 1, joint_hidden]
        enc_projected = self.joint.project_encoder(enc)  # [B, T, joint_hidden]

        # Compute parallel joint
        joint_logits = self.joint.joint_after_projection(enc_projected, pred_projected)  # [B, T, 1, V + 1 + num_extra]
        joint_logits = joint_logits.squeeze(2)  # [B, T, V + 1 + num_extra]

        num_extra = self.joint.num_extra_outputs
        vocab_size_with_blank = joint_logits.shape[-1] - num_extra

        token_logits = joint_logits[..., :vocab_size_with_blank]  # [B, T, V + 1]
        dur_logits = joint_logits[..., vocab_size_with_blank:]  # [B, T, num_extra]

        token_probs = torch.softmax(token_logits, dim=-1)
        dur_probs = torch.softmax(dur_logits, dim=-1)

        tokens = torch.argmax(token_probs, dim=-1)  # [B, T]
        durations = torch.argmax(dur_probs, dim=-1)  # [B, T]

        dur_list = self.durations

        # 1. Initial Hypotheses Generation (either standard greedy NAR or Viterbi DAG search)
        hypotheses = []
        if "viterbi" in self.mode:
            # Algorithm 4: Viterbi Decoding of HAINAN Models
            for b in range(batch_size):
                T_seq = encoded_lengths[b].item()
                
                # In log space to avoid underflow
                token_log_probs = torch.log_softmax(token_logits[b, :T_seq], dim=-1)
                dur_log_probs = torch.log_softmax(dur_logits[b, :T_seq], dim=-1)
                
                best_prob_val, best_prob_idx = token_log_probs.max(dim=-1) # [T]
                
                best_prob = [float('-inf') for _ in range(T_seq + 1)]
                best_prob[0] = 0.0 # log(1.0) = 0.0
                backtrack = [-1 for _ in range(T_seq + 1)]
                
                for target in range(1, T_seq + 1):
                    for idx, n in enumerate(dur_list):
                        if n == 0:
                            continue  # Duration 0 is not allowed in NAR modes to prevent loops
                        source = max(target - n, 0)
                        
                        alpha = best_prob[source]
                        trans_prob_log = best_prob_val[source] + dur_log_probs[source, idx]
                        
                        if alpha + trans_prob_log > best_prob[target]:
                            best_prob[target] = alpha + trans_prob_log
                            backtrack[target] = source
                
                # Backtrack to reconstruct optimal path
                curr = T_seq
                path = []
                while curr > 0:
                    prev = backtrack[curr]
                    if prev == -1:
                        prev = max(0, curr - 1)
                    path.append((prev, curr))
                    curr = prev
                path.reverse()
                
                hyp_seq = []
                timestamps = []
                for prev, curr in path:
                    if prev < T_seq:
                        tok = best_prob_idx[prev].item()
                        if tok != self.blank_id:
                            hyp_seq.append(tok)
                            timestamps.append(prev)
                
                hyp = Hypothesis(score=0.0, y_sequence=hyp_seq, dec_state=None, timestamp=timestamps)
                hypotheses.append(hyp)
        else:
            # Algorithm 2: Standard Non-AR (NAR) greedy decoding
            for b in range(batch_size):
                t = 0
                hyp_seq = []
                timestamps = []
                seq_len = encoded_lengths[b].item()

                while t < seq_len:
                    tok = tokens[b, t].item()
                    dur_idx = durations[b, t].item()
                    dur = dur_list[dur_idx]

                    if tok != self.blank_id:
                        hyp_seq.append(tok)
                        timestamps.append(t)
                    t += max(1, dur)

                hyp = Hypothesis(score=0.0, y_sequence=hyp_seq, dec_state=None, timestamp=timestamps)
                hypotheses.append(hyp)

        if self.mode in ["nar", "viterbi"]:
            return self._finalize_predictions(hypotheses, return_hypotheses)

        # 2. Semi-Autoregressive (SAR) pass / Hypothesis Refinement (Algorithm 3)
        refined_hypotheses = []
        
        # Save original _current_lang if it exists
        original_lang = None
        if hasattr(self.joint, 'joint_net') and len(self.joint.joint_net) > 2:
            original_lang = getattr(self.joint.joint_net[2], '_current_lang', None)

        for b in range(batch_size):
            if isinstance(original_lang, list) and len(original_lang) == batch_size:
                self.joint.joint_net[2]._current_lang = original_lang[b]
                
            hyp = hypotheses[b]
            y_seq = hyp.y_sequence
            t_steps = hyp.timestamp

            if len(y_seq) == 0:
                refined_hypotheses.append(hyp)
                continue

            useful_frames = enc_projected[b, t_steps, :].unsqueeze(0)  # [1, U, H]
            shifted_y = [self.blank_id] + y_seq[:-1]
            shifted_y_tensor = torch.tensor([shifted_y], device=device, dtype=torch.long)
            shifted_y_len = torch.tensor([len(shifted_y)], device=device, dtype=torch.long)

            # Pass to predictor
            dec_out, _, _ = self.decoder(targets=shifted_y_tensor, target_length=shifted_y_len)
            dec_out = dec_out.transpose(1, 2)  # [1, U, D]
            pred_projected_sar = self.joint.project_prednet(dec_out)  # [1, U, H]

            inp = useful_frames + pred_projected_sar  # [1, U, H]

            if hasattr(self.joint, "is_adapter_available") and self.joint.is_adapter_available():
                inp = self.joint.forward_enabled_adapters(inp)

            joint_logits_sar = self.joint.joint_net(inp)  # [1, U, V + 1 + num_extra]
            token_logits_sar = joint_logits_sar[..., :vocab_size_with_blank]  # [1, U, V + 1]
            token_probs_sar = torch.softmax(token_logits_sar, dim=-1)

            refined_tokens = torch.argmax(token_probs_sar, dim=-1).squeeze(0)  # [U]

            refined_seq = []
            refined_timestamps = []
            ref_toks = refined_tokens.tolist()
            if not isinstance(ref_toks, list):
                ref_toks = [ref_toks]

            for idx, r_tok in enumerate(ref_toks):
                if r_tok != self.blank_id:
                    refined_seq.append(r_tok)
                    refined_timestamps.append(t_steps[idx])

            refined_hyp = Hypothesis(score=0.0, y_sequence=refined_seq, dec_state=None, timestamp=refined_timestamps)
            refined_hypotheses.append(refined_hyp)

        # Restore original _current_lang
        if original_lang is not None and hasattr(self.joint, 'joint_net') and len(self.joint.joint_net) > 2:
            self.joint.joint_net[2]._current_lang = original_lang

        return self._finalize_predictions(refined_hypotheses, return_hypotheses)

    def _finalize_predictions(self, hypotheses, return_hypotheses):
        decoded = self.decode_hypothesis(hypotheses)
        if return_hypotheses:
            return decoded
        return [Hypothesis(h.score, h.y_sequence, h.text) for h in decoded]

    def _aggregate_token_confidence(self, *args, **kwargs):
        return []

    def decode_ids_to_langs(self, *args, **kwargs):
        return []

    def decode_ids_to_tokens(self, *args, **kwargs):
        return []

    def decode_tokens_to_lang(self, *args, **kwargs):
        return ""

    def decode_tokens_to_str(self, *args, **kwargs):
        return ""

    def get_words_offsets(self, *args, **kwargs):
        return []

ENABLE_GATED_PREDICTOR = False
ENABLE_FSMN_PREDICTOR = False

# Save original change_decoding_strategy
EncDecHybridRNNTCTCBPEModel._orig_change_decoding_strategy = EncDecHybridRNNTCTCBPEModel.change_decoding_strategy

def robust_change_decoding_strategy(self, decoding_cfg=None, decoder_type: str = None, verbose: bool = True):
    from omegaconf import OmegaConf
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecodingConfig
    from nemo.collections.asr.metrics.wer import WER

    # ── Initialize Gated Context Predictor if requested ──
    global ENABLE_GATED_PREDICTOR
    if globals().get("ENABLE_GATED_PREDICTOR", False):
        for sub_m in self.modules():
            if isinstance(sub_m, StatelessNet):
                if not hasattr(sub_m, "gated_predictor") or sub_m.gated_predictor is None:
                    sub_m.gated_predictor = GatedContextPredictor(emb_dim=sub_m.emb_dim).to(next(self.parameters()).device)
                    if verbose:
                        print("Initialized Gated Context Predictor (Gated Causal 1D Conv) on StatelessNet!")

    # ── Initialize FSMN Context Predictor if requested ──
    global ENABLE_FSMN_PREDICTOR
    if globals().get("ENABLE_FSMN_PREDICTOR", False):
        for sub_m in self.modules():
            if isinstance(sub_m, StatelessNet):
                if not hasattr(sub_m, "fsmn_predictor") or sub_m.fsmn_predictor is None:
                    sub_m.fsmn_predictor = FSMNContextPredictor(emb_dim=sub_m.emb_dim).to(next(self.parameters()).device)
                    if verbose:
                        print("Initialized FSMN Context Predictor (Causal c-FSMN) on StatelessNet!")

    strategy = getattr(decoding_cfg, 'strategy', None) if decoding_cfg is not None else None
    
    global HAINAN_DECODING_STRATEGY
    hainan_strat = globals().get("HAINAN_DECODING_STRATEGY", "none")
    
    if decoder_type in ['hainan_nar', 'hainan_sar', 'hainan_viterbi', 'hainan_viterbi_sar'] or strategy in ['hainan_nar', 'hainan_sar', 'hainan_viterbi', 'hainan_viterbi_sar'] or hainan_strat in ['hainan_nar', 'hainan_sar', 'hainan_viterbi', 'hainan_viterbi_sar']:
        if decoding_cfg is None:
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        mode = "nar"
        effective_strat = decoder_type or strategy or hainan_strat
        if effective_strat == 'hainan_sar':
            mode = "sar"
        elif effective_strat == 'hainan_viterbi':
            mode = "viterbi"
        elif effective_strat == 'hainan_viterbi_sar':
            mode = "viterbi_sar"

        self.decoding = HainanDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            blank_id=self.decoder.blank_idx if hasattr(self.decoder, "blank_idx") else self.tokenizer.tokenizer.vocab_size,
            mode=mode,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )
        if verbose:
            print(f"Switched decoding strategy to HAINAN with mode: {mode.upper()}")
        return

    return self._orig_change_decoding_strategy(decoding_cfg, decoder_type, verbose)

EncDecHybridRNNTCTCBPEModel.change_decoding_strategy = robust_change_decoding_strategy


# ── Custom Stateless Predictors ──
from nemo.collections.asr.parts.submodules.stateless_net import StatelessNet
from gated_context_predictor import GatedContextPredictor
from fsmn_context_predictor import FSMNContextPredictor

# Save and monkeypatch EncDecHybridRNNTCTCBPEModel.__init__ to swap in advanced CachedFSMNTransducerDecoder
EncDecHybridRNNTCTCBPEModel._orig_init = EncDecHybridRNNTCTCBPEModel.__init__

def robust_hybrid_model_init(self, cfg, trainer=None):
    # Call original constructor
    self._orig_init(cfg, trainer=trainer)
    
    # If the advanced FSMN decoder is requested, swap the original decoder
    global ENABLE_FSMN_DECODER
    if globals().get("ENABLE_FSMN_DECODER", False):
        from fsmn_decoder import CachedFSMNTransducerDecoder
        pred_hidden = self.decoder.pred_hidden
        vocab_size = self.decoder.vocab_size
        
        # Configure prednet dict
        prednet_cfg = {
            "pred_hidden": pred_hidden,
            "bottleneck_dim": pred_hidden // 2,
            "num_modules": 2,
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8],
            "dw_kernel_size": 7,
            "dropout": 0.1
        }
        
        # Instantiate the advanced FSMN decoder
        fsmn_dec = CachedFSMNTransducerDecoder(
            prednet=prednet_cfg,
            vocab_size=vocab_size,
            blank_as_pad=True
        ).to(next(self.parameters()).device)
        
        # Swap it!
        self.decoder = fsmn_dec
        print("Successfully swapped original RNNTDecoder with advanced CachedFSMNTransducerDecoder inside model constructor!")

EncDecHybridRNNTCTCBPEModel.__init__ = robust_hybrid_model_init

# Save the original forward of StatelessNet
StatelessNet._orig_forward = StatelessNet.forward

def robust_stateless_net_forward(self, y=None, state=None):
    # Call the original forward to get concatenated embeddings
    out, new_state = self._orig_forward(y, state)
    
    # If the gated predictor block is initialized, pass embeddings through it
    if hasattr(self, "gated_predictor") and self.gated_predictor is not None:
        out = self.gated_predictor(out)
        
    # If the FSMN predictor block is initialized, pass embeddings through it
    if hasattr(self, "fsmn_predictor") and self.fsmn_predictor is not None:
        out = self.fsmn_predictor(out)
        
    return out, new_state

StatelessNet.forward = robust_stateless_net_forward



# Attach custom training and validation pass to EncDecHybridRNNTCTCBPEModel class
EncDecHybridRNNTCTCBPEModel.training_step = robust_training_step
EncDecHybridRNNTCTCBPEModel.validation_pass = robust_validation_pass
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
import hashlib
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import regex
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.core.optim.lr_scheduler import prepare_lr_scheduler
from lightning.pytorch.callbacks import Callback


# def _lhotse_len(self):
#     # Try all known locations safely
#     if hasattr(self, "cuts") and self.cuts is not None:
#         return len(self.cuts)

#     if hasattr(self, "_cutset") and self._cutset is not None:
#         return len(self._cutset)

#     if hasattr(self, "cutset") and self.cutset is not None:
#         return len(self.cutset)

#     raise AttributeError(
#         "Cannot determine length of LhotseSpeechToTextBpeDataset. "
#         "No cuts/cutset attribute found."
#     )

# LhotseSpeechToTextBpeDataset.__len__ = _lhotse_len

def compute_epoch_steps(manifest_path, batch_duration, quadratic_duration, accumulate_grad_batches, world_size=1):
    """
    Compute how many optimizer steps = one full pass through the data.
    Reads NeMo JSONL manifest format (audio_filepath, duration, text).
    Accounts for Lhotse's quadratic duration penalty on long audio.
    """
    total_effective_duration = 0.0
    total_real_duration = 0.0
    num_cuts = 0

    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            d = entry.get("duration", 0.0)
            if d <= 0:
                continue

            if d <= quadratic_duration:
                effective = d
            else:
                effective = d + (d - quadratic_duration) ** 2 / quadratic_duration

            total_effective_duration += effective
            total_real_duration += d
            num_cuts += 1

    batches_per_pass = total_effective_duration / batch_duration
    batches_per_gpu = batches_per_pass / world_size
    optimizer_steps = batches_per_gpu / accumulate_grad_batches
    total_hours = total_real_duration / 3600

    return int(optimizer_steps), int(batches_per_gpu), total_hours

class ManualEpochCounter(Callback):
    def __init__(self, steps_per_epoch: int):
        self.steps_per_epoch = steps_per_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Manually increment epoch when we've completed enough steps
        if trainer.global_step > 0 and trainer.global_step % self.steps_per_epoch == 0:
            trainer.fit_loop.epoch_progress.increment_completed()


class PeriodicCacheClear(Callback):
    def __init__(self, every_n_steps=100):
        self.every_n_steps = every_n_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Only during training, never during validation
        # Only when not inside a CUDA graph capture
        if batch_idx % self.every_n_steps == 0:
            try:
                if not torch.cuda.is_current_stream_capturing():
                    torch.cuda.empty_cache()
            except Exception:
                pass  # silently skip if anything goes wrong

class WeightChangeLogger(Callback):
    def __init__(self, log_every_n_steps: int = 250):
        self.log_every_n_steps = log_every_n_steps
        self._prev_weights = {}
        self.tracked_blocks = {
            "encoder",
            "decoder",
            "joint",
            "ctc_decoder",
        }

    def _snapshot(self, model):
        snapshot = {}
        for block_name in self.tracked_blocks:
            block = getattr(model, block_name, None)
            if block is None:
                continue
            snapshot[block_name] = {
                name: param.detach().cpu().clone()
                for name, param in block.named_parameters()
                if param.requires_grad
            }
        return snapshot

    def _compute_deltas(self, model):
        deltas = {}
        for block_name in self.tracked_blocks:
            block = getattr(model, block_name, None)
            if block is None or block_name not in self._prev_weights:
                continue

            total_delta = 0.0
            total_norm  = 0.0
            prev = self._prev_weights[block_name]

            for name, param in block.named_parameters():
                if not param.requires_grad or name not in prev:
                    continue
                curr = param.detach().cpu()
                total_delta += (curr - prev[name]).norm().item()
                total_norm  += curr.norm().item()

            deltas[block_name] = {
                "abs_delta": total_delta,
                "rel_delta": total_delta / (total_norm + 1e-8),
            }

        return deltas

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step

        if not self._prev_weights:
            self._prev_weights = self._snapshot(pl_module)
            return

        if step % self.log_every_n_steps == 0:
            deltas = self._compute_deltas(pl_module)

            # ── Print to terminal ─────────────────────────────────────────
            logging.info(f"[WeightDelta] step={step}")
            for block_name, metrics in deltas.items():
                logging.info(
                    f"  {block_name:<15} abs={metrics['abs_delta']:>10.4f}  "
                    f"rel={metrics['rel_delta']:>10.6f}"
                )

            # ── Log to TensorBoard / WandB ────────────────────────────────
            for block_name, metrics in deltas.items():
                pl_module.log(f"weight_delta/abs/{block_name}", metrics["abs_delta"],
                              on_step=True, on_epoch=False, rank_zero_only=True)
                pl_module.log(f"weight_delta/rel/{block_name}", metrics["rel_delta"],
                              on_step=True, on_epoch=False, rank_zero_only=True)

            self._prev_weights = self._snapshot(pl_module)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step

        if not self._prev_weights:
            self._prev_weights = self._snapshot(pl_module)
            return

        if step % self.log_every_n_steps == 0:
            deltas = self._compute_deltas(pl_module)

            for block_name, metrics in deltas.items():
                pl_module.log(f"weight_delta/abs/{block_name}", metrics["abs_delta"],
                              on_step=True, on_epoch=False, rank_zero_only=True)
                pl_module.log(f"weight_delta/rel/{block_name}", metrics["rel_delta"],
                              on_step=True, on_epoch=False, rank_zero_only=True)

            self._prev_weights = self._snapshot(pl_module)

def update_nested_dict(main_dict, child_dict):
    for key, value in child_dict.items():
        if isinstance(value, dict) and key in main_dict and isinstance(main_dict[key], dict):
            update_nested_dict(main_dict[key], value)
        else:
            main_dict[key] = value
    
    return main_dict

def _tokenizer_exists(self, tokenizer_dir):
    return (
        os.path.exists(os.path.join(tokenizer_dir, "tokenizer.model")) and
        os.path.exists(os.path.join(tokenizer_dir, "tokenizer.vocab"))
    )

def df_parallel_apply(data, func, backend="loky", *args, **kwargs):
    """Apply a function to a DataFrame column in parallel using joblib.
    
    Parameters:
    - df_column: The DataFrame column to which the function will be applied.
    - func: The function to apply.
    - backend: The parallel backend to use (default is "loky").
    - args: Additional positional arguments to pass to the function.
    - kwargs: Additional keyword arguments to pass to the function.
    
    Returns:
    - A list of results.
    """
    # Extract the relevant column data
    if isinstance(data, pd.DataFrame):
        #For Dataframe
        results = Parallel(n_jobs=-1, backend=backend)(
            delayed(func)(item, *args, **kwargs) for _, item in tqdm(data.iterrows(), total=len(data))
        )
    else:
        #For Series
        if not isinstance(data, list):
            data = data.tolist()
        # Apply the function in parallel using joblib
        results = Parallel(n_jobs=-1, backend=backend)(
            delayed(func)(item, *args, **kwargs) for item in tqdm(data, total=len(data))
        )
    
    return results


LOCAL_CACHE_DIR = "cache_adv_v1"
ASR_TRAINING_CACHE_DIR_NAME = "audio/training"
ASR_TRAINING_NOBROKERK_NEMO_CHECKPOINT_GCS_PATH = "ssl_conformer_large_e178.nemo"


ASR_TRAINING_CACHE_DIR = os.path.join(LOCAL_CACHE_DIR, ASR_TRAINING_CACHE_DIR_NAME)
os.makedirs(ASR_TRAINING_CACHE_DIR, exist_ok=True)

# Get the current script directory
CURRENT_DIR = os.path.dirname(__file__)

# Construct the path to the desired file
CONFORMER_CONFIG_YAML_PATH = 'conformer_hybrid_transducer_ctc.yaml'

tqdm.pandas()

def get_duration(file_path):
    try:
        duration = sox.file_info.duration(file_path)
        return duration
    except sox.core.SoxiError as e:
        return 0

class NemoModel():

    def __init__(self, model_path=None, device="cuda"):

        self.model_loaded = False
        self.curated_data = pd.DataFrame()
        self.tokenizer_trained = False
        self.train_manifest_file_path = None
        self.valid_manifest_file_path = None
        self.train_manifest = None
        self.valid_manifest = None
        if model_path:
            self.load_model(model_path, device)
        

    def inference(self, audio_files, batch_size, return_hypotheses, decoder_type="rnnt"):
        if self.model_loaded:
            # Switch to requested decoder before transcribing
            self.model.change_decoding_strategy(decoder_type=decoder_type)

            # return_hypotheses=False saves memory
            results = self.model.transcribe(
                audio_files,
                batch_size = batch_size,
                return_hypotheses=return_hypotheses
            )

            transcriptions = []
            scores = []

            for result in results:
                if return_hypotheses:
                    transcription = result.text
                    score = result.score.item()
                else:
                    transcription = result
                    score = None
                transcriptions.append(transcription)
                scores.append(score)
            
            return transcriptions, scores
        
        else:
            raise Exception("Load the model first")

    def load_model(self, model_path, device):
        # ── CHANGED: restore as the Hybrid model ─────────────────────────────
        self.model = EncDecHybridRNNTCTCBPEModel.restore_from(
            restore_path=model_path, map_location=device
        )
        # Encoder is kept fully trainable – do NOT freeze here
        self.model_loaded = True
        self.device = device



    def __create_dirs__(self, experiment_name):
        
        self.base_dir = os.path.join(ASR_TRAINING_CACHE_DIR, experiment_name)
        self.manifest_dir = os.path.join(self.base_dir, "manifests")
        self.tokenizer_dir = os.path.join(self.base_dir, "tokenizers")
        self.results_dir = os.path.join(self.base_dir, "results")

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.manifest_dir, exist_ok=True)
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


    @staticmethod
    def __clean_transcription(text):
        text = text.lower()

        # protect language tags first
        text = text.replace("<|en|>", "LANGTAGEN").replace("<|hi|>", "LANGTAGHI")

        # clean the rest (remove unwanted punctuation/symbols)
        text = regex.sub(r'[^\p{L}\p{N}\p{M}\s\U0000200C\U0000200D]', '', text)

        # restore the language tags
        text = text.replace("LANGTAGEN", "<|en|>").replace("LANGTAGHI", "<|hi|>")

        return text


    @staticmethod
    def __process_manifest(row):
        """
        Validate audio and build NeMo manifest entry.
        Ensures mono audio (1-D tensor) for Lhotse compatibility.
        """
        
        wav_file = row.get("chunk_file")
        transcription = row.get("transcription")
        duration = row.get("audio_duration")

        try:
            # -------- Basic validation --------
            if not wav_file or not os.path.isfile(wav_file):
                logging.info(f"[MANIFEST] Missing audio file: {wav_file}")
                return None

            if not transcription or not isinstance(transcription, str):
                logging.info(f"[MANIFEST] Empty transcription for: {wav_file}")
                return None

            if duration is None or duration <= 0:
                logging.info(f"[MANIFEST] Invalid duration for: {wav_file}")
                return None

            if duration >= 120:
                logging.info(f"[MANIFEST] Skipping long audio ({duration:.2f}s): {wav_file}")
                return None

            # -------- Mono check (CRITICAL) --------
            # info = sf.info(wav_file)

            # if info.channels != 1:
            #     logging.info(
            #         f"[MANIFEST] Non-mono audio detected "
            #         f"(channels={info.channels}) → converting to mono (in-place): {wav_file}"
            #     )

            #     audio, sr = sf.read(wav_file, always_2d=True)

            #     # Convert to mono
            #     mono_audio = audio.mean(axis=1)

            #     # Overwrite original file with mono audio
            #     sf.write(wav_file, mono_audio, sr)

            # -------- Valid entry --------
            data = {
                "audio_filepath": wav_file,
                "duration": duration,
                "text": transcription.strip(),
            }
            if "lang" in row and pd.notna(row["lang"]):
                data["lang"] = row["lang"]
            return data

        except Exception as e:
            logging.info(
                f"[MANIFEST] Failed to process audio: {wav_file} | Error: {e}"
            )
            return None


    @staticmethod
    def __build_document_from_manifests(
        data_root: str, manifests: str,
    ):
        if ',' in manifests:
            manifests = manifests.split(',')
        else:
            manifests = [manifests]

        document_dir = os.path.join(data_root, 'text_corpus')
        if not os.path.exists(document_dir):
            os.makedirs(document_dir)

        document_path = os.path.join(document_dir, 'document.txt')

        if os.path.exists(document_path):
            logging.info('Corpus already exists at path : %s', document_path)
            return document_path

        num_lines = 0
        with open(document_path, 'w') as out_writer:
            for manifest in manifests:
                with open(manifest, 'r') as in_reader:
                    for line in in_reader:
                        item = json.loads(line)
                        text = item['text']

                        out_writer.write(text + '\n')
                        out_writer.flush()

                        num_lines += 1

                logging.info(f"Finished extracting manifest : {manifest}")

            logging.info("Finished extracting all manifests ! Number of sentences : {}".format(num_lines))
        return document_path


    @staticmethod
    def __process_data(
        text_path: str,
        dst_folder: str,
        vocab_size: int,
        tokenizer_type: str,
        spe_type: str,
        spe_character_coverage: float,
        spe_train_extremely_large_corpus: bool,
        spe_sample_size: int,
        spe_max_sentencepiece_length: int,
        spe_bos: bool,
        spe_eos: bool,
        spe_pad: bool,
        lower_case: bool,
    ):
        """
        Converts flac to wav and build manifests's json
        Args:
            text_path: source with text lines
            dst_folder: where wav files will be stored
            vocab_size: vocabular size used in encoding the text
            tokenizer_type: type of tokenization to perform - wpe or spe
            spe_type: type of tokenization model used for spe.
            spe_character_coverage: float value between 0 and 1 (as a percentage). For languages with a vast charset,
                can be < 1.0, but for all other languages, it should be set as 1.0
            spe_sample_size: int, default of -1. If positive integer is used, samples the dataset
                by given sample size.
            spe_train_extremely_large_corpus: bool. If dataset is too large, and user has sufficient RAM,
                this flag can be set to try to trained the tokenizer. Will silently fail if it runs out of RAM.
            spe_max_sentencepiece_length: Limits the maximum length of the SentencePiece subword that can be constructed.
                By default, no limit is placed.
            spe_bos: Bool flag, whether to add <s> to SentencePiece tokenizer vocabulary.
            spe_eos: Bool flag, whether to add </s> to SentencePiece tokenizer vocabulary.
            spe_pad: Bool flag, whether to add <pad> to SentencePiece tokenizer vocabulary.
            lower_case: whether to tokenize with lower case character set only (for english)

        Returns:
        """
        if tokenizer_type == 'spe':

            # Prepare directory of tokenizer
            if spe_max_sentencepiece_length > 0:
                tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_{}_v{}_max_{}').format(
                    tokenizer_type, spe_type, vocab_size, spe_max_sentencepiece_length
                )
            else:
                tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_{}_v{}').format(
                    tokenizer_type, spe_type, vocab_size
                )

            if spe_pad:
                tokenizer_dir = f'{tokenizer_dir}_pad'
            if spe_bos:
                tokenizer_dir = f'{tokenizer_dir}_bos'
            if spe_eos:
                tokenizer_dir = f'{tokenizer_dir}_eos'

            if not os.path.exists(tokenizer_dir):
                os.makedirs(tokenizer_dir)

            if os.path.exists(os.path.join(tokenizer_dir, 'tokenizer.model')):
                logging.warning("Model file already exists, overriding old model file !")
                os.remove(os.path.join(tokenizer_dir, 'tokenizer.model'))

            # Build tokenizer
            tokenizer_path, vocab_path = create_spt_model(
                data_file=text_path,
                vocab_size=vocab_size,
                sample_size=spe_sample_size,
                do_lower_case=lower_case,
                output_dir=tokenizer_dir,
                tokenizer_type=spe_type,
                character_coverage=spe_character_coverage,
                train_extremely_large_corpus=spe_train_extremely_large_corpus,
                max_sentencepiece_length=spe_max_sentencepiece_length,
                bos=spe_bos,
                eos=spe_eos,
                pad=spe_pad,
                byte_fallback=False,
                split_by_unicode_script=True,
            )

        else:
            tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_v{}').format(tokenizer_type, vocab_size)

            if not os.path.exists(tokenizer_dir):
                os.makedirs(tokenizer_dir)

            tokenizer = tokenizers.BertWordPieceTokenizer(lowercase=lower_case)

            tokenizer.train(text_path, vocab_size=vocab_size)
            tokenizer.save_model(tokenizer_dir)

        return tokenizer_dir


    @staticmethod
    def __load_and_verify_csv__(df_path, audio_col, transcription_col):
        rename_config = {
            audio_col: "chunk_file",
            transcription_col: "transcription"
        }

        df = pd.read_csv(df_path).rename(columns=rename_config)

        # ── Save and drop rows where chunk_file is NaN ────────────────────────
        nan_mask = df["chunk_file"].isna()
        if nan_mask.any():
            bad_csv = df_path.replace(".csv", "_bad_null_path.csv")
            df[nan_mask].to_csv(bad_csv, index=False)
            print(f"  ↳ {nan_mask.sum()} rows with null chunk_file saved to {bad_csv}")
        df = df[~nan_mask]

        # ── Save and drop rows where audio file does not exist on disk ────────
        df["verify"] = df["chunk_file"].apply(os.path.isfile)
        missing_mask = ~df["verify"]
        if missing_mask.any():
            bad_csv = df_path.replace(".csv", "_bad_missing_file.csv")
            df[missing_mask].to_csv(bad_csv, index=False)
            print(f"  ↳ {missing_mask.sum()} rows with missing audio saved to {bad_csv}")

        total = len(df)
        passed = df["verify"].sum()
        failed = total - passed
        print(f"Loaded {passed} audio from {df_path}, {failed} failed")

        df = df[df["verify"] == True]

        # ── Save and drop rows where transcription is NaN ─────────────────────
        null_trans_mask = df["transcription"].isna()
        if null_trans_mask.any():
            bad_csv = df_path.replace(".csv", "_bad_null_transcription.csv")
            df[null_trans_mask].to_csv(bad_csv, index=False)
            print(f"  ↳ {null_trans_mask.sum()} rows with null transcription saved to {bad_csv}")

        # Preserve audio_duration and lang if present in the CSV
        cols = ["chunk_file", "transcription"]
        if "audio_duration" in df.columns:
            cols.append("audio_duration")
            print(f"  ↳ using existing audio_duration from {df_path}")
            
        if "lang" in df.columns:
            cols.append("lang")
            print(f"  ↳ using existing lang from {df_path}")

        df = df[cols]
        df = df.dropna(subset=["transcription", "chunk_file"])
        return df

    @staticmethod
    def __load_and_verify_json__(json_path):
        import json

        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # Support JSONL or normal JSON
            if content.startswith("{") and "\n" in content:
                records = [json.loads(line) for line in content.splitlines()]
            else:
                records = json.loads(content)
                if isinstance(records, dict):
                    records = [records]

        df = pd.DataFrame(records)

        # Map JSON fields → internal schema (INTENTIONALLY IGNORE duration)
        df = df.rename(
            columns={
                "audio_filepath": "chunk_file",
                "text": "transcription",
            }
        )

        df["verify"] = df["chunk_file"].apply(os.path.isfile)

        total = len(df)
        passed = df["verify"].sum()
        failed = total - passed

        print(f"Loaded {passed} audio from {json_path}, {failed} failed")

        df = df[df["verify"] == True]

        # Do NOT include audio_duration here
        cols = ["chunk_file", "transcription"]
        if "lang" in df.columns:
            cols.append("lang")
        df = df[cols]

        df = df.dropna(subset=["chunk_file", "transcription"])

        return df


    def add_data_for_training(
        self,
        audio_data,
        audio_col="chunk_file",
        transcription_col="transcription"
    ):
        """
        audio_data can be:
          - a single CSV path           → "path/to/data.csv"
          - a single JSON/JSONL path    → "path/to/data.json"
          - a directory                 → scans for *.csv / *.json inside
          - comma-separated paths       → "path/a.csv,path/b.csv,path/c.json"
            (CSV and JSON can be mixed)
        All CSVs must contain 'chunk_file' and 'transcription' columns.
        audio_duration column is preserved from CSV/JSON if present, otherwise computed via sox.
        """

        if not isinstance(audio_data, str):
            raise Exception("give a path or audio dataset")

        # ── Preserve old curated data including audio_duration if present ─────
        if len(self.curated_data):
            cols = ["chunk_file", "transcription"]
            if "audio_duration" in self.curated_data.columns:
                cols.append("audio_duration")
            if "lang" in self.curated_data.columns:
                cols.append("lang")
            old_curated_data = self.curated_data[cols]
        else:
            old_curated_data = self.curated_data

        # ── Split on commas to support multiple input files ───────────────────
        candidate_paths = [p.strip() for p in audio_data.split(",") if p.strip()]

        dfs = []

        if len(candidate_paths) == 1 and os.path.isdir(candidate_paths[0]):
            # ── Directory scan branch ─────────────────────────────────────────
            data_dir = candidate_paths[0]
            csv_files  = glob.glob(os.path.join(data_dir, "*.csv"))
            json_files = glob.glob(os.path.join(data_dir, "*.json"))

            for csv_file in csv_files:
                dfs.append(self.__load_and_verify_csv__(csv_file, audio_col, transcription_col))
            for json_file in json_files:
                dfs.append(self.__load_and_verify_json__(json_file))

            if not dfs:
                raise Exception("No CSV or JSON files found in directory")

        else:
            # ── One or more explicit file paths (single or comma-separated) ───
            for path in candidate_paths:
                if path.endswith(".csv") and os.path.isfile(path):
                    dfs.append(self.__load_and_verify_csv__(path, audio_col, transcription_col))
                elif path.endswith(".json") and os.path.isfile(path):
                    dfs.append(self.__load_and_verify_json__(path))
                else:
                    raise Exception(
                        f"Invalid path or unsupported file type: '{path}'. "
                        "Each entry must be an existing .csv or .json file."
                    )

        # ── Merge, clean, dedup ───────────────────────────────────────────────
        _curated_data = pd.concat(dfs, ignore_index=True)
        _curated_data["transcription"] = _curated_data["transcription"].apply(
            self.__clean_transcription
        )

        new_curated_data = pd.concat([old_curated_data, _curated_data], ignore_index=True)
        new_curated_data = new_curated_data.drop_duplicates(subset=["chunk_file"])
        self.curated_data = new_curated_data

        # ── Duration: use existing column if present, only compute missing ─────
        if "audio_duration" not in self.curated_data.columns:
            print("No audio_duration column found, computing via sox for all rows...")
            self.curated_data["audio_duration"] = df_parallel_apply(
                self.curated_data["chunk_file"], get_duration, "loky"
            )
        else:
            missing_mask = self.curated_data["audio_duration"].isna()
            if missing_mask.any():
                print(f"Computing duration for {missing_mask.sum()} rows missing audio_duration...")
                self.curated_data.loc[missing_mask, "audio_duration"] = df_parallel_apply(
                    self.curated_data.loc[missing_mask, "chunk_file"], get_duration, "loky"
                )
            else:
                print("All rows have audio_duration, skipping sox computation.")

        self.curated_data = self.curated_data.dropna(subset=["audio_duration"])

        len_dataset = len(self.curated_data)
        total_duration = self.curated_data["audio_duration"].sum() / 3600

        print(f"Currently have {len_dataset} chunks totaling {total_duration:.2f} hours")


    def _create_manifest(self, train_size):

        # self.train_manifest_file_path = os.path.join(self.manifest_dir, "train_manifest_main.jsonl")
        # self.valid_manifest_file_path = os.path.join(self.manifest_dir, "valid_manifest_main.jsonl")
        self.train_manifest_file_path = "/home/ubuntu/Vallabh/conformer/parakeet_tdt_training/cache_clean_v1/audio/training/Parakeet_Hybrid/manifests/train_manifest.jsonl"
        self.valid_manifest_file_path = "/home/ubuntu/Vallabh/conformer/parakeet_tdt_training/cache_clean_v1/audio/training/Parakeet_Hybrid/manifests/valid_manifest.jsonl"
        # self.train_manifest_file_path = "/home/ubuntu/Vallabh/conformer/parakeet_tdt_training/cache_v3/audio/training/Parakeet_Hybrid/manifests/train_manifest_9lang_filtered_cleaned.jsonl"
        # self.valid_manifest_file_path = "/home/ubuntu/Vallabh/conformer/parakeet_tdt_training/cache_v3/audio/training/Parakeet_Hybrid/manifests/valid_manifest_sanitized_cleaned.jsonl"
        # ── Reuse existing manifests if present ───────────────────────────────
        if os.path.exists(self.train_manifest_file_path) and \
        os.path.exists(self.valid_manifest_file_path):
            print("Manifests already exist, reusing:")
            print(f"  train: {self.train_manifest_file_path}")
            print(f"  valid: {self.valid_manifest_file_path}")
            return
        curated_data = self.curated_data
        manifest = curated_data.progress_apply(self.__process_manifest, axis=1)
        self.train_manifest, self.valid_manifest = train_test_split(manifest, 
                                                          train_size=train_size, 
                                                          random_state=42)

        self.train_manifest_file_path = os.path.join(self.manifest_dir, "train_manifest.jsonl")
        self.valid_manifest_file_path = os.path.join(self.manifest_dir, "valid_manifest.jsonl")

        with open(self.train_manifest_file_path, mode='w+') as file:
            for entry in self.train_manifest:
                if entry:
                    json.dump(entry, file)
                    file.write('\n')

        with open(self.valid_manifest_file_path, mode='w+') as file:
            for entry in self.valid_manifest:
                if entry:
                    json.dump(entry, file)
                    file.write('\n')


    def _train_tokenizer(self):
    
        data_root = self.tokenizer_dir
        # manifests = self.train_manifest_file_path
        manifests = self.train_manifest_file_path + "," + self.valid_manifest_file_path
        data_file = None
        vocab_size = 512
        tokenizer = "spe"
        spe_type = "unigram"
        spe_character_coverage = 1.0
        spe_sample_size = -1
        spe_train_extremely_large_corpus = True
        spe_max_sentencepiece_length = -1
        spe_bos, spe_eos, spe_pad = False, False, False
        lower_case = False

        if not os.path.exists(data_root):
            os.makedirs(data_root)
        # ── Reuse existing tokenizer if present ───────────────────────────────
        expected_tokenizer_dir = os.path.join(data_root, f"tokenizer_spe_unigram_v{vocab_size}")
        if os.path.exists(os.path.join(expected_tokenizer_dir, "tokenizer.model")) and \
        os.path.exists(os.path.join(expected_tokenizer_dir, "tokenizer.vocab")):
            print(f"Tokenizer already exists, reusing: {expected_tokenizer_dir}")
            self.tokenizer_path = expected_tokenizer_dir
            self.tokenizer_trained = True
            return

        # ── Train fresh tokenizer ─────────────────────────────────────────────
        print("Training tokenizer...")
        print(manifests)

        if manifests:
            text_corpus_path = self.__build_document_from_manifests(data_root, manifests)
        else:
            text_corpus_path = data_file

        tokenizer_path = self.__process_data(
            text_corpus_path,
            data_root,
            vocab_size,
            tokenizer,
            spe_type,
            lower_case=lower_case,
            spe_character_coverage=spe_character_coverage,
            spe_sample_size=spe_sample_size,
            spe_train_extremely_large_corpus=spe_train_extremely_large_corpus,
            spe_max_sentencepiece_length=spe_max_sentencepiece_length,
            spe_bos=spe_bos,
            spe_eos=spe_eos,
            spe_pad=spe_pad,
        )

        print("Serialized tokenizer at location :", tokenizer_path)
        self.tokenizer_path = tokenizer_path
        self.tokenizer_trained = True

        if globals().get("ENABLE_LANGUAGE_TAG", False):
            from language_tag_utils import build_lang_token_map
            manifests = self.train_manifest_file_path + "," + self.valid_manifest_file_path
            build_lang_token_map(self.tokenizer_path, data_root, manifests, vocab_size)


    def _train_model(
        self,
        experiment_name,
        max_epoch,
        max_steps,
        pretrained_model,
        load_decoder_from_pretrained,
        wandb_details,
        config
    ):
        if os.path.isfile(pretrained_model):
            init_from_pretrained_model = pretrained_model
        else:
            raise Exception("pretrained model file_path is empty or not valid")

        training_params = self.__load_ctc_default_trainer_config()
        training_params = update_nested_dict(training_params, config)

        training_params["name"] = experiment_name
        training_params["exp_manager"]["name"] = experiment_name
        training_params["trainer"]["max_epochs"] = max_epoch
        training_params["trainer"]["max_steps"] = max_steps
        training_params["init_from_pretrained_model"] = init_from_pretrained_model

        if wandb_details:
            training_params["exp_manager"]["create_wandb_logger"] = True
            training_params["exp_manager"]["wandb_logger_kwargs"]["name"] = wandb_details.get("name", "default")
            training_params["exp_manager"]["wandb_logger_kwargs"]["project"] = wandb_details.get("project", "default")
        else:
            training_params["exp_manager"]["create_wandb_logger"] = False

        cfg = OmegaConf.create(training_params)

        # ── Compute epoch length ──────────────────────────────────────────────
        world_size = cfg.trainer.devices if cfg.trainer.devices > 0 else torch.cuda.device_count()
        # steps_per_epoch, batches_per_epoch, total_hours = compute_epoch_steps(
        #     manifest_path=self.train_manifest_file_path,
        #     batch_duration=cfg.model.train_ds.batch_duration,
        #     quadratic_duration=cfg.model.train_ds.quadratic_duration,
        #     accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        #     world_size=world_size,
        # )

        # print(f"Dataset: {total_hours:.2f} hours")
        # print(f"Batches per epoch (per GPU): {batches_per_epoch}")
        # print(f"Optimizer steps per epoch:   {steps_per_epoch}")

        # with open("training_stats.txt", "w") as f:
        #     f.write(f"Dataset: {total_hours:.2f} hours\n")
        #     f.write(f"Batches per epoch (per GPU): {batches_per_epoch}\n")
        #     f.write(f"Optimizer steps per epoch: {steps_per_epoch}\n")

        cfg.trainer.max_steps = max_steps if max_steps and max_steps > 0 else -1

        # trainer = pl.Trainer(**cfg.trainer, use_distributed_sampler=False)
        trainer = pl.Trainer(**cfg.trainer)
        exp_manager(trainer, cfg.get("exp_manager", None))

        # ── Instantiate model ─────────────────────────────────────────────────
        asr_model = EncDecHybridRNNTCTCBPEModel(cfg=cfg.model, trainer=trainer)
        
        if globals().get("ENABLE_LANGUAGE_TAG", False):
            from language_tag_utils import patch_model_for_language_tag
            patch_model_for_language_tag(asr_model, self.tokenizer_path)

        # ── Initialize robust auxiliary loss if weight > 0 ──────────────────
        robust_weight = cfg.model.get("robust_loss_weight", 0.0)
        if robust_weight > 0.0:
            robust_loss_name = cfg.model.get("robust_loss_name", "target_robust_transducer")
            robust_loss_kwargs_name = f"{robust_loss_name}_kwargs"
            robust_loss_kwargs = cfg.model.get(robust_loss_kwargs_name, {})
            
            # Resolve and instantiate the robust loss module
            from nemo.collections.asr.losses.rnnt import resolve_rnnt_loss
            # Vocabulary size index is computed by excluding the duration predictions
            num_extra = getattr(asr_model.joint, "num_extra_outputs", 0)
            blank_idx = asr_model.joint.num_classes_with_blank - 1 - num_extra
            
            robust_loss_module = resolve_rnnt_loss(
                loss_name=robust_loss_name,
                blank_idx=blank_idx,
                loss_kwargs=robust_loss_kwargs
            )
            
            # Attach robust loss properties to model
            asr_model.robust_loss_weight = robust_weight
            asr_model.robust_loss_module = robust_loss_module
            logging.info(f"Initialized robust auxiliary loss: {robust_loss_name} with weight {robust_weight}")

        # ── Checkpoint path ───────────────────────────────────────────────────
        #ckpt_path = "/home/shourya_1_nobroker_in/shourya/STT-Nobroker/cache_adv_v1/audio/training/Parakeet_Hybrid_hi_en/results/Parakeet_Hybrid_hi_en/2026-06-17_12-38-58/checkpoints/Parakeet_Hybrid_hi_en--val_wer=0.2031-epoch=3-last.ckpt"
        ckpt_path = None
        FREEZE_ENCODER = 0
        if FREEZE_ENCODER:
            # ── Load weights FIRST ────────────────────────────────────────────────
            if ckpt_path is None:
                print("No checkpoint found — loading encoder weights from SSL pretrained model...")
                pretrained_model_obj = SpeechEncDecSelfSupervisedModel.restore_from(
                    cfg.init_from_pretrained_model
                )
                asr_model.encoder.load_state_dict(
                    pretrained_model_obj.encoder.state_dict(), strict=False
                )
                del pretrained_model_obj
                torch.cuda.empty_cache()
                print("Encoder weights loaded and SSL model freed from memory.")
            else:
                print(f"Loading checkpoint weights from: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                asr_model.load_state_dict(checkpoint["state_dict"], strict=False)

            # ── Freeze ONLY encoder ───────────────────────────────────────────────
            for name, param in asr_model.named_parameters():
                if name.startswith("encoder."):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # ── Disable CUDA graphs ───────────────────────────────────────────────
            # asr_model.encoder.use_cuda_graph = False
            # asr_model.decoding.rnnt_cuda_graphs = False
            # if hasattr(asr_model, "joint"):
            #     asr_model.joint.use_cuda_graph = False

            asr_model.change_decoding_strategy(cfg.model.decoding)

            # ── Optimizer: ONLY trainable params ───────────────────────────────────
            trainable_params = [p for p in asr_model.parameters() if p.requires_grad]

            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=cfg.model.optim.lr,
                betas=tuple(cfg.model.optim.betas),
                weight_decay=cfg.model.optim.weight_decay,
            )
            trainer.optimizers = [optimizer]

            # ── Debug info ─────────────────────────────────────────────────────────
            frozen = sum(p.numel() for p in asr_model.encoder.parameters())
            trainable = sum(p.numel() for p in trainable_params)

            print(asr_model)
            print("Tokenizer vocab size:", asr_model.tokenizer.vocab_size)
            print(f"Frozen encoder params:   {frozen:,}")
            print(f"Trainable other params:  {trainable:,}")

            # ── Callbacks ─────────────────────────────────────────────────────────
            step_ckpt_callback = ModelCheckpoint(
                dirpath=trainer.log_dir,
                filename="step-epoch={epoch}-step={step}",
                every_n_train_steps=8000,
                save_top_k=-1,
                save_on_train_epoch_end=False,
                save_last=False,
            )

            # epoch_counter_callback = ManualEpochCounter(steps_per_epoch=steps_per_epoch)
            weight_change_callback = WeightChangeLogger(log_every_n_steps=250)
            trainer.callbacks.append(weight_change_callback)
            trainer.callbacks.append(step_ckpt_callback)
            # trainer.callbacks.append(epoch_counter_callback)

            # ── Train (NO ckpt_path → fresh optimizer) ─────────────────────────────
            trainer.fit(asr_model)
            

            # ── Test ──────────────────────────────────────────────────────────────
            if (
                hasattr(cfg.model, "test_ds")
                and cfg.model.test_ds.manifest_filepath is not None
            ):
                if asr_model.prepare_test(trainer):
                    trainer.test(asr_model)
        else:
            if ckpt_path is None:
                print("No checkpoint found — loading encoder weights from SSL pretrained model...")
                pretrained_model_obj = SpeechEncDecSelfSupervisedModel.restore_from(
                    cfg.init_from_pretrained_model
                )
                asr_model.encoder.load_state_dict(
                    pretrained_model_obj.encoder.state_dict(), strict=False
                )
                del pretrained_model_obj
                torch.cuda.empty_cache()
                print("Encoder weights loaded and SSL model freed from memory.")
            else:
                print(f"Resuming from checkpoint — skipping SSL encoder load.")

            # ── Encoder setup ─────────────────────────────────────────────────────
            # asr_model.encoder.gradient_checkpointing = True
            # for p in asr_model.encoder.parameters():
            #     p.requires_grad = True
            for name, param in asr_model.named_parameters():
                param.requires_grad = True

            # asr_model.encoder.use_cuda_graph = False
            # asr_model.decoding.rnnt_cuda_graphs = False
            # if hasattr(asr_model, 'joint'):
            #     asr_model.joint.use_cuda_graph = False
            asr_model.change_decoding_strategy(cfg.model.decoding)

            # ── Optimizer ─────────────────────────────────────────────────────────
            trainable_params = list(asr_model.parameters())
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, trainable_params),
                lr=cfg.model.optim.lr,
                betas=tuple(cfg.model.optim.betas),
                weight_decay=cfg.model.optim.weight_decay,
            )
            trainer.optimizers = [optimizer]

            print(asr_model)
            print("Tokenizer vocab size:", asr_model.tokenizer.vocab_size)

            # ── Callbacks ─────────────────────────────────────────────────────────
            step_ckpt_callback = ModelCheckpoint(
                dirpath=trainer.log_dir,
                filename="step-epoch={epoch}-step={step}",
                every_n_train_steps=8000,
                save_top_k=-1,
                save_on_train_epoch_end=False,
                save_last=False,
            )
            # cache_clear_callback = PeriodicCacheClear(every_n_steps=100)
            # epoch_counter_callback = ManualEpochCounter(steps_per_epoch=steps_per_epoch)
            weight_change_callback = WeightChangeLogger(log_every_n_steps=250)  # ← add this
            trainer.callbacks.append(weight_change_callback) 
            trainer.callbacks.append(step_ckpt_callback)
            # trainer.callbacks.append(cache_clear_callback)
            # trainer.callbacks.append(epoch_counter_callback)

            # ── Train ─────────────────────────────────────────────────────────────
            trainer.fit(asr_model, ckpt_path=ckpt_path)

            if (
                hasattr(cfg.model, "test_ds")
                and cfg.model.test_ds.manifest_filepath is not None
            ):
                if asr_model.prepare_test(trainer):
                    trainer.test(asr_model)
        # ── Restore full model from .nemo ─────────────────────────────────────
        # nemo_path = "/training-data/Vallabh/conformer/parakeet_tdt_training/cache_hybrid_v3/audio/training/Parakeet_Hybrid/results/Parakeet_Hybrid/2026-02-23_17-34-18/checkpoints/Parakeet_Hybrid.nemo"

        # print(f"Restoring model from: {nemo_path}")

        # asr_model = EncDecHybridRNNTCTCBPEModel.restore_from(
        #     restore_path=nemo_path
            
        # )

        # print("Model restored successfully.")
        # print("Tokenizer vocab size:", asr_model.tokenizer.vocab_size)
        # asr_model.setup_training_data(cfg.model.train_ds)
        # asr_model.setup_validation_data(cfg.model.validation_ds)
        # FREEZE_ENCODER = 0

        # # ── Freeze logic ───────────────────────────────────────────────────────
        # if FREEZE_ENCODER:
        #     for name, param in asr_model.named_parameters():
        #         if name.startswith("encoder."):
        #             param.requires_grad = False
        #         else:
        #             param.requires_grad = True

        #     print("Encoder frozen.")

        #     trainable_params = [p for p in asr_model.parameters() if p.requires_grad]

        # else:
        #     # Train everything
        #     for param in asr_model.parameters():
        #         param.requires_grad = True

        #     trainable_params = asr_model.parameters()

        # # ── ALWAYS create optimizer (since .nemo does NOT restore it) ──────────
        # optimizer = torch.optim.AdamW(
        #     trainable_params,
        #     lr=cfg.model.optim.lr,
        #     betas=tuple(cfg.model.optim.betas),
        #     weight_decay=cfg.model.optim.weight_decay,
        # )

        # trainer.optimizers = [optimizer]

        # # ── Apply decoding config ───────────────────────────────────────────────
        # asr_model.change_decoding_strategy(cfg.model.decoding)

        # # ── Checkpoint callback ─────────────────────────────────────────────────
        # step_ckpt_callback = ModelCheckpoint(
        #     dirpath=trainer.log_dir,
        #     filename="epoch={epoch}-step={step}-val_wer={val_wer:.4f}",
        #     monitor="val_wer",
        #     mode="min",
        #     save_top_k=3,
        #     save_last=True,
        #     every_n_train_steps=4000,
        #     save_on_train_epoch_end=False,
        # )

        # weight_change_callback = WeightChangeLogger(log_every_n_steps=250)

        # trainer.callbacks.append(step_ckpt_callback)
        # trainer.callbacks.append(weight_change_callback)

        # # ── Train ───────────────────────────────────────────────────────────────
        # trainer.fit(asr_model)
        # if (
        #         hasattr(cfg.model, "test_ds")
        #         and cfg.model.test_ds.manifest_filepath is not None
        # ):
        #     if asr_model.prepare_test(trainer):
        #         trainer.test(asr_model)

        
    def train(
        self,
        experiment_name,
        max_epoch,
        max_steps=-1,
        train_size=0.9,
        pretrained_model="default",
        load_decoder_from_pretrained=False,
        wandb_details=None,
        config={}
    ):
        
        self.__create_dirs__(experiment_name)

        self._create_manifest(train_size)

        self._train_tokenizer()

        self._train_model(
            experiment_name=experiment_name,
            max_epoch=max_epoch,
            max_steps=max_steps,
            pretrained_model=pretrained_model,
            load_decoder_from_pretrained=load_decoder_from_pretrained,
            wandb_details=wandb_details,
            config=config
        )


    def __load_ctc_default_trainer_config(self):

        blank_ctc_train_config = OmegaConf.load(CONFORMER_CONFIG_YAML_PATH)

        training_params = CONFORMER_CTC_DEFAULT_TRAINING_PARAMS

        training_params["model"]["train_ds"]["manifest_filepath"] = self.train_manifest_file_path
        training_params["model"]["validation_ds"]["manifest_filepath"] = self.valid_manifest_file_path
        training_params["model"]["tokenizer"]["dir"] = self.tokenizer_path
        training_params["exp_manager"]["exp_dir"] = self.results_dir

        training_params = update_nested_dict(
            OmegaConf.to_object(blank_ctc_train_config), training_params
        )

        return training_params


    def __load_rnnt_hat_default_trainer_config(self, config):
        pass

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a NeMo ASR model with specified configuration.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="unext_v3",
        help=(
            "Dataset source. Accepts:\n"
            "  - a single CSV path:               path/to/data.csv\n"
            "  - a single JSON/JSONL path:         path/to/data.json\n"
            "  - a directory (scans *.csv/*.json): path/to/dir/\n"
            "  - comma-separated CSV/JSON paths:   path/a.csv,path/b.csv,path/c.json\n"
            "CSVs must contain 'chunk_file' and 'transcription' columns."
        ),
    )
    parser.add_argument("--hub", type=str, default="gcs", help="Hub for the dataset.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--n_layers", type=int, default=17, help="Number of layers in the encoder.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warmup steps for the scheduler.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for the scheduler.")
    parser.add_argument("--devices", type=int, default=-1, help="Number of devices to use for training.")
    parser.add_argument("--val_check_interval", type=float, default=0.25, help="Validation check interval.")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type.")
    parser.add_argument("--strategy", type=str, default="ddp", help="Training strategy.")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision for training: 32, 16, 16-mixed, bf16-mixed")
    parser.add_argument("--log_every_n_steps", type=int, default=250, help="Log every n steps.")
    parser.add_argument("--experiment_name", type=str, default="unext_bifrost_voilla", help="Experiment name.")
    parser.add_argument("--max_epoch", type=int, default=30, help="Maximum number of epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps.")
    parser.add_argument("--train_size", type=float, default=0.9, help="Training size proportion.")
    parser.add_argument("--pretrained_model", type=str, default="default", help="Pretrained model name.")
    parser.add_argument("--load_decoder_from_pretrained", type=bool, default=False, help="Load decoder from pretrained model.")
    parser.add_argument("--wandb_name", type=str, default="broft", help="WandB name.")
    parser.add_argument("--wandb_project", type=str, default="Conformer-Unext", help="WandB project name.")
    parser.add_argument("--enable_wandb", action='store_true', help="Enable WandB logging.")
    
    # ── Robust Transducer Loss Arguments ──────────────────────────────────
    parser.add_argument(
        "--loss_name",
        type=str,
        default="tdt",
        choices=["tdt", "star_transducer", "bypass_transducer", "target_robust_transducer"],
        help="RNN-T Loss type to use. Choices: tdt, star_transducer, bypass_transducer, target_robust_transducer",
    )
    parser.add_argument(
        "--skip_frame_penalty",
        type=float,
        default=0.0,
        help="Skip frame penalty for star and target robust transducer losses.",
    )
    parser.add_argument(
        "--skip_token_penalty",
        type=float,
        default=0.0,
        help="Skip token penalty for bypass and target robust transducer losses.",
    )
    parser.add_argument(
        "--skip_token_mode",
        type=str,
        default="sumexcl",
        choices=["constant", "mean", "max", "maxexcl", "sumexcl"],
        help="Skip token mode for bypass and target robust transducer losses.",
    )
    parser.add_argument(
        "--cast_to_float32",
        action="store_true",
        default=True,
        help="Force cast logprobs/logits to float32 before k2 graph-based loss computation.",
    )
    parser.add_argument(
        "--robust_loss_weight",
        type=float,
        default=0.0,
        help="Weight of the auxiliary robust loss. If > 0, robust loss will be computed and combined with TDT/RNNT.",
    )
    parser.add_argument(
        "--robust_loss_name",
        type=str,
        default="target_robust_transducer",
        choices=["star_transducer", "bypass_transducer", "target_robust_transducer"],
        help="Robust Transducer Loss type to use as auxiliary loss.",
    )
    parser.add_argument(
        "--hainan_masking_prob",
        type=float,
        default=0.5,
        help="Stochastic predictor masking probability for HAINAN model training. If <= 0, HAINAN masking is disabled.",
    )
    parser.add_argument(
        "--hainan_decoding_strategy",
        type=str,
        default="none",
        choices=["none", "hainan_nar", "hainan_sar", "hainan_viterbi", "hainan_viterbi_sar"],
        help="Force HAINAN decoding strategy for evaluation and validation (hainan_nar, hainan_sar, hainan_viterbi, hainan_viterbi_sar).",
    )
    parser.add_argument(
        "--enable_gated_predictor",
        action="store_true",
        default=False,
        help="Enable Gated Causal 1D Convolutional Predictor (GCP) for accuracy boost on large/multilingual datasets.",
    )
    parser.add_argument(
        "--enable_fsmn_predictor",
        action="store_true",
        default=False,
        help="Enable Causal FSMN Predictor for state-of-the-art accuracy-latency on large/multilingual datasets.",
    )
    parser.add_argument(
        "--enable_fsmn_decoder",
        action="store_true",
        default=False,
        help="Enable advanced MossFormer2-based CachedFSMNTransducerDecoder as a drop-in replacement for the original LSTM/RNNTDecoder.",
    )
    parser.add_argument(
        "--enable_language_tag",
        action="store_true",
        default=False,
        help="Enable language-specific tokenizer mapping and projection layers.",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    global ENABLE_GATED_PREDICTOR
    ENABLE_GATED_PREDICTOR = args.enable_gated_predictor
    global ENABLE_FSMN_PREDICTOR
    ENABLE_FSMN_PREDICTOR = args.enable_fsmn_predictor
    global ENABLE_FSMN_DECODER
    ENABLE_FSMN_DECODER = args.enable_fsmn_decoder
    global HAINAN_DECODING_STRATEGY
    HAINAN_DECODING_STRATEGY = args.hainan_decoding_strategy
    global ENABLE_LANGUAGE_TAG
    ENABLE_LANGUAGE_TAG = args.enable_language_tag
    if ENABLE_LANGUAGE_TAG:
        from language_tag_utils import patch_dataset_for_language_tag
        patch_dataset_for_language_tag()
        
    nemo_model = NemoModel()

    # Pass dataset_name directly – add_data_for_training handles all formats:
    # single file, directory, or comma-separated list of files
    nemo_model.add_data_for_training(
        args.dataset_name,
        audio_col="chunk_file",
        transcription_col="transcription",
    )
    
    trainer_strategy = args.strategy
    if args.devices == 1 and trainer_strategy == "ddp":
        trainer_strategy = "auto"
        print("INFO: Dynamically switching training strategy from DDP to 'auto' because only 1 GPU/device is requested.")

    config = {
        "model": {
            "train_ds": {
                "batch_size": args.batch_size,
                "max_duration": 20,        
                "pin_memory": True,
                "num_workers": 8,
                "shuffle": True,
            },
            "validation_ds": {
                "batch_size": args.batch_size,
                "shuffle": False,
                "num_workers": 8,
                "pin_memory": True,
            },
            "encoder": {
                "n_layers": args.n_layers,
            },
        },
        "trainer": {
            "devices": args.devices,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "val_check_interval": args.val_check_interval,
            "accelerator": args.accelerator,
            "strategy": trainer_strategy,
            "precision": args.precision,
            "log_every_n_steps": args.log_every_n_steps,
        }
    }

    # ── Dynamic Robust Loss overrides ─────────────────────────────────────
    if args.loss_name in ["star_transducer", "bypass_transducer", "target_robust_transducer"]:
        loss_kwargs_name = f"{args.loss_name}_kwargs"
        loss_kwargs = {
            "cast_to_float32": args.cast_to_float32,
            "use_grid_implementation": True,
        }
        if args.loss_name in ["star_transducer", "target_robust_transducer"]:
            loss_kwargs["skip_frame_penalty"] = args.skip_frame_penalty
        if args.loss_name in ["bypass_transducer", "target_robust_transducer"]:
            loss_kwargs["skip_token_penalty"] = args.skip_token_penalty
            loss_kwargs["skip_token_mode"] = args.skip_token_mode
            
        config["model"]["loss"] = {
            "loss_name": args.loss_name,
            loss_kwargs_name: loss_kwargs
        }
        # Disable fused loss & WER as k2 graph losses do not support fused joint kernel
        config["model"]["joint"] = {
            "fuse_loss_wer": False,
            "num_extra_outputs": 0
        }
        # Change model type from TDT to standard RNNT for decoding and evaluation
        config["model"]["decoding"] = {
            "strategy": "greedy",
            "model_type": "rnnt"
        }

    # ── Robust auxiliary loss combined with TDT/RNNT overrides ───────────
    if args.robust_loss_weight > 0.0:
        robust_loss_kwargs_name = f"{args.robust_loss_name}_kwargs"
        robust_loss_kwargs = {
            "cast_to_float32": args.cast_to_float32,
            "use_grid_implementation": True,
        }
        if args.robust_loss_name in ["star_transducer", "target_robust_transducer"]:
            robust_loss_kwargs["skip_frame_penalty"] = args.skip_frame_penalty
        if args.robust_loss_name in ["bypass_transducer", "target_robust_transducer"]:
            robust_loss_kwargs["skip_token_penalty"] = args.skip_token_penalty
            robust_loss_kwargs["skip_token_mode"] = args.skip_token_mode
            
        config["model"]["robust_loss_weight"] = args.robust_loss_weight
        config["model"]["robust_loss_name"] = args.robust_loss_name
        config["model"][robust_loss_kwargs_name] = robust_loss_kwargs
        
        # Disable fused loss & WER as k2 graph losses do not support fused joint kernel
        if "joint" not in config["model"]:
            config["model"]["joint"] = {}
        config["model"]["joint"]["fuse_loss_wer"] = False

    # ── HAINAN Config overrides ──────────────────────────────────────────
    if "joint" not in config["model"]:
        config["model"]["joint"] = {}
    if args.hainan_masking_prob > 0.0:
        config["model"]["joint"]["masking_prob"] = args.hainan_masking_prob
        
    if args.hainan_decoding_strategy != "none":
        if "decoding" not in config["model"]:
            config["model"]["decoding"] = {}
        # Set decoding strategy to 'greedy_batch' to bypass NeMo's strict TDT constructor validation check
        # (which only allows 'greedy', 'greedy_batch', 'beam', 'maes').
        # Our intercepted change_decoding_strategy will dynamically override it to Hainan SAR/NAR.
        config["model"]["decoding"]["strategy"] = "greedy_batch"
        config["model"]["decoding"]["model_type"] = "tdt"

    wandb_details = {
        "name": args.wandb_name,
        "project": args.wandb_project
    } if args.enable_wandb else None

    nemo_model.train(
        experiment_name=args.experiment_name,
        max_epoch=args.max_epoch,
        max_steps=args.max_steps,
        train_size=args.train_size,
        pretrained_model=args.pretrained_model,
        load_decoder_from_pretrained=False,
        wandb_details=wandb_details,
        config=config
    )

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()

