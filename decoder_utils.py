import numpy as np
import pandas as pd
import torch
import soundfile as sf
from typing import List, Dict, Optional, Tuple
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
import re
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from jiwer import cer, wer
import torch.nn.functional as F
import torch.nn as nn
import json
import os

import nemo.collections.asr.data.audio_to_text as nemo_data
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset


def patch_dataset_for_language_tag():
   print("Patching NeMo datasets to yield 'lang' variable...")
   original_process_sample = nemo_data._AudioTextDataset._process_sample


   def patched_process_sample(self, index):
       sample = self.manifest_processor.collection[index]
       lang = getattr(sample, 'lang', None)
       if lang is None and not getattr(self, 'return_sample_id', False):
           raise ValueError(
               f"Language tag 'lang' is missing from sample {index}! "
               f"You must include a 'lang' column in your CSV or a 'lang' key in your JSONL "
               f"when running with --enable_language_tag."
           )
          
       offset = sample.offset if sample.offset is not None else 0


       features = self.featurizer.process(
           sample.audio_file,
           offset=offset,
           duration=sample.duration,
           trim=self.trim,
           orig_sr=sample.orig_sr,
           channel_selector=self.channel_selector,
       )
       f, fl = features, torch.tensor(features.shape[0]).long()
       t, tl = self.manifest_processor.process_text_by_sample(sample=sample)


       if self.return_sample_id:
           output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), index
       else:
           output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), lang


       return output


   nemo_data._AudioTextDataset._process_sample = patched_process_sample


   original_speech_collate_fn = nemo_data._speech_collate_fn


   def patched_speech_collate_fn(batch, pad_id):
       has_lang = len(batch[0]) == 5 and isinstance(batch[0][4], str)


       if not has_lang:
           return original_speech_collate_fn(batch, pad_id)


       langs = [b[4] for b in batch]
       batch_without_lang = [b[:4] for b in batch]
       result = original_speech_collate_fn(batch_without_lang, pad_id)
       return result[0], result[1], result[2], result[3], langs


   nemo_data._speech_collate_fn = patched_speech_collate_fn


   original_collate_fn = _AudioTextDataset._collate_fn


   def patched_collate_fn(self, batch):
       has_lang = len(batch[0]) == 5 and isinstance(batch[0][4], str)


       if not has_lang:
           return original_collate_fn(self, batch)


       langs = [b[4] for b in batch]
       batch_without_lang = [b[:4] for b in batch]
       result = original_speech_collate_fn(batch_without_lang, pad_id=self.manifest_processor.pad_id)
       return result[0], result[1], result[2], result[3], langs


   _AudioTextDataset._collate_fn = patched_collate_fn
   _AudioTextDataset.collate_fn = _AudioTextDataset._collate_fn



def build_lang_token_map(tokenizer_path, data_root, manifests, vocab_size):
   lang_map_path = os.path.join(tokenizer_path, "lang_token_map.json")
   if os.path.exists(lang_map_path):
       print(f"Language token map already exists: {lang_map_path}")
       return
      
   print("Building language token map...")
   import sentencepiece as spm
   sp_model = spm.SentencePieceProcessor()
   sp_model.Load(os.path.join(tokenizer_path, "tokenizer.model"))
  
   lang_to_tokens = {}
   for manifest_path in manifests.split(','):
       if not os.path.exists(manifest_path):
           continue
       with open(manifest_path, 'r', encoding='utf-8') as f:
           for line in f:
               entry = json.loads(line)
               lang = entry.get("lang", "en")
               text = entry.get("text", "")
               tokens = sp_model.EncodeAsIds(text)
              
               if lang not in lang_to_tokens:
                   lang_to_tokens[lang] = set()
               lang_to_tokens[lang].update(tokens)
  
   # Map them and include blank ID (which is vocab_size in NeMo)
   lang_map = {}
   for lang, token_set in lang_to_tokens.items():
       token_list = sorted(list(token_set))
       # Explicitly add blank ID
       if vocab_size not in token_list:
           token_list.append(vocab_size)
      
       lang_map[lang] = {
           "local_to_global": token_list,
           "global_to_local": {str(g): l for l, g in enumerate(token_list)}
       }
  
   with open(lang_map_path, 'w', encoding='utf-8') as f:
       json.dump(lang_map, f, indent=4)
   print(f"Saved language token map to {lang_map_path}")


class LangCTCDecoder(nn.Module):
   def __init__(self, encoder_hidden_dim, lang_map, global_vocab_size):
       super().__init__()
       self.lang_map = lang_map
       self.global_vocab_size = global_vocab_size
       self.heads = nn.ModuleDict({
           lang: nn.Linear(encoder_hidden_dim, len(mapping["local_to_global"]))
           for lang, mapping in lang_map.items()
       })
       self.fallback_lang = list(lang_map.keys())[0] if lang_map else "en"
       self._current_lang = None

   def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

   def unfreeze(self, partial: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.train()

   def forward(self, encoder_output, **kwargs):
        lang = self._current_lang
        device = encoder_output.device
        dtype = encoder_output.dtype

        # encoder_output is (B, D, T), transpose to (B, T, D)
        x = encoder_output.transpose(1, 2)
    
        # Inference case
        if isinstance(lang, str) or lang is None:
            if lang not in self.heads:
                lang = self.fallback_lang
            local_logits = self.heads[lang](x)  # (B, T, local_V)

            global_logits = torch.full(
                (*local_logits.shape[:-1], self.global_vocab_size),
                fill_value=-1e9, device=device, dtype=local_logits.dtype
            )
            indices = torch.tensor(self.lang_map[lang]["local_to_global"], device=device)
            indices_expanded = indices.expand(*local_logits.shape[:-1], -1)
            global_logits.scatter_(-1, indices_expanded, local_logits)
            return global_logits.float()  # (B, T, global_V)

        # Training case — group by lang, one batched head call per language
        groups = {}
        for i, l in enumerate(lang):
            if l not in self.heads:
                l = self.fallback_lang
            groups.setdefault(l, []).append(i)

        global_logits = torch.full(
            (*x.shape[:-1], self.global_vocab_size),
            fill_value=-1e9, device=device, dtype=dtype
        )

        for l, idxs in groups.items():
            idx_t = torch.tensor(idxs, device=device)
            sub_x = x[idx_t]                          # (b, T, D)
            local_logits = self.heads[l](sub_x)        # (b, T, local_V)

            indices = torch.tensor(self.lang_map[l]["local_to_global"], device=device)
            indices_expanded = indices.expand(*local_logits.shape[:-1], -1)

            sub_global = global_logits[idx_t].clone()
            sub_global.scatter_(-1, indices_expanded, local_logits.to(sub_global.dtype))
            global_logits[idx_t] = sub_global

        return global_logits.float()

#    def forward(self, encoder_output, **kwargs):
#        lang = self._current_lang
#        device = encoder_output.device
#        dtype = encoder_output.dtype
      
#        # encoder_output is (B, D, T), transpose to (B, T, D)
#        x = encoder_output.transpose(1, 2)
      
#        # Inference case
#        if isinstance(lang, str) or lang is None:
#            if lang not in self.heads:
#                lang = self.fallback_lang
#            local_logits = self.heads[lang](x)  # (B, T, local_V)
          
#            global_logits = torch.full(
#                (*local_logits.shape[:-1], self.global_vocab_size),
#                fill_value=-1e9, device=device, dtype=local_logits.dtype
#            )
#            indices = torch.tensor(self.lang_map[lang]["local_to_global"], device=device)
#            indices_expanded = indices.expand(*local_logits.shape[:-1], -1)
#            global_logits.scatter_(-1, indices_expanded, local_logits)
#            return global_logits.float()  # (B, T, global_V)


#        # Training case
#        local_logits_list = []
#        for i, l in enumerate(lang):
#            if l not in self.heads:
#                l = self.fallback_lang
#            local_logits_list.append((l, self.heads[l](x[i])))


#        global_logits = torch.full(
#            (*x.shape[:-1], self.global_vocab_size),
#            fill_value=-1e9, device=device, dtype=local_logits_list[0][1].dtype
#     )
#        for i, (l, local_logits) in enumerate(local_logits_list):
#            indices = torch.tensor(self.lang_map[l]["local_to_global"], device=device)
#            indices_expanded = indices.expand(*local_logits.shape[:-1], -1)
#            global_logits[i].scatter_(-1, indices_expanded, local_logits)


#        return global_logits.float()


class SingleLangJointPostNet(nn.Module):
    """
    Single-language replacement for LangJointPostNet.
    One Linear head, scatters into global vocab space.
    """
    def __init__(self, joint_hidden_dim, local_to_global, global_vocab_size):
        super().__init__()
        self.local_to_global = local_to_global
        self.global_vocab_size = global_vocab_size
        self.head = nn.Linear(joint_hidden_dim, len(local_to_global))

    def forward(self, joint_hidden):
        device = joint_hidden.device
        local_logits = self.head(joint_hidden)  

        global_logits = torch.full(
            (*local_logits.shape[:-1], self.global_vocab_size),
            fill_value=-1e9, device=device, dtype=local_logits.dtype
        )
        indices = torch.tensor(self.local_to_global, device=device)
        indices_expanded = indices.expand(*local_logits.shape[:-1], -1)
        global_logits.scatter_(-1, indices_expanded, local_logits)
        return global_logits.float()


# class LangStatelessDecoder(nn.Module):
#     def __init__(self, lang_decoders: dict):
#         super().__init__()
#         self.decoders = nn.ModuleDict(lang_decoders)
#         self.fallback_lang = list(lang_decoders.keys())[0]
#         self._current_lang = None

#     def forward(self, targets, target_length, *args, **kwargs):
#         lang = self._current_lang

#         # Inference case: single string or None
#         if isinstance(lang, str) or lang is None:
#             if lang is None or lang not in self.decoders:
#                 lang = self.fallback_lang
#             return self.decoders[lang](targets=targets, target_length=target_length, *args, **kwargs)

#         # Group sample indices by resolved language
#         groups = {}
#         for i, l in enumerate(lang):
#             if l not in self.decoders:
#                 l = self.fallback_lang
#             groups.setdefault(l, []).append(i)

#         outputs = []
#         order = []
#         for l, idxs in groups.items():
#             idx_t = torch.tensor(idxs, device=targets.device)
#             out = self.decoders[l](
#                 targets=targets[idx_t],
#                 target_length=target_length[idx_t],
#                 *args, **kwargs
#             )
#             outputs.append(out)
#             order.extend(idxs)

#         # out is (decoder_output, target_length, states) — stack decoder_output
#         decoder_outputs = torch.cat([o[0] for o in outputs], dim=0)
#         target_lengths  = torch.cat([o[1] for o in outputs], dim=0)
#         states = [o[2] for o in outputs]

#         # Undo the grouping permutation so outputs line up with original batch order
#         inv_order = torch.argsort(torch.tensor(order, device=decoder_outputs.device))
#         decoder_outputs = decoder_outputs[inv_order]
#         target_lengths = target_lengths[inv_order]

#         return decoder_outputs, target_lengths, states

    # def __getattr__(self, name):
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         lang = self._current_lang or self.fallback_lang
    #         active = self.fallback_lang if (lang is None or (isinstance(lang, list) and lang[0] not in self.decoders)) else (lang[0] if isinstance(lang, list) else lang)
    #         return getattr(self.decoders[active], name)


# class LangRNNTJoint(nn.Module):
#     def __init__(self, lang_joints: dict):
#         super().__init__()
#         self.joints = nn.ModuleDict(lang_joints)
#         self.fallback_lang = list(lang_joints.keys())[0]
#         self._current_lang = None

#     def forward(self, encoder_outputs, decoder_outputs, *args, **kwargs):
#         lang = self._current_lang

#         # Inference case: single string or None
#         if isinstance(lang, str) or lang is None:
#             if lang is None or lang not in self.joints:
#                 lang = self.fallback_lang
#             return self.joints[lang](
#                 encoder_outputs=encoder_outputs,
#                 decoder_outputs=decoder_outputs,
#                 *args, **kwargs
#             )

#         # Group sample indices by resolved language
#         groups = {}
#         for i, l in enumerate(lang):
#             if l not in self.joints:
#                 l = self.fallback_lang
#             groups.setdefault(l, []).append(i)

#         outputs = []
#         order = []
#         for l, idxs in groups.items():
#             idx_t = torch.tensor(idxs, device=encoder_outputs.device)
#             out = self.joints[l](
#                 encoder_outputs=encoder_outputs[idx_t],
#                 decoder_outputs=decoder_outputs[idx_t],
#                 *args, **kwargs
#             )
#             outputs.append(out)
#             order.extend(idxs)

#         result = torch.cat(outputs, dim=0)
#         inv_order = torch.argsort(torch.tensor(order, device=result.device))
#         return result[inv_order]

#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             lang = self._current_lang or self.fallback_lang
#             active = self.fallback_lang if (lang is None or isinstance(lang, list)) else lang
#             if isinstance(lang, list):
#                 active = lang[0] if lang[0] in self.joints else self.fallback_lang
#             return getattr(self.joints[active], name)


# def patch_model_for_language_tag(asr_model, tokenizer_dir):
#     print("Patching model components with Language Projection Layers...")
#     lang_map_path = os.path.join(tokenizer_dir, "lang_token_map.json")
#     if not os.path.exists(lang_map_path):
#         raise FileNotFoundError(f"Cannot patch model because {lang_map_path} does not exist.")

#     with open(lang_map_path, 'r', encoding='utf-8') as f:
#         lang_map = json.load(f)

#     import copy
#     ctc_lang_map = copy.deepcopy(lang_map)
#     ctc_global_vocab_size = asr_model.tokenizer.vocab_size + 1

#     num_extra = getattr(asr_model.joint, "num_extra_outputs", 0) if hasattr(asr_model, "joint") else 0
#     if hasattr(asr_model, "joint"):
#         global_vocab_size = asr_model.joint.num_classes_with_blank + num_extra
#     else:
#         global_vocab_size = asr_model.tokenizer.vocab_size + 1

#     if hasattr(asr_model, "joint"):
#         for lang in lang_map:
#             for extra_idx in range(1, num_extra + 1):
#                 idx_to_add = asr_model.joint.num_classes_with_blank - 1 + extra_idx
#                 if idx_to_add not in lang_map[lang]["local_to_global"]:
#                     lang_map[lang]["local_to_global"].append(idx_to_add)
#                     lang_map[lang]["global_to_local"][str(idx_to_add)] = (
#                         len(lang_map[lang]["local_to_global"]) - 1
#                     )

#     hidden_dim = asr_model.cfg.joint.jointnet.joint_hidden if hasattr(asr_model.cfg, "joint") else 1024
#     device = next(asr_model.parameters()).device

#     # --- Patch decoder: deepcopy pretrained weights, NO reset ---
#     if hasattr(asr_model, "decoder"):
#         lang_decoders = {
#             lang: copy.deepcopy(asr_model.decoder)
#             for lang in lang_map.keys()
#         }
#         asr_model.decoder = LangStatelessDecoder(lang_decoders).to(device)

#     if hasattr(asr_model, "joint") and hasattr(asr_model.joint, "joint_net"):
#         lang_joints = {}
#         for lang in lang_map.keys():
#             joint = copy.deepcopy(asr_model.joint)
#             # Only the head is new — init it fresh, leave enc/pred/ReLU/Dropout as-is
#             joint.joint_net[2] = SingleLangJointPostNet(
#                 joint_hidden_dim=hidden_dim,
#                 local_to_global=lang_map[lang]["local_to_global"],
#                 global_vocab_size=global_vocab_size,
#             )
#             # joint_net[2] is randomly initialized by nn.Linear constructor, no explicit reset needed
#             lang_joints[lang] = joint
#         asr_model.joint = LangRNNTJoint(lang_joints).to(device)

    
#     if hasattr(asr_model, "ctc_decoder"):
#         ctc_hidden_dim = asr_model.cfg.encoder.d_model
#         asr_model.ctc_decoder = LangCTCDecoder(
#             ctc_hidden_dim, ctc_lang_map, ctc_global_vocab_size
#         ).to(device)

#     print("Model patching complete!")

# def set_current_lang(asr_model, lang):
#     """Call before each forward pass to route all patched components to the correct lang."""
#     if isinstance(asr_model.decoder, LangStatelessDecoder):
#         asr_model.decoder._current_lang = lang
#     if isinstance(asr_model.joint, LangRNNTJoint):
#         asr_model.joint._current_lang = lang
#     if isinstance(asr_model.ctc_decoder, LangCTCDecoder):
#         asr_model.ctc_decoder._current_lang = lang



class LHUCLangAdapter(nn.Module):
    def __init__(self, hidden_dim, languages, scale_range=2.0, channel_dim=-1):
        super().__init__()
        self.languages = languages
        self.scale_range = scale_range
        self.channel_dim = channel_dim
        self.lang_scale = nn.ParameterDict({
            lang: nn.Parameter(torch.zeros(hidden_dim)) for lang in languages
        })

    def forward(self, h, lang):
        scale = self.scale_range * torch.sigmoid(self.lang_scale[lang])
        if self.channel_dim == -1 or self.channel_dim == h.dim() - 1:
            return h * scale
        shape = [1] * h.dim()
        shape[self.channel_dim] = -1
        return h * scale.view(*shape)

    def get_scale_stats(self):
        stats = {}
        with torch.no_grad():
            for lang, p in self.lang_scale.items():
                scale = self.scale_range * torch.sigmoid(p)
                stats[lang] = (scale - 1.0).abs().mean().item()
        return stats


class LHUCStatelessDecoder(nn.Module):
    def __init__(self, shared_decoder, languages, hidden_dim, channel_dim=1):
        super().__init__()
        self.decoder = shared_decoder  # single shared RNNTDecoder, pretrained, untouched
        self.lhuc = LHUCLangAdapter(hidden_dim, languages, channel_dim=channel_dim)
        self.fallback_lang = languages[0]
        self._current_lang = None

    def forward(self, targets, target_length, *args, **kwargs):
        decoder_output, target_length_out, states = self.decoder(
            targets=targets, target_length=target_length, *args, **kwargs
        )
        lang = self._current_lang

        if isinstance(lang, str) or lang is None:
            l = lang if lang in self.lhuc.languages else self.fallback_lang
            return self.lhuc(decoder_output, l), target_length_out, states

        groups = {}
        for i, l in enumerate(lang):
            if l not in self.lhuc.languages:
                l = self.fallback_lang
            groups.setdefault(l, []).append(i)

        out = decoder_output.clone()
        for l, idxs in groups.items():
            idx_t = torch.tensor(idxs, device=decoder_output.device)
            out[idx_t] = self.lhuc(decoder_output[idx_t], l)
        return out, target_length_out, states

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)


class UnsupervisedTokenMoE(nn.Module):
    """Frame-level, unsupervised expert gating. No language supervision —
    router learns its own specialization axis (intended to capture
    intra-utterance code-switch structure that a static language tag can't)."""
    def __init__(self, hidden_dim, num_experts=2, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.last_gate_probs = None  

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        self.last_gate_probs = gate_probs

        topk_probs, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Dense: compute every expert on every token (fine at num_experts=4, hidden=1024 scale)
        all_expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # (..., num_experts, hidden)

        # Build a full (..., num_experts) weight tensor: zero for non-topk experts
        full_weights = torch.zeros_like(gate_probs)
        full_weights.scatter_(-1, topk_idx, topk_probs)

        out = (all_expert_outs * full_weights.unsqueeze(-1)).sum(dim=-2)
        return out + x
        


class LHUCJointNet(nn.Module):
    """
    pre = Sequential(ReLU, Dropout) — shared, pretrained, untouched.
    lhuc = per-language elementwise scaling (existing mechanism, kept).
    token_moe = NEW: unsupervised, frame-level expert refinement, inserted
                after LHUC's coarse per-language scaling, before the
                per-language output head.
    heads = existing per-language SingleLangJointPostNet, unchanged.
    """
    def __init__(self, base_joint_net, lang_map, joint_hidden_dim, scale_range=2.0,
                 use_token_moe=False, num_experts=4, top_k=2):
        super().__init__()
        self.pre = nn.Sequential(base_joint_net[0], base_joint_net[1])
        self.languages = list(lang_map.keys())
        self.lhuc = LHUCLangAdapter(joint_hidden_dim, self.languages, scale_range=scale_range, channel_dim=-1)

        self.use_token_moe = use_token_moe
        if use_token_moe:
            self.token_moe = UnsupervisedTokenMoE(joint_hidden_dim, num_experts=num_experts, top_k=top_k)

        self.heads = nn.ModuleDict()  # populated by patch function
        self.fallback_lang = self.languages[0]
        self._current_lang = None

    def forward(self, x):
        h = self.pre(x)
        lang = self._current_lang

        if isinstance(lang, str) or lang is None:
            l = lang if lang in self.heads else self.fallback_lang
            h_scaled = self.lhuc(h, l)
            if self.use_token_moe:
                h_scaled = self.token_moe(h_scaled)
            return self.heads[l](h_scaled)

        groups = {}
        for i, l in enumerate(lang):
            if l not in self.heads:
                l = self.fallback_lang
            groups.setdefault(l, []).append(i)

        global_vocab_size = self.heads[self.fallback_lang].global_vocab_size
        out = torch.full((*h.shape[:-1], global_vocab_size), -1e9, device=h.device, dtype=torch.float32)
        for l, idxs in groups.items():
            idx_t = torch.tensor(idxs, device=h.device)
            h_scaled = self.lhuc(h[idx_t], l)
            if self.use_token_moe:
                h_scaled = self.token_moe(h_scaled)
            out[idx_t] = self.heads[l](h_scaled)
        return out

    def get_load_balancing_loss(self):
        if not self.use_token_moe or self.token_moe.last_gate_probs is None:
            return torch.tensor(0.0)
        gate_probs = self.token_moe.last_gate_probs
        mean_gate = gate_probs.reshape(-1, gate_probs.shape[-1]).mean(dim=0)
        return self.token_moe.num_experts * (mean_gate ** 2).sum()
    

# class LHUCJointNet(nn.Module):
#     """
#     base_joint_net = Sequential(ReLU, Dropout, Linear) — confirmed structure.
#     pre = Sequential(ReLU, Dropout), shared, pretrained, untouched.
#     LHUC scales the 1024-dim fused hidden state before the per-language head.
#     """
#     def __init__(self, base_joint_net, lang_map, joint_hidden_dim, scale_range=2.0):
#         super().__init__()
#         self.pre = nn.Sequential(base_joint_net[0], base_joint_net[1])  # ReLU, Dropout — shared
#         self.languages = list(lang_map.keys())
#         self.lhuc = LHUCLangAdapter(joint_hidden_dim, self.languages, scale_range=scale_range, channel_dim=-1)

#         num_extra = 0  # set below by caller if needed
#         self.heads = nn.ModuleDict()  # populated by patch function below
#         self.fallback_lang = self.languages[0]
#         self._current_lang = None

#     def forward(self, x):
#         h = self.pre(x)  # shared trunk, one pass regardless of language
#         lang = self._current_lang

#         if isinstance(lang, str) or lang is None:
#             l = lang if lang in self.heads else self.fallback_lang
#             return self.heads[l](self.lhuc(h, l))

#         groups = {}
#         for i, l in enumerate(lang):
#             if l not in self.heads:
#                 l = self.fallback_lang
#             groups.setdefault(l, []).append(i)

#         global_vocab_size = self.heads[self.fallback_lang].global_vocab_size
#         out = torch.full((*h.shape[:-1], global_vocab_size), -1e9, device=h.device, dtype=torch.float32)
#         for l, idxs in groups.items():
#             idx_t = torch.tensor(idxs, device=h.device)
#             out[idx_t] = self.heads[l](self.lhuc(h[idx_t], l))
#         return out


# def patch_model_for_language_tag(asr_model, tokenizer_dir):
#     print("Patching model with LHUC-conditioned shared decoder/joint...")
#     lang_map_path = os.path.join(tokenizer_dir, "lang_token_map.json")
#     with open(lang_map_path, 'r', encoding='utf-8') as f:
#         lang_map = json.load(f)

#     import copy
#     ctc_lang_map = copy.deepcopy(lang_map)
#     ctc_global_vocab_size = asr_model.tokenizer.vocab_size + 1

#     num_extra = getattr(asr_model.joint, "num_extra_outputs", 0)
#     global_vocab_size = asr_model.joint.num_classes_with_blank + num_extra
#     for lang in lang_map:
#         for extra_idx in range(1, num_extra + 1):
#             idx_to_add = asr_model.joint.num_classes_with_blank - 1 + extra_idx
#             if idx_to_add not in lang_map[lang]["local_to_global"]:
#                 lang_map[lang]["local_to_global"].append(idx_to_add)
#                 lang_map[lang]["global_to_local"][str(idx_to_add)] = len(lang_map[lang]["local_to_global"]) - 1

#     joint_hidden_dim = asr_model.cfg.joint.jointnet.joint_hidden  # 1024
#     pred_hidden_dim = asr_model.cfg.decoder.prednet.pred_hidden   # 768, confirmed
#     device = next(asr_model.parameters()).device
#     languages = list(lang_map.keys())

#     # --- Decoder: single shared instance + LHUC ---
#     asr_model.decoder = LHUCStatelessDecoder(
#         asr_model.decoder, languages, pred_hidden_dim, channel_dim=1  # VERIFY shape first
#     ).to(device)

#     # --- Joint: shared ReLU+Dropout trunk + LHUC + your existing per-lang heads ---
#     lhuc_joint_net = LHUCJointNet(asr_model.joint.joint_net, lang_map, joint_hidden_dim)
#     for lang in languages:
#         lhuc_joint_net.heads[lang] = SingleLangJointPostNet(
#             joint_hidden_dim=joint_hidden_dim,
#             local_to_global=lang_map[lang]["local_to_global"],
#             global_vocab_size=global_vocab_size,
#         )
#     asr_model.joint.joint_net = lhuc_joint_net.to(device)

#     # --- CTC: unchanged, your existing lightweight per-lang heads ---
#     ctc_hidden_dim = asr_model.cfg.encoder.d_model
#     asr_model.ctc_decoder = LangCTCDecoder(ctc_hidden_dim, ctc_lang_map, ctc_global_vocab_size).to(device)

#     print("LHUC patching complete!")

def patch_model_for_language_tag(asr_model, tokenizer_dir, use_token_moe=False, num_experts=4, top_k=2):
    print("Patching model with LHUC-conditioned shared decoder/joint + token-MoE...")
    lang_map_path = os.path.join(tokenizer_dir, "lang_token_map.json")
    with open(lang_map_path, 'r', encoding='utf-8') as f:
        lang_map = json.load(f)

    import copy
    ctc_lang_map = copy.deepcopy(lang_map)
    ctc_global_vocab_size = asr_model.tokenizer.vocab_size + 1

    num_extra = getattr(asr_model.joint, "num_extra_outputs", 0)
    global_vocab_size = asr_model.joint.num_classes_with_blank + num_extra
    for lang in lang_map:
        for extra_idx in range(1, num_extra + 1):
            idx_to_add = asr_model.joint.num_classes_with_blank - 1 + extra_idx
            if idx_to_add not in lang_map[lang]["local_to_global"]:
                lang_map[lang]["local_to_global"].append(idx_to_add)
                lang_map[lang]["global_to_local"][str(idx_to_add)] = len(lang_map[lang]["local_to_global"]) - 1

    joint_hidden_dim = asr_model.cfg.joint.jointnet.joint_hidden
    pred_hidden_dim = asr_model.cfg.decoder.prednet.pred_hidden
    device = next(asr_model.parameters()).device
    languages = list(lang_map.keys())

    # --- Decoder: single shared instance + LHUC (unchanged) ---
    asr_model.decoder = LHUCStatelessDecoder(
        asr_model.decoder, languages, pred_hidden_dim, channel_dim=1
    ).to(device)

    # --- Joint: shared trunk + LHUC + NEW token-MoE + existing per-lang heads ---
    lhuc_joint_net = LHUCJointNet(
        asr_model.joint.joint_net, lang_map, joint_hidden_dim,
        use_token_moe=use_token_moe, num_experts=num_experts, top_k=top_k,
    )
    for lang in languages:
        lhuc_joint_net.heads[lang] = SingleLangJointPostNet(
            joint_hidden_dim=joint_hidden_dim,
            local_to_global=lang_map[lang]["local_to_global"],
            global_vocab_size=global_vocab_size,
        )
    asr_model.joint.joint_net = lhuc_joint_net.to(device)

    # --- CTC: unchanged ---
    ctc_hidden_dim = asr_model.cfg.encoder.d_model
    asr_model.ctc_decoder = LangCTCDecoder(ctc_hidden_dim, ctc_lang_map, ctc_global_vocab_size).to(device)

    print(f"Patching complete! token_moe={use_token_moe}, num_experts={num_experts}, top_k={top_k}")


def set_current_lang(asr_model, lang):
    if isinstance(asr_model.decoder, LHUCStatelessDecoder):
        asr_model.decoder._current_lang = lang
    if isinstance(asr_model.joint.joint_net, LHUCJointNet):
        asr_model.joint.joint_net._current_lang = lang
    if isinstance(asr_model.ctc_decoder, LangCTCDecoder):
        asr_model.ctc_decoder._current_lang = lang