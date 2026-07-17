import os
import io
import json
import yaml
import torch
import torch.nn as nn
import nemo.collections.asr.data.audio_to_text as nemo_data
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from omegaconf import OmegaConf


class LangJointPostNet(nn.Module):
   def __init__(self, joint_hidden_dim, lang_map, global_vocab_size):
       super().__init__()
       self.lang_map = lang_map
       self.global_vocab_size = global_vocab_size
       self.heads = nn.ModuleDict({
           lang: nn.Linear(joint_hidden_dim, len(mapping["local_to_global"]))
           for lang, mapping in lang_map.items()
       })
       self.fallback_lang = list(lang_map.keys())[0] if lang_map else "en"
       self._current_lang = None


   def forward(self, joint_hidden, lang=None):
       if lang is None:
           lang = self._current_lang


       device = joint_hidden.device
       dtype = joint_hidden.dtype
      
       # Inference case: single string
       if isinstance(lang, str):
           if lang not in self.heads:
               lang = self.fallback_lang
           local_logits = self.heads[lang](joint_hidden)
          
           global_logits = torch.full(
               (*local_logits.shape[:-1], self.global_vocab_size),
               fill_value=-1e9, device=device, dtype=local_logits.dtype
           )
           indices = torch.tensor(self.lang_map[lang]["local_to_global"], device=device)
           indices_expanded = indices.expand(*local_logits.shape[:-1], -1)
           global_logits.scatter_(-1, indices_expanded, local_logits)
           return global_logits.float()


       # Training case: list of strings
       local_logits_list = []
       for i, l in enumerate(lang):
           if l not in self.heads:
               l = self.fallback_lang
           local_logits_list.append((l, self.heads[l](joint_hidden[i])))


       global_logits = torch.full(
           (*joint_hidden.shape[:-1], self.global_vocab_size),
           fill_value=-1e9, device=device, dtype=local_logits_list[0][1].dtype
       )
       for i, (l, local_logits) in enumerate(local_logits_list):
           indices = torch.tensor(self.lang_map[l]["local_to_global"], device=device)
           indices_expanded = indices.expand(*local_logits.shape[:-1], -1)
           global_logits[i].scatter_(-1, indices_expanded, local_logits)


       return global_logits.float()


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


       # Training case
       local_logits_list = []
       for i, l in enumerate(lang):
           if l not in self.heads:
               l = self.fallback_lang
           local_logits_list.append((l, self.heads[l](x[i])))


       global_logits = torch.full(
           (*x.shape[:-1], self.global_vocab_size),
           fill_value=-1e9, device=device, dtype=local_logits_list[0][1].dtype
       )
       for i, (l, local_logits) in enumerate(local_logits_list):
           indices = torch.tensor(self.lang_map[l]["local_to_global"], device=device)
           indices_expanded = indices.expand(*local_logits.shape[:-1], -1)
           global_logits[i].scatter_(-1, indices_expanded, local_logits)


       return global_logits.float()  # (B, T, global_V)


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




def patch_model_for_language_tag(asr_model, tokenizer_dir):
   print("Patching model components with Language Projection Layers...")
   lang_map_path = os.path.join(tokenizer_dir, "lang_token_map.json")
   if not os.path.exists(lang_map_path):
       raise FileNotFoundError(f"Cannot patch model because {lang_map_path} does not exist.")
      
   with open(lang_map_path, 'r', encoding='utf-8') as f:
       lang_map = json.load(f)


   import copy
   ctc_lang_map = copy.deepcopy(lang_map)
   ctc_global_vocab_size = asr_model.tokenizer.vocab_size + 1


   # For TDT/joint models, compute global vocab size including blank and extras
   num_extra = getattr(asr_model.joint, "num_extra_outputs", 0) if hasattr(asr_model, "joint") else 0
   # Fallback to standard tokenizer if no joint
   if hasattr(asr_model, "joint"):
       global_vocab_size = asr_model.joint.num_classes_with_blank + num_extra
   else:
       global_vocab_size = asr_model.tokenizer.vocab_size + 1 # +1 for blank


   # Ensure extra outputs and blank (if not already) are in local_to_global
   if hasattr(asr_model, "joint"):
       for lang in lang_map:
           for extra_idx in range(1, num_extra + 1):
               idx_to_add = asr_model.joint.num_classes_with_blank - 1 + extra_idx
               if idx_to_add not in lang_map[lang]["local_to_global"]:
                   lang_map[lang]["local_to_global"].append(idx_to_add)
                   lang_map[lang]["global_to_local"][str(idx_to_add)] = len(lang_map[lang]["local_to_global"]) - 1


   hidden_dim = asr_model.cfg.joint.jointnet.joint_hidden if hasattr(asr_model, "cfg") and hasattr(asr_model.cfg, "joint") else 1024
  
   if hasattr(asr_model, "joint") and hasattr(asr_model.joint, "joint_net") and len(asr_model.joint.joint_net) > 2:
       asr_model.joint.joint_net[2] = LangJointPostNet(hidden_dim, lang_map, global_vocab_size).to(next(asr_model.parameters()).device)


   # Patch CTC Decoder to use LangCTCDecoder
   if hasattr(asr_model, "ctc_decoder"):
       ctc_hidden_dim = asr_model.cfg.encoder.d_model
       asr_model.ctc_decoder = LangCTCDecoder(ctc_hidden_dim, ctc_lang_map, ctc_global_vocab_size).to(next(asr_model.parameters()).device)
  
   print("Model patching complete!")





