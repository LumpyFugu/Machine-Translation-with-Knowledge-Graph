import torch

import fairseq
from torch import nn
from torch.nn import functional as F

from fairseq.models import FairseqEncoder, FairseqDecoder
from fairseq.models.fairseq_model import BaseFairseqModel
#from fairseq.models.transformer import TransformerModel, EncoderOut, DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
import copy
import random

import numpy as np

from collections import namedtuple

from .utils import *


# Newly added below
from fairseq.models.transformer import (
    TransformerModel,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)
'''
# EncoderOut location varies by version: new version uses fairseq.models.fairseq_encoder
try:
    from fairseq.models.fairseq_encoder import EncoderOut  # fairseq >= 0.10/0.12
except Exception:
    from fairseq.models.transformer import EncoderOut       # only old fairseq exposes it here
# End of newly added section
'''

'''
This file defines an entity-enhanced encoder-decoder model for integrated NER + translation tasks.
It is based on Fairseq Transformer, extends entity-label handling, and supports multiple train/inference modes.
'''

ModelOut = namedtuple('ModelOut', [
    'decoder_out',  # the (decoder out, extra), same as original,
    'encoder_ne_logit',
    'decoder_ne_logit',
    'entity_out',
    'entity_label',
    'result_entity_id',
    'encoder_ne'
])

NE_PENALTY=1e8

all_types = ['ORG', 'EVENT', 'PRODUCT', 'FAC', 'PERCENT', 'WORK_OF_ART', 'ORDINAL', 'LOC',
                 'LANGUAGE', 'LAW', 'PERSON', 'TIME', 'CARDINAL', 'GPE', 'QUANTITY', 'DATE', 'NORP', 'MONEY']
class EntityEncoderDecoderModel(BaseFairseqModel):

    def __init__(self, args, ne_dict, encoder, decoder, tgt_ne_start_id, bert_emb_id_dict, bert_emb_value, entity_mapping):
        super().__init__()

        self.args = args
        self.ne_dict = ne_dict
        self.encoder = encoder
        self.decoder = decoder
        self.mode = args.mode
        self.tgt_ne_start_id = tgt_ne_start_id
        self.bert_emb_id_dict = bert_emb_id_dict  # tuple ot token -> bert id
        self.bert_emb_value = torch.nn.Parameter(bert_emb_value, requires_grad=False)
        self.entity_mapping = entity_mapping  # tupe of src id -> set (list of tgt id)

        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)
        assert 0 <= self.mode <= 5
        assert 0 <= self.args.bert_lookup_layer <= self.args.decoder_layers # 0 is the input
        assert 1 <= self.args.src_ne_layer <= self.args.encoder_layers

        self.encoder_ne_process_mask = {}
        if self.mode == 1 or self.mode == 2 or self.mode == 4 or self.mode == 5:
            self.src_ne_fc1 = nn.Linear(args.encoder_embed_dim, args.src_ne_project, bias=True)
            self.src_ne_fc2 = nn.Linear(args.src_ne_project, len(ne_dict), bias=True)

        if self.mode == 1 or self.mode == 4 or self.mode == 5:
            self.tgt_ne_fc1 = nn.Linear(args.decoder_embed_dim, args.tgt_ne_project, bias=True)
            self.tgt_ne_fc2 = nn.Linear(args.tgt_ne_project, len(ne_dict), bias=True)
        elif self.mode == 2:
            self.bert_sample_count = min(args.bert_sample_count, len(bert_emb_id_dict))

            self.bert_dim = bert_emb_value.shape[1]
            self.tgt_bert_ne_fc1 = nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim, bias=True)
            self.tgt_bert_ne_fc2 = nn.Linear(args.decoder_embed_dim, self.bert_dim, bias=True)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--src-ne-project', type=int)
        parser.add_argument('--src-ne-project-dropout', type=float, default=0.0)
        parser.add_argument('--tgt-ne-project', type=int)
        parser.add_argument('--concat-ne-emb', action='store_true')
        parser.add_argument('--bert-lookup-layer', type=int) # 0 is the input, and n is the last layer
        parser.add_argument('--bert-lookup-dropout', type=float, default=0.0) #
        parser.add_argument('--src-ne-layer', type=int) # 1 is the first layer output, and n is the last layer
        # newly added KG
        parser.add_argument(
            "--kg-embed-path",
            type=str,
            default=None,
            help="path to KG embedding tensor (.pt) aligned with src dict",
        )
        parser.add_argument(
            "--kg-embed-dim",
            type=int,
            default=300,  # depends on your downloaded wikipedia2vec dimension
            help="dimension of KG embedding (before projection)",
        )


    def to_bert_emb_space(self, x):
        # Is this too simple? Do we need more layers? e.g. attention?
        # Do we need enable dropout here?
        x = F.relu(self.tgt_bert_ne_fc1(x))
        x = F.dropout(x, self.args.bert_lookup_dropout, self.training)
        return self.tgt_bert_ne_fc2(x)

    #@profile
    def encoder_ne_process(self, encoder_out, ne_type, need_logit):
        """
        ne_type:0 => O, B-XX, I-XX
        ne_type:1 => O, XX-0, XX-1

        Compatible with:
        - encoder_out is a dict (new fairseq encoder output)
        - encoder_out is an EncoderOut object (old/original code)
        - encoder_out["encoder_out"] can be Tensor or list[Tensor]
        - encoder_states can be None / Tensor / list[Tensor]
        Guarantees encoder_out passed to decoder is always 3D T x B x C Tensor.
        """
        import torch
        import torch.nn.functional as F
        #from fairseq.models.transformer import EncoderOut

        assert ne_type in (0, 1)

        # === 0. Record original type for repacking ===
        is_dict = isinstance(encoder_out, dict)
        orig_encoder_out = encoder_out

        # === 1. Unpack encoder_out / padding_mask / encoder_embedding / encoder_states ===
        if is_dict:
            raw_enc_out = encoder_out["encoder_out"]                    # can be Tensor or list[Tensor]
            padding_mask = encoder_out.get("encoder_padding_mask", None)
            encoder_embedding = encoder_out.get("encoder_embedding", None)
            enc_states = encoder_out.get("encoder_states", None)
            src_tokens = encoder_out.get("src_tokens", None)
            src_lengths = encoder_out.get("src_lengths", None)
        else:
            raw_enc_out = encoder_out.encoder_out                       # can be Tensor or list[Tensor]
            padding_mask = getattr(encoder_out, "encoder_padding_mask", None)
            encoder_embedding = getattr(encoder_out, "encoder_embedding", None)
            enc_states = getattr(encoder_out, "encoder_states", None)
            src_tokens = getattr(encoder_out, "src_tokens", None)
            src_lengths = getattr(encoder_out, "src_lengths", None)

        def _is_nested_tensor(x):
            checker = getattr(torch, "is_nested", None)
            if checker is not None:
                try:
                    return checker(x)
                except Exception:
                    pass
            return hasattr(x, "to_padded_tensor")

        def _densify_nested(x):
            """
            Try to turn a nested tensor into a padded dense tensor without relying
            on backend-specific to_padded_tensor kernels.
            """
            if not _is_nested_tensor(x):
                return x

            # First attempt the direct API (may be available in some builds)
            try:
                return x.to_padded_tensor(0.0)
            except Exception:
                pass

            # Fallback: materialize into a list of tensors and pad manually.
            seq_list = None
            to_list_fn = getattr(x, "to_tensor_list", None)
            if callable(to_list_fn):
                try:
                    seq_list = to_list_fn()
                except Exception:
                    seq_list = None

            if seq_list is None:
                try:
                    seq_list = list(x.unbind())
                except Exception:
                    try:
                        seq_list = list(x)
                    except Exception as e:
                        raise RuntimeError(f"Failed to densify nested tensor: {e}")

            seq_list = [s if isinstance(s, torch.Tensor) else torch.as_tensor(s) for s in seq_list]
            if len(seq_list) == 0:
                return torch.empty(0)

            seq_proc = []
            for s in seq_list:
                if s.dim() == 1:
                    seq_proc.append(s.unsqueeze(1))
                else:
                    seq_proc.append(s)

            padded = torch.nn.utils.rnn.pad_sequence(seq_proc, batch_first=True)
            return padded

        def _to_dense_3d(x):
            """
            Some encoder implementations return nested tensors; convert them to padded dense tensors.
            Keep the shape as close as possible to T x B x C afterward.
            """
            x = _densify_nested(x)
            if x.dim() != 3:
                return x

            # Try to infer whether the first dim is batch; if so, transpose to T x B x C.
            def _batch_hint_from(v):
                if isinstance(v, list) and len(v) > 0:
                    v = v[0]
                if hasattr(v, "size"):
                    try:
                        return v.size(0)
                    except Exception:
                        return None
                if isinstance(v, torch.Tensor):
                    return v.shape[0]
                return None

            batch_hint = None
            if padding_mask is not None:
                batch_hint = _batch_hint_from(padding_mask)
            if batch_hint is None and src_tokens is not None:
                batch_hint = _batch_hint_from(src_tokens)
            if batch_hint is None and src_lengths is not None:
                batch_hint = _batch_hint_from(src_lengths)

            if batch_hint is not None and x.size(0) == batch_hint and x.size(1) != batch_hint:
                x = x.transpose(0, 1)
            return x

        # === 2. Normalize raw_enc_out shape => 3D (T x B x C) ===
        # Some implementations return list[Tensor]; use last layer
        if isinstance(raw_enc_out, list):
            if len(raw_enc_out) == 0:
                raise RuntimeError("encoder_out['encoder_out'] is an empty list.")
            raw_enc_out = raw_enc_out[-1]

        if not isinstance(raw_enc_out, torch.Tensor):
            raise TypeError(f"encoder_out['encoder_out'] must be Tensor or list[Tensor], got {type(raw_enc_out)}")

        raw_enc_out = _to_dense_3d(raw_enc_out)

        # If 2D, add time dim: B x C -> 1 x B x C
        if raw_enc_out.dim() == 2:
            raw_enc_out = raw_enc_out.unsqueeze(0)
        elif raw_enc_out.dim() != 3:
            raise RuntimeError(f"encoder_out tensor must be 3D, got shape {raw_enc_out.size()}")

        enc_out = raw_enc_out  # T x B x C

        # === 3. Normalize encoder_embedding shape (optional field) ===
        if isinstance(encoder_embedding, list):
            if len(encoder_embedding) > 0:
                encoder_embedding = encoder_embedding[-1]
            else:
                encoder_embedding = None

        if encoder_embedding is not None:
            if not isinstance(encoder_embedding, torch.Tensor):
                raise TypeError(f"encoder_embedding must be Tensor or list[Tensor], got {type(encoder_embedding)}")

            # Common shapes: B x T x C or B x C
            if encoder_embedding.dim() == 2:
                # B x C -> B x 1 x C
                encoder_embedding = encoder_embedding.unsqueeze(1)
            elif _is_nested_tensor(encoder_embedding):
                encoder_embedding = _densify_nested(encoder_embedding)
            # Otherwise (3D), keep as-is
            # Do not force T x B x C here; decoder rarely uses embedding

        # === 4. Normalize encoder_states => list[3D Tensor] ===
        if enc_states is None:
            enc_states = [enc_out]
        elif isinstance(enc_states, torch.Tensor):
            enc_states = _to_dense_3d(enc_states)
            # can be T x B x C or B x C
            if enc_states.dim() == 2:
                enc_states = enc_states.unsqueeze(0)
            elif enc_states.dim() != 3:
                raise RuntimeError(f"encoder_states tensor must be 3D, got shape {enc_states.size()}")
            enc_states = [enc_states]
        elif isinstance(enc_states, list):
            norm_states = []
            for s in enc_states:
                if not isinstance(s, torch.Tensor):
                    continue
                s = _to_dense_3d(s)
                if s.dim() == 2:
                    s = s.unsqueeze(0)
                elif s.dim() != 3:
                    raise RuntimeError(f"one encoder_state must be 3D, got shape {s.size()}")
                norm_states.append(s)
            if len(norm_states) == 0:
                norm_states = [enc_out]
            enc_states = norm_states
        else:
            raise TypeError(f"encoder_states must be None / Tensor / list[Tensor], got {type(enc_states)}")

        # === 5. Select layer used for NE prediction ===
        layer_idx = getattr(self.args, "src_ne_layer", 1) - 1  # src_ne_layer is 1-based
        if layer_idx < 0:
            layer_idx = 0
        if layer_idx >= len(enc_states):
            entity_input = enc_states[-1]   # if out of range, use last layer
        else:
            entity_input = enc_states[layer_idx]  # T x B x C

        # === 6. Get NE embedding via linear layer + dropout ===
        encoder_ne_emb = F.dropout(
            F.relu(self.src_ne_fc1(entity_input)),
            self.args.src_ne_project_dropout,
            self.training,
        )  # T x B x C' (C' may equal C)

        # === 7. If logits are needed (training) ===
        if need_logit:
            encoder_ne_logit = self.src_ne_fc2(encoder_ne_emb)      # T x B x |NE|
            encoder_ne_logit = encoder_ne_logit.transpose(0, 1)     # B x T x |NE|

            C = encoder_ne_logit.shape[2]
            max_ne_id = self.args.max_ne_id

            if (C, max_ne_id) not in self.encoder_ne_process_mask:
                logit_mask = [False] * C
                for i in range(C):
                    if i < self.ne_dict.nspecial + 1:  # special + 'O'
                        if i == self.ne_dict.bos_index or i == self.ne_dict.unk_index:
                            logit_mask[i] = False
                        else:
                            logit_mask[i] = True
                    else:
                        k = (i - (self.ne_dict.nspecial + 1)) % (max_ne_id + 2)
                        if ne_type == 0:
                            logit_mask[i] = k < 2
                        else:
                            logit_mask[i] = k >= 2
                self.encoder_ne_process_mask[(C, max_ne_id)] = logit_mask
            else:
                logit_mask = self.encoder_ne_process_mask[(C, max_ne_id)]

            logit_mask = torch.as_tensor(logit_mask, device=encoder_ne_logit.device)
            encoder_ne_logit[:, :, ~logit_mask] = float("-inf")
        else:
            encoder_ne_logit = None

        # === 8. Fuse NE embedding with original encoder_out, keep 3D T x B x C ===
        if self.args.concat_ne_emb:
            # enc_out: T x B x C1, encoder_ne_emb: T x B x C2
            combined_encoder_out = torch.cat((enc_out, encoder_ne_emb), dim=-1)
        else:
            # requires identical shapes
            if enc_out.shape != encoder_ne_emb.shape:
                raise RuntimeError(
                    f"enc_out shape {enc_out.shape} and encoder_ne_emb shape {encoder_ne_emb.shape} "
                    f"must match for addition; consider using concat_ne_emb."
                )
            combined_encoder_out = enc_out + encoder_ne_emb

        if combined_encoder_out.dim() == 2:
            combined_encoder_out = combined_encoder_out.unsqueeze(0)
        elif combined_encoder_out.dim() != 3:
            raise RuntimeError(f"combined_encoder_out must be 3D, got shape {combined_encoder_out.size()}")

        # === 9. Return using the same container type as original encoder_out ===
        if is_dict:
            # shallow copy dict to avoid mutating original object
            encoder_out_with_emb = dict(orig_encoder_out)

            # Was original encoder_out field list[Tensor] or Tensor?
            old_enc_out = orig_encoder_out["encoder_out"]
            if isinstance(old_enc_out, list):
                # Decoder may index encoder_out[-1], so return a list here
                encoder_out_with_emb["encoder_out"] = [combined_encoder_out]
            else:
                # If original was Tensor, return Tensor directly
                encoder_out_with_emb["encoder_out"] = combined_encoder_out

            encoder_out_with_emb["encoder_padding_mask"] = padding_mask
            encoder_out_with_emb["encoder_embedding"] = encoder_embedding
            encoder_out_with_emb["encoder_states"] = enc_states
            encoder_out_with_emb["src_tokens"] = src_tokens
            encoder_out_with_emb["src_lengths"] = src_lengths
        else:
            # In old fairseq EncoderOut is usually namedtuple; use _replace
            if hasattr(orig_encoder_out, "_replace"):
                old_enc_out = getattr(orig_encoder_out, "encoder_out", None)
                new_enc_out = combined_encoder_out
                if isinstance(old_enc_out, list):
                    new_enc_out = [combined_encoder_out]

                encoder_out_with_emb = orig_encoder_out._replace(
                    encoder_out=new_enc_out,
                    encoder_padding_mask=padding_mask,
                    encoder_embedding=encoder_embedding,
                    encoder_states=enc_states,
                    src_tokens=src_tokens,
                    src_lengths=src_lengths,
                )
            else:
                # For safety, keep unchanged if not a namedtuple
                encoder_out_with_emb = orig_encoder_out


        return encoder_out_with_emb, encoder_ne_logit

    
    #@profile
    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
                - the encoder ne logit
                - the decoder ne logit
        """
# === OOB hard check (print via exception message) ===
        num_src_emb = self.encoder.embed_tokens.num_embeddings
        mx_src = int(src_tokens.max())
        mn_src = int(src_tokens.min())
        if mx_src >= num_src_emb:
            bad = (src_tokens >= num_src_emb).nonzero(as_tuple=False)
            # Print first few positions (row/col) and sample tokens for debugging
            print(f"[SRC OOB] num_emb={num_src_emb}, bad_count={bad.size(0)}", flush=True)
            print(f"[SRC OOB] first_bad_positions={bad[:10].tolist()}", flush=True)
            # If possible, also print a short token snippet from that sample
            bi, ti = bad[0].tolist()
            print(f"[SRC OOB] sample[{bi}] head tokens={src_tokens[bi][:50].tolist()}", flush=True)
            raise RuntimeError(
                f"[SRC OOB] max_id={mx_src} >= num_embeddings={num_src_emb} (min_id={mn_src})"
            )   
# ==========================================

        src_ne = kwargs.get("src_ne", None)
        # decoder does not need src_ne; avoid forwarding unexpected args
        kwargs.pop("src_ne", None)
        kg_embedding = None
        if getattr(self, "kg_table", None) is not None:
            # kg_table: |V| x kg_dim
            # src_tokens: B x T
            B, T = src_tokens.size()
            kg_embedding = self.kg_table.index_select(0, src_tokens.view(-1)).view(B, T, -1)

            # align device/dtype (important for mixed precision)
            kg_embedding = kg_embedding.to(device=src_tokens.device)
            kg_embedding = kg_embedding.to(dtype=self.encoder.embed_tokens.weight.dtype)
        
        if kg_embedding is not None and not hasattr(self, "_printed_kg"):
            self._printed_kg = True
            self._printed_kg_tok = True
            pad_id = self.encoder.embed_tokens.padding_idx
            print("[DEBUG] pad_id:", pad_id, "kg_pad_norm:", float(self.kg_table[pad_id].norm()), flush=True)
            print("[DEBUG] kg_table:", tuple(self.kg_table.shape), flush=True)
            print("[DEBUG] kg_embedding:", tuple(kg_embedding.shape),
                kg_embedding.dtype, kg_embedding.device, flush=True)
            print("[DEBUG] src_tokens range:", int(src_tokens.min()), int(src_tokens.max()), flush=True)

        entity_mask = None
        if src_ne is not None:
            # infer O id robustly
            if hasattr(self.ne_dict, "index") and "O" in getattr(self.ne_dict, "symbols", []):
                o_id = self.ne_dict.index("O")
            else:
                o_id = self.ne_dict.nspecial  # common case: first token after specials is O

            entity_mask = (src_ne != o_id) & (src_tokens != self.encoder.embed_tokens.padding_idx)
            # entity_mask: B x T (bool)

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=(self.mode != 0),
        )

        if src_ne is not None and not hasattr(self, "_printed_src_ne"):
            self._printed_src_ne = True
            print("[DEBUG] src_ne:", src_ne.shape, int(src_ne.min()), int(src_ne.max()), flush=True)


        if self.mode == 0 or self.mode==3:
            return ModelOut(
                decoder_out=self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs),
                encoder_ne_logit=None,
                decoder_ne_logit=None,
                entity_out=None,
                entity_label=None,
                result_entity_id=None,
                encoder_ne=None
            )
        elif self.mode == 1 or self.mode == 4 or self.mode == 5:
            encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=True)

            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out_with_emb,
                features_only=True,
                **kwargs
            )

            # For new fairseq: prefer output_projection, fallback to embedding weights
            if getattr(self.decoder, "output_projection", None) is not None:
                # output_projection: Linear(embed_dim, vocab_size)
                decoder_out = self.decoder.output_projection(decoder_out_feature)
            else:
                # Secondary fallback: use embedding weights for tied softmax
                # embed_tokens.weight: (vocab_size, embed_dim)
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_tokens.weight)


            assert (decoder_out_feature == extra['inner_states'][-1].transpose(0, 1)).all()

            ne_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1)

            decoder_ne_emb = F.relu(self.tgt_ne_fc1(ne_input_feature))
            decoder_ne_logit = self.tgt_ne_fc2(decoder_ne_emb)
            return ModelOut(
                decoder_out=(decoder_out, extra),
                encoder_ne_logit=encoder_ne_logit,
                decoder_ne_logit=decoder_ne_logit,
                entity_out=None,
                entity_label=None,
                result_entity_id=None,
                encoder_ne=None
            )
        elif self.mode == 2:
            encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=True)

            # decoder_out_feature: B x T x C
            '''
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, features_only=True, **kwargs)
            
            # Use decoder's projection head (handles tied or separate embeddings)
            decoder_out = self.decoder.output_layer(decoder_out_feature)
            '''

            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, features_only=True, **kwargs)
            if self.decoder.share_input_output_embed:
                out_weight = self.decoder.embed_tokens.weight
            else:
                # newer fairseq may not have embed_out; use output_projection.weight
                if getattr(self.decoder, "output_projection", None) is not None:
                    out_weight = self.decoder.output_projection.weight
                else:
                    out_weight = self.decoder.embed_tokens.weight
            decoder_out = F.linear(decoder_out_feature, out_weight)
            
            bert_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1)
            assert (decoder_out_feature == extra['inner_states'][-1].transpose(0,1)).all()
            if self.training:
                """
                During the training, there is no need to find entity from src as we have the ground truch in tgt.
                So, extract from the tgt (via prev_output_tokens) and make negative sampling.
                """
                ne_source = kwargs['ne_source']
                tgt_ne_pos = kwargs['tgt_ne_pos']
                origin_tgt = kwargs['origin_tgt']
                target = kwargs['target']
                B, T, C = decoder_out_feature.shape
                entity_position = (target > self.tgt_ne_start_id)  # B * T. Note we don't check 'O' here. We don't expect that here
                #entity_bert_id = torch.zeros((B, T), dtype=torch.long, device=decoder_out_feature.device)
                entity_bert_id = [[-1] * T for _ in range(B)]
                all_entitiy_ids = set()

                # Some of them are using entity in the same batch. As the batch is random generated, it ok to use them as false label
                # 1. entities from target (ground truth)
                # TODO: optimize
                for i in range(B):
                    origin_tgt_i = origin_tgt[i].tolist()
                    for j in range(T):
                        if entity_position[i, j]:
                            origin_tgt_toks = tuple(origin_tgt_i[tgt_ne_pos[i][j]])
                            if origin_tgt_toks in self.bert_emb_id_dict:
                                ent_id_cur = self.bert_emb_id_dict[origin_tgt_toks]
                                entity_bert_id[i][j] = ent_id_cur
                                all_entitiy_ids.add(ent_id_cur)
                            else:
                                # This means we don't preprocess it in BERT. so cannot train on it.
                                # Just skip for now.
                                entity_position[i, j] = False
                                pass
                entity_bert_id = torch.as_tensor(entity_bert_id, device=decoder_out_feature.device)
                # 2. entityies from src mapping. They are harder than random sampling
                for i in range(B):
                    src_entities, src_entity_types = extract_ne_from_text(src_tokens[i], ne_source[i], self.ne_dict, need_type=True)
                    for entity, entity_type in zip(src_entities, src_entity_types):
                        if (entity_type, entity) in self.entity_mapping:
                            mapped_entities = self.entity_mapping[(entity_type, entity)]
                            for tgt_entity in mapped_entities:
                                if tgt_entity in self.bert_emb_id_dict:
                                    all_entitiy_ids.add(self.bert_emb_id_dict[tgt_entity])


                # 3. Random sample from the bert dict, which in theory should be unlimited large
                if len(all_entitiy_ids) < self.bert_sample_count:
                    # Not exact match, bust much faster
                    all_ids = np.random.choice(len(self.bert_emb_id_dict), self.bert_sample_count)
                    for v in all_ids:
                        all_entitiy_ids.add(v)

                all_entitiy_ids = list(all_entitiy_ids)
                ent_id_to_sample_id = { v:i for i, v in enumerate(all_entitiy_ids)}

                """
                bert_emt_matrix = torch.zeros((self.bert_sample_count, self.bert_dim), device=target.device)

                for i, vec_id in enumerate(all_entitiy_ids):
                    # print(i, vec_id, bert_emt_matrix.shape, self.bert_emb_value.shape)
                    bert_emt_matrix[i, :] = self.bert_emb_value[vec_id]
                """
                bert_emt_matrix = self.bert_emb_value[all_entitiy_ids]

                entity_position = entity_position.view(-1)

                entity_embeddings = self.to_bert_emb_space(bert_input_feature.reshape((B*T, C))[entity_position])

                entity_out = F.linear(entity_embeddings, bert_emt_matrix)  # ( (entity count) B*T, C)
                
                entity_label = entity_bert_id.view(-1)[entity_position]   # (( (entity count)B*T)) # this is the orginal id, need to map to sampled id.
                entity_label_in_sample_id = torch.zeros_like(entity_label, dtype=torch.long ,device=entity_label.device)

                for i in range(len(entity_label)):
                    entity_label_in_sample_id[i] = ent_id_to_sample_id[entity_label[i].item()]

                return ModelOut(
                    decoder_out=(decoder_out, extra),
                    encoder_ne_logit=encoder_ne_logit,
                    decoder_ne_logit=None,
                    entity_out=entity_out,
                    entity_label=entity_label_in_sample_id,
                    result_entity_id=None,
                    encoder_ne=None
                )
            else:
                """
                During test time, we need to extract from the src ne prediction, then find the candidate. 
                This is for speed and accuracy.
                Note: this could be under test, under dev. In dev, the label is still required to compute the loss

                Actually, target will be always available. When inference, it will use 'forward_encoder' and 'forward_decoder' sepreately.
                Not this method. TODO: clean me up.
                """

                encoder_ne_pred = encoder_ne_logit.argmax(axis=-1)  # B x T x C => B x T

                # When dev, that target is avabiable to compute the loss
                target = kwargs.get('target', None)
                origin_tgt = kwargs.get('origin_tgt', None)
                tgt_ne_pos = kwargs.get('tgt_ne_pos', None)

                B, T, C = decoder_out_feature.shape
                entities_in_batch = [[] for _ in range(B)]
                all_entitiy_ids = set()

                if target is not None:
                    target_entity_position = (target > self.tgt_ne_start_id).cpu().numpy()  # B * T. Note we don't check 'O' here. We don't expect that here
                    #target_entity_bert_id = torch.zeros((B, T), dtype=torch.long, device=decoder_out_feature.device)
                    target_entity_bert_id = [ [-1]*T for _ in range(B)]
                    # When target is avaible, we need to add them in the entity ids. So that it can appear in the dictionary to compute loss.
                    for i in range(B):
                        for j in range(T):
                            if target_entity_position[i, j]:
                                origin_tgt_toks = tuple(origin_tgt[i].tolist()[tgt_ne_pos[i][j]])
                                if origin_tgt_toks in self.bert_emb_id_dict:
                                    ent_id_cur = self.bert_emb_id_dict[origin_tgt_toks]
                                    target_entity_bert_id[i][j] = ent_id_cur
                                    all_entitiy_ids.add(ent_id_cur)
                                    entities_in_batch[i].append(ent_id_cur)
                                else:
                                    # This means we don't preprocess it in BERT. so cannot train on it.
                                    # Just ignore here, as we cannot compute loss for unknown entity
                                    # TODO: find a better way to handle it
                                    target_entity_position[i, j] = False
                                    target_entity_bert_id[i][j] = -1
                    target_entity_bert_id = torch.as_tensor(target_entity_bert_id, device=decoder_out_feature.device)

                for i in range(B):
                    src_entities, src_entity_types = extract_ne_from_text(src_tokens[i], encoder_ne_pred[i], self.ne_dict, need_type=True)
                    for entity, entity_type in zip(src_entities, src_entity_types):
                        if (entity_type, entity) in self.entity_mapping:
                            mapped_entities = self.entity_mapping[(entity_type, entity)]
                            for tgt_entity in mapped_entities:
                                if tgt_entity in self.bert_emb_id_dict:
                                    entities_in_batch[i].append(self.bert_emb_id_dict[tgt_entity])
                    
                    all_entitiy_ids = all_entitiy_ids.union(entities_in_batch[i])

                all_entitiy_ids = list(all_entitiy_ids)

                # We need to combine all the entity in a batch, look up then mask by -inf
                # TODO:and we need to allow copy

                if len(all_entitiy_ids) > 0:
                    ent_id_to_sample_id = { v:i for i, v in enumerate(all_entitiy_ids)}
                    
                    """
                    bert_ent_matrix = torch.zeros((len(all_entitiy_ids), self.bert_dim), device=prev_output_tokens.device)
                    for i, vec_id in enumerate(all_entitiy_ids):
                        bert_ent_matrix[i, :] = self.bert_emb_value[vec_id]
                    """
                    bert_ent_matrix = self.bert_emb_value[all_entitiy_ids]

                    entity_position = (decoder_out.argmax(-1) > self.tgt_ne_start_id).cpu().numpy()  # B * T, Note we don't check 'O' here. We don't expect that here

                    entity_embeddings =  self.to_bert_emb_space(bert_input_feature)
                    entity_out = F.linear(entity_embeddings, bert_ent_matrix)  # B * T * C

                    # mask token not in that sentence
                    """
                    # No need to mask now. Only dev need to use it.
                    for i in range(B):
                        masked_entity_id = [True] * len(all_entitiy_ids)

                        for vec_id in entities_in_batch[i]:
                            masked_entity_id[ent_id_to_sample_id[vec_id]] = False

                        entity_out[i, :, masked_entity_id] = float('-inf')
                    """
                    entity_out_max = entity_out.argmax(-1)  # (B * T)

                    # map back to original entity id
                    #result_entity_id = torch.zeros_like(entity_out_max, device=entity_out_max.device)
                    
                    result_entity_id = [[-1]*T for _ in range(B)]
                    for i in range(B):
                        for j in range(T):
                            if entity_position[i, j]:
                                result_entity_id[i][j] = all_entitiy_ids[entity_out_max[i, j]]

                    result_entity_id = torch.as_tensor(result_entity_id, device=entity_out_max.device)

                    if target is not None:
                        target_entity_position = torch.as_tensor(target_entity_position)
                        entity_out = entity_out[target_entity_position]
                        entity_label = target_entity_bert_id[target_entity_position].view(-1)

                        # Map to sample dict id
                        entity_label_in_sample_id = torch.zeros_like(entity_label, dtype=torch.long ,device=entity_label.device)
                        for i in range(len(entity_label)):
                            entity_label_in_sample_id[i] = ent_id_to_sample_id[entity_label[i].item()]
                    else:
                        # Not required as we don't compute loss
                        entity_out, entity_label_in_sample_id = None, None
                else:
                    # There is no entity extracted from src. :(
                    # TODO: shall we do anything to rescue it? Or it just that there is no entity in this batch?
                    result_entity_id = torch.ones((B, T), dtype=torch.long, device=decoder_out_feature.device) * -1
                    if target is not None:
                        entity_out = torch.zeros((0), dtype=torch.long, device=decoder_out_feature.device)
                        entity_label_in_sample_id = torch.zeros((0), dtype=torch.long, device=decoder_out_feature.device)
                    else:
                        entity_out, entity_label_in_sample_id = None, None

                return ModelOut(
                    decoder_out=(decoder_out, extra),
                    encoder_ne_logit=encoder_ne_logit,
                    decoder_ne_logit=None,
                    entity_out=entity_out,
                    entity_label=entity_label_in_sample_id,
                    result_entity_id=result_entity_id,
                    encoder_ne=None
                )
        else:
            raise Exception(f'Bad mode {self.mode}')

    def forward_decoder(self, prev_output_tokens, encoder_out, **kwargs):
        if self.mode == 0 or self.mode == 3:
            return self.decoder(prev_output_tokens, encoder_out, **kwargs)
        elif self.mode == 1 or self.mode == 4 or self.mode == 5:
            # encoder_out_with_emb, _ = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=False)
            # return self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, **kwargs)

            encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=True)

            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, features_only=True, **kwargs)
            # Use decoder's projection head (handles tied or separate embeddings)
            decoder_out = self.decoder.output_layer(decoder_out_feature)

            assert (decoder_out_feature == extra['inner_states'][-1].transpose(0, 1)).all()

            ne_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1)

            decoder_ne_emb = F.relu(self.tgt_ne_fc1(ne_input_feature))
            decoder_ne_logit = self.tgt_ne_fc2(decoder_ne_emb)
            return ModelOut(
                decoder_out=(decoder_out, extra),
                encoder_ne_logit=encoder_ne_logit,
                decoder_ne_logit=decoder_ne_logit,
                entity_out=None,
                entity_label=None,
                result_entity_id=None,
                encoder_ne=None
            )

        else:
            src_tokens = kwargs['src_tokens']
            encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=True)
            
            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, features_only=True, **kwargs)
            # Use decoder's projection head (handles tied or separate embeddings)
            decoder_out = self.decoder.output_layer(decoder_out_feature)
            encoder_ne_pred = encoder_ne_logit.argmax(axis=-1)  # B x T x C => B x T
            bert_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1) 

            #entity_position = (decoder_out.argmax(-1) > self.tgt_ne_start_id).cpu().numpy()  # B * T, Note we don't check 'O' here. We don't expect that here
            entity_pred_type = torch.topk(decoder_out, k=5, dim=-1)[1].cpu()
            entity_position = (entity_pred_type > self.tgt_ne_start_id).any(dim=-1).numpy() # B * T, Note we don't check 'O' here. We don't expect that here
            B, T, C = decoder_out_feature.shape
            entities_in_batch = [[] for _ in range(B)]
            
            all_typed_entities = set()
            all_entitiy_ids = set()

            src_entities = [[] for _ in range(B)]
            src_entity_types = [[] for _ in range(B)]
            for i in range(B):
                src_entities[i], src_entity_types[i] = extract_ne_from_text(src_tokens[i], encoder_ne_pred[i], self.ne_dict, need_type=True)
                for entity, entity_type in zip(src_entities[i], src_entity_types[i]):
                    if (entity_type, entity) in self.entity_mapping:
                        mapped_entities = self.entity_mapping[(entity_type, entity)]
                        for tgt_entity in mapped_entities:
                            if tgt_entity in self.bert_emb_id_dict:
                                entities_in_batch[i].append((entity_type, self.bert_emb_id_dict[tgt_entity]))
                
                # When there is an entity in the hypo, but can't find any mapping from src.
                # We iterate all src sub str, and hope to find sth. It will make the inference even slower...
                if not entities_in_batch[i] and entity_position[i].any():
                    src_token_list = src_tokens[i].tolist()
                    for s in range(0, len(src_token_list)):
                        # assume the src is pad left
                        if src_token_list[s] == self.ne_dict.pad():
                            continue

                        if src_token_list[s] == self.ne_dict.eos():
                            break

                        for e in range(s + 1, len(src_token_list)):
                            entity = tuple(src_token_list[s:e])
                            for t in all_types:
                                if (t, entity) in self.entity_mapping:
                                    mapped_entities = self.entity_mapping[(t, entity)]
                                    for tgt_entity in mapped_entities:
                                        if tgt_entity in self.bert_emb_id_dict:
                                            entities_in_batch[i].append((t, self.bert_emb_id_dict[tgt_entity]))


                all_typed_entities = all_typed_entities.union(entities_in_batch[i])

            all_typed_entities = list(all_typed_entities)
            all_entitiy_ids = [x[1] for x in all_typed_entities]

            # We need to combine all the entity in a batch, look up then mask by -inf
            # TODO:and we need to allow copy

            if len(all_entitiy_ids) > 0:
                ent_id_to_sample_id = { v:i for i, v in enumerate(all_entitiy_ids)}
                
                """
                bert_ent_matrix = torch.zeros((len(all_entitiy_ids), self.bert_dim), device=prev_output_tokens.device)
                for i, vec_id in enumerate(all_entitiy_ids):
                    bert_ent_matrix[i, :] = self.bert_emb_value[vec_id]
                """
                bert_ent_matrix = self.bert_emb_value[all_entitiy_ids]

                
                entity_embeddings =  self.to_bert_emb_space(bert_input_feature)
                entity_out = F.linear(entity_embeddings, bert_ent_matrix)  # B * T * C

                # mask token, if not in that sent, or wrong type
                for i in range(B):
                    for j in range(T):
                        if not entity_position[i, j]:
                            entity_out[i, j, :] = float('-inf')
                        else:
                            pred_target_types = entity_pred_type[i][j][ entity_pred_type[i][j] > self.tgt_ne_start_id ].tolist()
                            pred_target_type_text = set([self.ne_dict[x - self.tgt_ne_start_id].split('-')[0] for x in pred_target_types])
                            
                            assert len(pred_target_types) > 0

                            masked_entity_id = [True] * len(all_entitiy_ids)

                            for (ent_type, vec_id) in entities_in_batch[i]:
                                if ent_type in pred_target_type_text: #vec type is correct:
                                    masked_entity_id[ent_id_to_sample_id[vec_id]] = False

                            entity_out[i, j, masked_entity_id] = float('-inf')

                entity_out_max = entity_out.argmax(-1)  # (B * T)

                # map back to original entity id
                #result_entity_id = torch.zeros_like(entity_out_max, device=entity_out_max.device)
                result_entity_id = [[-1]*T for _ in range(B)]
                for i in range(B):
                    for j in range(T):
                        #if entity_position[i, j]:
                        ent_pred = entity_out_max[i, j].item()
                        if entity_out[i, j, ent_pred] != float('-inf'):
                            result_entity_id[i][j] = all_entitiy_ids[ent_pred]
                        else:
                            result_entity_id[i][j] = -1

                            if entity_position[i, j]:
                                # Here is predicted as an entity, but cannot find anything to fill it.
                                # So reduce the logit so that it will output normal token
                                decoder_out[i][j][self.tgt_ne_start_id:] -= NE_PENALTY

                result_entity_id = torch.as_tensor(result_entity_id, device=entity_out_max.device)
                # Not required as we don't compute loss
                # TODO: maybe add them in output for debug ?
                entity_out, entity_label_in_sample_id = None, None
            else:
                # There is no entity extracted from src. :(
                # TODO: shall we do anything to rescue it? Or it just that there is no entity in this batch?
                result_entity_id = torch.ones((B, T), dtype=torch.long, device=decoder_out_feature.device) * -1

                entity_out, entity_label_in_sample_id = None, None

                ## Prevent to output any entity
                decoder_out[:, :, self.tgt_ne_start_id:] -= NE_PENALTY

            return ModelOut(
                decoder_out=(decoder_out, extra),
                encoder_ne_logit=encoder_ne_logit,
                decoder_ne_logit=None,
                entity_out=entity_out,
                entity_label=entity_label_in_sample_id,
                result_entity_id=result_entity_id,
                encoder_ne = encoder_ne_pred
            )

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        kwargs.pop("src_ne", None)
        kwargs.pop("kg_embedding", None)
        kwargs.pop("entity_mask", None)
        num_emb = self.encoder.embed_tokens.num_embeddings
        mx = src_tokens.max().item()
        mn = src_tokens.min().item()
        assert mx < num_emb, f"[SRC OOB] max_id={mx} >= num_embeddings={num_emb} (min_id={mn})"

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


@fairseq.models.register_model('entity_transformer')
class EntityTransformer(EntityEncoderDecoderModel):

    def __init__(self, args, ne_dict, encoder, decoder, tgt_ne_start_id, bert_emb_id_dict, bert_emb_value, entity_mapping):
        super().__init__(args, ne_dict, encoder, decoder, tgt_ne_start_id, bert_emb_id_dict, bert_emb_value, entity_mapping)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        EntityEncoderDecoderModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        ##### Copy from transformer.py ####
        # make sure all arguments are present in older models
        base_architecture(args)

        if getattr(args, 'encoder_layers_to_keep', None):
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if getattr(args, 'decoder_layers_to_keep', None):
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = fairseq.models.transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerModel.build_encoder(args, src_dict, encoder_embed_tokens)

        # The decoder-encoder attion K, V is larger, as we combine the NE emb
        if args.mode != 0 and args.concat_ne_emb:
            new_args = copy.deepcopy(args)
            new_args.encoder_embed_dim = args.encoder_embed_dim + args.src_ne_project
            decoder = TransformerModel.build_decoder(new_args, tgt_dict, decoder_embed_tokens)
        else:
            assert args.mode == 0 or args.encoder_embed_dim == args.src_ne_project, f'mode {args.mode}, {args.encoder_embed_dim} != {args.src_ne_project}'
            decoder = TransformerModel.build_decoder(args, tgt_dict, decoder_embed_tokens)

        tgt_ne_start_id = len(task.tgt_dict.lang_dict)
        model = cls(
            args,
            task.ne_dict,
            encoder,
            decoder,
            tgt_ne_start_id,
            task.bert_emb_id_dict,
            task.bert_emb_value,
            task.entity_mapping,
        )

        # ===== Load KG embedding table (|V| x kg_dim), aligned with src_dict =====
        if getattr(args, "kg_embed_path", None):
            kg = torch.load(args.kg_embed_path, map_location="cpu")
            assert isinstance(kg, torch.Tensor), "kg_embed_path must load a torch.Tensor"
            assert kg.dim() == 2, f"kg table must be 2D, got {tuple(kg.shape)}"
            vocab_size = len(src_dict)
            if kg.size(0) < vocab_size:
                # KG table may only cover the base language vocab; pad rows for NE tokens.
                pad_rows = vocab_size - kg.size(0)
                kg = torch.cat([kg, kg.new_zeros((pad_rows, kg.size(1)))], dim=0)
                print(f"| [kg] padded {pad_rows} rows to match src_dict: {kg.size(0)}")
            assert kg.size(0) == vocab_size, f"kg vocab mismatch: {kg.size(0)} vs {vocab_size}"
            if getattr(args, "kg_embed_dim", None) is not None:
                assert kg.size(1) == args.kg_embed_dim, f"kg dim mismatch: {kg.size(1)} vs {args.kg_embed_dim}"

            model.register_buffer("kg_table", kg, persistent=False)
        else:
            model.kg_table = None

        return model


@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer')
def base_architecture(args):
    fairseq.models.transformer.base_architecture(args)
    args.concat_ne_emb = getattr(args, 'concat_ne_emb', False)
    args.src_ne_project = getattr(args, 'src_ne_project', args.encoder_embed_dim)
    args.tgt_ne_project = getattr(args, 'tgt_ne_project', args.src_ne_project)
    args.bert_lookup_layer = getattr(args, 'bert_lookup_layer', args.decoder_layers) # Use last layer by default
    args.src_ne_layer = getattr(args, 'src_ne_layer', args.encoder_layers)
    args.tgt_ne_drop_rate = getattr(args, 'tgt_ne_drop_rate', 0.0)


@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    fairseq.models.transformer.transformer_iwslt_de_en(args)
    base_architecture(args)

@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    fairseq.models.transformer.transformer_vaswani_wmt_en_de_big(args)
    base_architecture(args)
