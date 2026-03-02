import torch
import math

from fairseq import search
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        # Disable incremental cache to avoid key/value desync shape mismatch
        self.incremental_states = None

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        # Clear incremental state per new batch to avoid stale state shape mismatch
        if self.incremental_states is not None:
            for k in self.incremental_states:
                self.incremental_states[k].clear()
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1., src_tokens=None):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                src_tokens=src_tokens
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1., src_tokens=None
    ):
        decoder_kwargs = {}
        if src_tokens is not None:
            decoder_kwargs["src_tokens"] = src_tokens
        input_tokens = tokens
        if self.incremental_states is not None:
            # Standard fairseq incremental decoding: feed only last token
            input_tokens = tokens[:, -1:]
            decoder_out = model.forward_decoder(
                input_tokens,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states[model],
                **decoder_kwargs,
            )
        else:
            decoder_out = model.forward_decoder(input_tokens, encoder_out=encoder_out, **decoder_kwargs)
        
        decoder_out_internel = decoder_out.decoder_out[0][:, -1:, :]

        if temperature != 1.:
            decoder_out_internel.div_(temperature)
        
        attn = decoder_out.decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            if isinstance(attn, list):
                # Compatible with new fairseq where decoder_attn may be list[Tensor]
                attn = attn[0]
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs((decoder_out_internel, None), log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn, decoder_out

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        # 1. Drop encoder_states first to avoid index_select errors on NestedTensor
        for enc_out in encoder_outs:
            if enc_out is not None and "encoder_states" in enc_out:
                # Keep empty list so downstream len() checks don't fail
                enc_out["encoder_states"] = []
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)
