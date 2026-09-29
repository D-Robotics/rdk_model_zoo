"""Fixed-shape Torch export boundaries; no file I/O or global monkey patches.

Decoder composition follows FunASR contextual_paraformer/export_meta.py and
contextual_paraformer/decoder.py (MIT, Copyright FunASR contributors).
CIF remains outside the exported models, in the shared NumPy CPU bridge.
"""

import torch
from torch import nn


def fixed_mask(lengths, width):
    """Keep a fixed physical width while masking the variable valid prefix."""
    return (
        torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]
    ).float()


class EncoderStage(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("lengths", torch.tensor([400], dtype=torch.int32))

    def forward(self, speech):
        context, _ = self.encoder(speech, self.lengths)
        return context.reshape(1, 400, 512)


class PredictorStage(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("mask", torch.ones(1, 1, 400))

    def forward(self, context):
        alphas, _ = self.predictor.forward_cnn(context, self.mask)
        hidden, alphas, _ = self.predictor.tail_process_fn(
            context, alphas, mask=self.mask[:, 0, :]
        )
        return alphas.reshape(1, 401), hidden.reshape(1, 401, 512)


class DecoderStage(nn.Module):
    """Source decoder composition with explicit 100/400/1 mask geometry.

    Unlike the generic upstream decoder, token_num changes mask values, not the
    tensor width. This is the source deployment's padded 100-token contract.
    All rows are returned; the application decodes only the valid prefix.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("memory_mask", torch.zeros(1, 1, 1, 400))
        self.register_buffer("contextual_mask", torch.ones(1, 1, 1, 1))

    def forward(self, context, token_num, bias_embed, acoustic):
        decoder = self.decoder
        tgt_mask = fixed_mask(token_num, 100)[:, :, None]
        memory_mask = self.memory_mask
        x, tgt_mask, memory, memory_mask, _ = decoder.model.decoders(
            acoustic, tgt_mask, context, memory_mask
        )
        _, _, x_self_attn, x_src_attn = decoder.last_decoder(
            x, tgt_mask, memory, memory_mask
        )
        cx, tgt_mask, _, _, _ = decoder.bias_decoder(
            x_self_attn, tgt_mask, bias_embed, memory_mask=self.contextual_mask
        )
        if decoder.bias_output is not None:
            x = torch.cat([x_src_attn, cx], dim=2)
            x = decoder.bias_output(x.transpose(1, 2)).transpose(1, 2)
            x = x_self_attn + decoder.dropout(x)
        if decoder.model.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = decoder.model.decoders2(
                x, tgt_mask, memory, memory_mask
            )
        x, tgt_mask, memory, memory_mask, _ = decoder.model.decoders3(
            x, tgt_mask, memory, memory_mask
        )
        x = decoder.output_layer(decoder.after_norm(x))
        return torch.log_softmax(x, dim=-1).reshape(1, 100, 8404)


def build_stages(model):
    """Wrap a freshly loaded, caller-owned model; upstream wrappers mutate it."""
    from funasr.models.sanm.encoder import SANMEncoderExport
    from funasr.models.paraformer.cif_predictor import CifPredictorV2Export
    from funasr.models.contextual_paraformer.decoder import (
        ContextualParaformerDecoderExport,
    )

    return {
        "encoder": EncoderStage(SANMEncoderExport(model.encoder)).eval(),
        "predictor": PredictorStage(CifPredictorV2Export(model.predictor)).eval(),
        "decoder": DecoderStage(
            ContextualParaformerDecoderExport(model.decoder)
        ).eval(),
    }
