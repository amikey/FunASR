#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        model,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = model.self_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.in_size = model.in_size
        self.size = model.size

    def forward(self, x, mask):

        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        if self.in_size == self.size:
            x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x, mask


class EncoderLayerConformer(nn.Module):
    def __init__(
        self,
        model,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = model.self_attn
        self.feed_forward = model.feed_forward
        self.feed_forward_macaron = model.feed_forward_macaron
        self.conv_module = model.conv_module
        self.norm_ff = model.norm_ff
        self.norm_mha = model.norm_mha
        self.norm_ff_macaron = model.norm_ff_macaron
        self.norm_conv = model.norm_conv
        self.norm_final = model.norm_final
        self.size = model.size

    def forward(self, x, mask):
        if isinstance(x, tuple):
            x, pos_emb = x[0], x[1]
        else:
            x, pos_emb = x, None

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.feed_forward_macaron(x) * 0.5

        residual = x
        x = self.norm_mha(x)

        x_q = x

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)
        x = residual + x_att

        if self.conv_module is not None:
            residual = x
            x = self.norm_conv(x)
            x = residual +  self.conv_module(x)

        residual = x
        x = self.norm_ff(x)
        x = residual + self.feed_forward(x) * 0.5

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask

class OnnxConformerLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(self, model):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.model = model
        self.size = model.size
        self.stoch_layer_coeff = 1.0

    def forward(self, x_input, mask, cache=None):
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.model.feed_forward_macaron is not None:
            residual = x
            if self.model.normalize_before:
                x = self.model.norm_ff_macaron(x)
            x = (
                residual
                + self.stoch_layer_coeff
                * self.model.ff_scale
                * self.model.feed_forward_macaron(x)
            )
            if not self.model.normalize_before:
                x = self.model.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.model.normalize_before:
            x = self.model.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.model.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.model.self_attn(x_q, x, x, mask)

        if self.model.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.stoch_layer_coeff * self.model.concat_linear(x_concat)
        else:
            x = residual + self.stoch_layer_coeff * x_att
        if not self.model.normalize_before:
            x = self.model.norm_mha(x)

        # convolution module
        if self.model.conv_module is not None:
            residual = x
            if self.model.normalize_before:
                x = self.model.norm_conv(x)
            x = residual + self.stoch_layer_coeff * self.model.conv_module(x)
            if not self.model.normalize_before:
                x = self.model.norm_conv(x)

        # feed forward module
        residual = x
        if self.model.normalize_before:
            x = self.model.norm_ff(x)
        x = (
            residual
            + self.stoch_layer_coeff * self.model.ff_scale * self.model.feed_forward(x)
        )
        if not self.model.normalize_before:
            x = self.model.norm_ff(x)

        if self.model.conv_module is not None:
            x = self.model.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask