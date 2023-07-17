#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class DecoderLayerSANM(nn.Module):

    def __init__(
        self,
        model
    ):
        super().__init__()
        self.self_attn = model.self_attn
        self.src_attn = model.src_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2 if hasattr(model, 'norm2') else None
        self.norm3 = model.norm3 if hasattr(model, 'norm3') else None
        self.size = model.size

    def forward(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):

        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x, cache = self.self_attn(tgt, tgt_mask, cache=cache)
            x = residual + x

        if self.src_attn is not None:
            residual = x
            x = self.norm3(x)
            x = residual + self.src_attn(x, memory, memory_mask)

        return x, tgt_mask, memory, memory_mask, cache

class DecoderLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.self_attn = model.self_attn
        self.src_attn = model.src_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.norm3 = model.norm3
        self.size = model.size
        self.normalize_before = model.normalize_before
        self.concat_after = model.concat_after
        if self.concat_after:
            self.concat_linear1 = model.concat_linear1
            self.concat_linear2 = model.concat_linear2

    def forward(self, x, mask, memory, memory_mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        residual = x

        if self.normalize_before:
            x = self.norm1(x)

        if cache is not None:
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if mask is not None:
                tgt_q_mask = mask[:, :, -1:]
        else:
            x_q = x
            tgt_q_mask = mask

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, tgt_q_mask)), dim=-1)
            x = self.concat_linear(x_concat) + residual
        else:
            x = self.self_attn(x_q, x, x, tgt_q_mask) + residual

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x

        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = self.concat_linear2(x_concat) + residual
        else:
            x = self.src_attn(x, memory, memory, memory_mask) + residual
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)

        x = self.feed_forward(x) + residual
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask
