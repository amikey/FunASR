import os

import torch
import torch.nn as nn


from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask

from funasr.modules.attention import MultiHeadedAttentionSANMDecoder
from funasr.export.models.modules.multihead_att import MultiHeadedAttentionSANMDecoder as MultiHeadedAttentionSANMDecoder_export
from funasr.modules.attention import MultiHeadedAttentionCrossAtt
from funasr.export.models.modules.multihead_att import MultiHeadedAttentionCrossAtt as MultiHeadedAttentionCrossAtt_export
from funasr.modules.positionwise_feed_forward import PositionwiseFeedForwardDecoderSANM
from funasr.export.models.modules.feedforward import PositionwiseFeedForwardDecoderSANM as PositionwiseFeedForwardDecoderSANM_export
from funasr.export.models.modules.decoder_layer import DecoderLayerSANM as DecoderLayerSANM_export


class ParaformerSANMDecoder(nn.Module):
    def __init__(self, model,
                 max_seq_len=512,
                 model_name='decoder',
                 onnx: bool = True,):
        super().__init__()
        # self.embed = model.embed #Embedding(model.embed, max_seq_len)
        self.model = model
        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        for i, d in enumerate(self.model.decoders):
            if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                d.feed_forward = PositionwiseFeedForwardDecoderSANM_export(d.feed_forward)
            if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                d.self_attn = MultiHeadedAttentionSANMDecoder_export(d.self_attn)
            if isinstance(d.src_attn, MultiHeadedAttentionCrossAtt):
                d.src_attn = MultiHeadedAttentionCrossAtt_export(d.src_attn)
            self.model.decoders[i] = DecoderLayerSANM_export(d)

        if self.model.decoders2 is not None:
            for i, d in enumerate(self.model.decoders2):
                if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                    d.feed_forward = PositionwiseFeedForwardDecoderSANM_export(d.feed_forward)
                if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                    d.self_attn = MultiHeadedAttentionSANMDecoder_export(d.self_attn)
                self.model.decoders2[i] = DecoderLayerSANM_export(d)

        for i, d in enumerate(self.model.decoders3):
            if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                d.feed_forward = PositionwiseFeedForwardDecoderSANM_export(d.feed_forward)
            self.model.decoders3[i] = DecoderLayerSANM_export(d)
        
        self.output_layer = model.output_layer
        self.after_norm = model.after_norm
        self.model_name = model_name
        

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0
    
        return mask_3d_btd, mask_4d_bhlt

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ):

        tgt = ys_in_pad
        tgt_mask = self.make_pad_mask(ys_in_lens)
        tgt_mask, _ = self.prepare_mask(tgt_mask)
        # tgt_mask = myutils.sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]

        memory = hs_pad
        memory_mask = self.make_pad_mask(hlens)
        _, memory_mask = self.prepare_mask(memory_mask)
        # memory_mask = myutils.sequence_mask(hlens, device=memory.device)[:, None, :]

        x = tgt
        x, tgt_mask, memory, memory_mask, _ = self.model.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.model.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = self.model.decoders2(
                x, tgt_mask, memory, memory_mask
            )
        x, tgt_mask, memory, memory_mask, _ = self.model.decoders3(
            x, tgt_mask, memory, memory_mask
        )
        x = self.after_norm(x)
        x = self.output_layer(x)

        return x, ys_in_lens


class ParaformerSANMDecoderOnline(nn.Module):
    def __init__(self, model,
                 max_seq_len=512,
                 model_name='decoder',
                 onnx: bool = True, ):
        super().__init__()
        # self.embed = model.embed #Embedding(model.embed, max_seq_len)
        self.model = model
        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)
        
        for i, d in enumerate(self.model.decoders):
            if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                d.feed_forward = PositionwiseFeedForwardDecoderSANM_export(d.feed_forward)
            if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                d.self_attn = MultiHeadedAttentionSANMDecoder_export(d.self_attn)
            if isinstance(d.src_attn, MultiHeadedAttentionCrossAtt):
                d.src_attn = MultiHeadedAttentionCrossAtt_export(d.src_attn)
            self.model.decoders[i] = DecoderLayerSANM_export(d)
        
        if self.model.decoders2 is not None:
            for i, d in enumerate(self.model.decoders2):
                if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                    d.feed_forward = PositionwiseFeedForwardDecoderSANM_export(d.feed_forward)
                if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                    d.self_attn = MultiHeadedAttentionSANMDecoder_export(d.self_attn)
                self.model.decoders2[i] = DecoderLayerSANM_export(d)
        
        for i, d in enumerate(self.model.decoders3):
            if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                d.feed_forward = PositionwiseFeedForwardDecoderSANM_export(d.feed_forward)
            self.model.decoders3[i] = DecoderLayerSANM_export(d)
        
        self.output_layer = model.output_layer
        self.after_norm = model.after_norm
        self.model_name = model_name
    
    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0
        
        return mask_3d_btd, mask_4d_bhlt
    
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        *args,
    ):
        
        tgt = ys_in_pad
        tgt_mask = self.make_pad_mask(ys_in_lens)
        tgt_mask, _ = self.prepare_mask(tgt_mask)
        # tgt_mask = myutils.sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]
        
        memory = hs_pad
        memory_mask = self.make_pad_mask(hlens)
        _, memory_mask = self.prepare_mask(memory_mask)
        # memory_mask = myutils.sequence_mask(hlens, device=memory.device)[:, None, :]
        
        x = tgt
        out_caches = list()
        for i, decoder in enumerate(self.model.decoders):
            in_cache = args[i]
            x, tgt_mask, memory, memory_mask, out_cache = decoder(
                x, tgt_mask, memory, memory_mask, cache=in_cache
            )
            out_caches.append(out_cache)
        if self.model.decoders2 is not None:
            for i, decoder in enumerate(self.model.decoders2):
                in_cache = args[i+len(self.model.decoders)]
                x, tgt_mask, memory, memory_mask, out_cache = decoder(
                    x, tgt_mask, memory, memory_mask, cache=in_cache
                )
                out_caches.append(out_cache)
        x, tgt_mask, memory, memory_mask, _ = self.model.decoders3(
            x, tgt_mask, memory, memory_mask
        )
        x = self.after_norm(x)
        x = self.output_layer(x)
        
        return x, out_caches
    
    def get_dummy_inputs(self, enc_size):
        enc = torch.randn(2, 100, enc_size).type(torch.float32)
        enc_len = torch.tensor([30, 30], dtype=torch.int32)
        acoustic_embeds = torch.randn(2, 10, enc_size).type(torch.float32)
        acoustic_embeds_len = torch.tensor([5, 10], dtype=torch.int32)
        cache_num = len(self.model.decoders) + len(self.model.decoders2)
        cache = [
            torch.zeros((1, self.model.decoders[0].size, self.model.decoders[0].self_attn.kernel_size-1), dtype=torch.float32)
            for _ in range(cache_num)
        ]
        return (enc, enc_len, acoustic_embeds, acoustic_embeds_len, cache)
    

