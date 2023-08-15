import torch
import torch.nn as nn

from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask
from funasr.modules.attention import MultiHeadedAttentionSANM
from funasr.export.models.modules.multihead_att import MultiHeadedAttentionSANM as MultiHeadedAttentionSANM_export
from funasr.export.models.modules.encoder_layer import EncoderLayerSANM as EncoderLayerSANM_export
from funasr.export.models.modules.encoder_layer import OnnxConformerLayer as EncoderLayerConformer_export
from funasr.modules.positionwise_feed_forward import PositionwiseFeedForward
from funasr.export.models.modules.feedforward import PositionwiseFeedForward as PositionwiseFeedForward_export
from funasr.export.models.encoder.sanm_encoder import SANMEncoder
from funasr.modules.attention import RelPositionMultiHeadedAttention
# from funasr.export.models.modules.multihead_att import RelPositionMultiHeadedAttention as RelPositionMultiHeadedAttention_export
from funasr.export.models.modules.multihead_att import OnnxRelPosMultiHeadedAttention as RelPositionMultiHeadedAttention_export

class ConformerEncoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        max_seq_len = 512,
        feats_dim = 560,
        model_name = 'encoder',
        onnx: bool = True,
    ):
        super().__init__()
        self.embed = model.embed
        self.model = model

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        self.feats_dim = feats_dim

        # replace multihead attention module into customized module.
        for i, d in enumerate(self.model.encoders):
            # d is EncoderLayer
            if isinstance(d.self_attn, MultiHeadedAttentionSANM):
                d.self_attn = MultiHeadedAttentionSANM_export(d.self_attn)
            if isinstance(d.self_attn, RelPositionMultiHeadedAttention):
                d.self_attn = RelPositionMultiHeadedAttention_export(d.self_attn)
            if isinstance(d.feed_forward, PositionwiseFeedForward):
                d.feed_forward = PositionwiseFeedForward_export(d.feed_forward)
            self.model.encoders[i] = EncoderLayerConformer_export(d)

        self.model_name = model_name
        self.num_heads = model.encoders[0].model.self_attn.h
        self.hidden_size = model.encoders[0].model.self_attn.linear_out.out_features

    def prepare_mask(self, mask):
        if len(mask.shape) == 2:
            mask = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask = 1 - mask[:, None, :]

        return mask * -10000.0

    def forward(self, feats):
        feats_length = torch.ones(feats[:, :, 0].shape).sum(dim=-1).type(torch.long)
        mask = self.make_pad_mask(feats_length)
        xs_pad = self.embed(feats)
        mask = self.prepare_mask(mask)

        intermediate_outs = []
        if len(self.model.interctc_layer_idx) == 0:
            xs_pad, mask = self.model.encoders(xs_pad, mask)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, mask = encoder_layer(xs_pad, mask)

                if layer_idx + 1 in self.model.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.model.normalize_before:
                        encoder_out = self.model.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.model.interctc_use_conditioning:
                        ctc_out = self.ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.model.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.model.conditioning_layer(ctc_out)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.model.normalize_before:
            xs_pad = self.model.after_norm(xs_pad)

        olens = mask.squeeze(1).sum(1)
        return xs_pad, olens

    def get_output_size(self):
        return self.model.encoders[0].size

    def is_optimizable(self):
        return True

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        return feats

    def get_input_names(self):
        return ["feats"]

    def get_output_names(self):
        return ["encoder_out", "encoder_out_lens"]

    def get_dynamic_axes(self):
        return {"feats": {1: "feats_length"}, "encoder_out": {1: "enc_out_length"}}

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        ret.update(
            enc_type="XformerEncoder",
            model_path=os.path.join(path, f"{self.model_name}.onnx"),
            is_vggrnn=False,
            frontend=get_frontend_config(
                asr_model.frontend, self.frontend_model, path=path
            ),
            do_normalize=asr_model.normalize is not None,
            do_postencoder=asr_model.postencoder is not None,
        )
        if ret["do_normalize"]:
            ret.update(normalize=get_norm_config(asr_model.normalize, path))
        # Currently postencoder is not supported.
        # if ret['do_postencoder']:
        #     ret.update(postencoder=get_postenc_config(self.model.postencoder))
        return ret