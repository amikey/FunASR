import logging
import torch
import torch.nn as nn

from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask
from funasr.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.export.models.encoder.sanm_encoder import SANMEncoder as SANMEncoder_export
from funasr.export.models.encoder.conformer_encoder import ConformerEncoder as ConformerEncoder_export


class SemanticVADModel(nn.Module):
    """
    Author: Speech Lab, Alibaba Group, China
    """
    
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=80,
        model_name='model',
        **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        if isinstance(model.encoder, SANMEncoder) or isinstance(model.encoder, SANMEncoderChunkOpt):
            self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)
        elif isinstance(model.encoder, ConformerEncoder):
            self.encoder = ConformerEncoder_export(model.encoder, onnx=onnx)
        else:
            raise "Not support model encoder"
        
        self.feats_dim = feats_dim
        self.model_name = model_name
        self.stride_conv = model.stride_conv
        self.point_linear_layer = model.point_linear_layer
        self.vad_linear_layer = model.vad_linear_layer
        self.point_classifier = model.point_classifier
        self.classifier = model.classifier
        self.leaky_relu= model.leaky_relu

    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        # a. To device
        # batch = {"speech": speech, "speech_lengths": speech_lengths}
        # batch = to_device(batch, device=self.device)
        speech_stride, speech_lengths_stride = self.stride_conv(speech, speech_lengths)
        enc, enc_len = self.encoder(speech_stride, speech_lengths_stride)
        point_hid_out = self.point_linear_layer(enc)
        point_out = self.leaky_relu(point_hid_out)
        point_out = self.point_classifier(point_out)

        vad_out = self.vad_linear_layer(enc + point_hid_out)
        vad_out = self.leaky_relu(vad_out)
        vad_out = self.classifier(vad_out)
        
        return vad_out, point_out
    
    def get_dummy_inputs(self):
        #speech = torch.randn(2, 30, self.feats_dim)
        speech = torch.randn(1, 270, self.feats_dim)
        #speech_lengths = torch.tensor([25, 30], dtype=torch.int32)
        speech_lengths = torch.tensor([270], dtype=torch.int32)
        return (speech, speech_lengths)
    
    def get_input_names(self):
        return ['speech', 'speech_lengths']
    
    def get_output_names(self):
        return ['vad_out', 'point_out']
    
    def get_dynamic_axes(self):
        return {
            'speech': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'speech_lengths': {
                0: 'batch_size',
            },
        }
