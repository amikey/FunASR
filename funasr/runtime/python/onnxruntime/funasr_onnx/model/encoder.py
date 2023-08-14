from typing import List

from funasr_onnx.model.encoders.encoder import Encoder
from funasr_onnx.model.encoders.streaming import StreamingEncoder
from funasr_onnx.utils.config import Config


def get_encoder(config: Config, providers: List[str], use_quantized: bool = False):
    if config.enc_type == "ContextualXformerEncoder":
        return StreamingEncoder(config, providers, use_quantized)
    else:
        return Encoder(config, providers, use_quantized)
