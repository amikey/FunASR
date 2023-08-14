from typing import List

from funasr_onnx.model.decoders.rnn import RNNDecoder
from funasr_onnx.model.decoders.transducer import TransducerDecoder
from funasr_onnx.model.decoders.xformer import XformerDecoder
from funasr_onnx.utils.config import Config


def get_decoder(config: Config, providers: List[str], use_quantized: bool = False):
    if config.dec_type == "RNNDecoder":
        return RNNDecoder(config, providers, use_quantized)
    elif config.dec_type == "TransducerDecoder":
        return TransducerDecoder(config, providers, use_quantized)
    else:
        return XformerDecoder(config, providers, use_quantized)
