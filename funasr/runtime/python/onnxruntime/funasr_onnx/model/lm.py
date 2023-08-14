from typing import List

from funasr_onnx.model.lms.seqrnn_lm import SequentialRNNLM
from funasr_onnx.model.lms.transformer_lm import TransformerLM
from funasr_onnx.utils.config import Config


def get_lm(config: Config, providers: List[str], use_quantized: bool = False):
    if config.lm.use_lm:
        if config.lm.lm_type == "SequentialRNNLM":
            return SequentialRNNLM(config.lm, providers, use_quantized)
        elif config.lm.lm_type == "TransformerLM":
            return TransformerLM(config.lm, providers, use_quantized)
    return None
