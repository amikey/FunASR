import glob
import logging
import os
import warnings
from abc import ABC
from typing import List

import onnxruntime

from funasr_onnx.postprocess.build_tokenizer import build_tokenizer
#from funasr_onnx.postprocess.token_id_converter import TokenIDConverter
from funasr_onnx.utils.utils import TokenIDConverter
from funasr_onnx.utils.config import get_config, get_tag_config


class AbsModel(ABC):
    def _check_argument(self, model_dir):
        self.model_dir = model_dir

        if model_dir is None:
            raise ValueError("tag_name or model_dir should be defined.")

    def _load_config(self):
        config_file = glob.glob(os.path.join(self.model_dir, "config.*"))[0]
        self.config = get_config(config_file)

    def _build_tokenizer(self):
        if self.config.tokenizer.token_type is None:
            self.tokenizer = None
        elif self.config.tokenizer.token_type == "bpe":
            self.tokenizer = build_tokenizer("bpe", self.config.tokenizer.bpemodel)
        else:
            self.tokenizer = build_tokenizer(**self.config.tokenizer)

    def _build_token_converter(self):
        self.converter = TokenIDConverter(token_list=self.config.token.list)

    def _build_model(self, providers, use_quantized):
        raise NotImplementedError

    def _check_ort_version(self, providers: List[str]):
        # check cpu
        if (
            onnxruntime.get_device() == "CPU"
            and "CPUExecutionProvider" not in providers
        ):
            raise RuntimeError(
                "If you want to use GPU, then follow `How to use GPU on espnet_onnx` chapter in readme to install onnxruntime-gpu."
            )

        # check GPU
        if onnxruntime.get_device() == "GPU" and providers == ["CPUExecutionProvider"]:
            warnings.warn(
                "Inference will be executed on the CPU. Please provide gpu providers. Read `How to use GPU on espnet_onnx` in readme in detail."
            )

        logging.info(f'Providers [{" ,".join(providers)}] detected.')

