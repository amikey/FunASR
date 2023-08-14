from typing import List, Tuple

import numpy as np
import onnxruntime

from funasr_onnx.frontend.frontend import Frontend
from funasr_onnx.frontend.normalize.global_mvn import GlobalMVN
from funasr_onnx.frontend.normalize.utterance_mvn import UtteranceMVN
from funasr_onnx.utils.config import Config
from funasr_onnx.utils.function import make_pad_mask, mask_fill
from funasr_onnx.utils.frontend import WavFrontend

class Encoder:
    def __init__(
        self,
        encoder_config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.config = encoder_config
        # Note that id model was optimized and quantized,
        # then the quantized model should be optimized.
        if use_quantized:
            self.encoder = onnxruntime.InferenceSession(
                self.config.quantized_model_path, providers=providers
            )
        else:
            self.encoder = onnxruntime.InferenceSession(
                self.config.model_path, providers=providers
            )

        #wav frontend
        if self.config.frontend == 'wav_frontend':
            self.frontend = WavFrontend(self.config.cmvn_file, **self.config['frontend_conf'])
        #self.frontend = Frontend(self.config.frontend, providers, use_quantized)
        #if self.config.do_normalize:
        #    if self.config.normalize.type == "gmvn":
        #        self.normalize = GlobalMVN(self.config.normalize)
        #    elif self.config.normalize.type == "utterance_mvn":
        #        self.normalize = UtteranceMVN(self.config.normalize)

        # if self.config.do_preencoder:
        #     self.preencoder = Preencoder(self.config.preencoder)

        # if self.config.do_postencoder:
        #     self.postencoder = Postencoder(self.config.postencoder)

    def __call__(
        self, speech: np.ndarray, speech_length: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # 1. Extract feature and make lfr cmvn
        if self.config.frontend == 'wav_frontend':
            fbank, _ = self.frontend.fbank(speech)
            feats, feat_length = self.frontend.lfr_cmvn(fbank)
            feats = np.expand_dims(feats, axis = 0).astype(np.float32)
            feat_length = np.expand_dims(feat_length, axis = 0)
  
        print("feats shape {}".format(feats.shape))
        print("feat_length {}".format(feat_length))
        # 3. forward encoder
        encoder_out, encoder_out_lens = self.forward_encoder(feats, feat_length)
        #encoder_out = self.mask_output(encoder_out, encoder_out_lens)

        # if self.config.do_postencoder:
        #     encoder_out, encoder_out_lens = self.postencoder(
        #         encoder_out, encoder_out_lens
        #     )
        #if isinstance(encoder_out, tuple):
        #    encoder_out = encoder_out[0]
        #assert len(encoder_out) == 1, len(encoder_out)
    
        #print("encoder out shape {}".format(encoder_out.shape))
        #print("encoder out lens {}".format(encoder_out_lens.shape))
        #print("encoder out lens {} ".format(encoder_out_lens))

        return encoder_out, encoder_out_lens

    def mask_output(self, feats, feat_length):
        if self.config.is_vggrnn:
            feats = mask_fill(feats, make_pad_mask(feat_length, feats, 1), 0.0)
        return feats, feat_length

    def forward_encoder(self, feats, feat_length):
        encoder_out, encoder_out_lens = self.encoder.run(
            ["encoder_out", "encoder_out_lens"], {"feats": feats}
        )

        if self.config.enc_type == "RNNEncoder":
            encoder_out = mask_fill(
                encoder_out, make_pad_mask(feat_length, encoder_out, 1), 0.0
            )

        return encoder_out, encoder_out_lens
