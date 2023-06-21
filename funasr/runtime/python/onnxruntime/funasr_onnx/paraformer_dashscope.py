# -*- encoding: utf-8 -*-

import os.path
from pathlib import Path
from typing import List, Union, Tuple

import copy
import librosa
import math
import numpy as np

from .utils.utils import (CharTokenizer, Hypothesis, ONNXRuntimeError,
                          OrtInferSession, TokenIDConverter, get_logger,
                          read_yaml)
from .utils.postprocess_utils import sentence_postprocess
from .utils.frontend import WavFrontend, SinusoidalPositionEncoderOnline
from .utils.timestamp_utils import time_stamp_lfr6_onnx

logging = get_logger()


class Paraformer_dashscope():
    def __init__(self, model_dir: Union[str, Path] = None,
                 batch_size: int = 1,
                 chunk_size: List = [15, 90, 15],
                 quantize: bool = False,
                 device_id: Union[str, int] = "-1",
                 pred_bias: int = 1,
                 intra_op_num_threads: int = 4,
                 cache_dir: str = None,
                 ):

        if not Path(model_dir).exists():
            from modelscope.hub.snapshot_download import snapshot_download
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(model_dir)

        encoder_model_file = os.path.join(model_dir, 'encoder.onnx')
        decoder_model_file = os.path.join(model_dir, 'decoder.onnx')
        if quantize:
            encoder_model_file = os.path.join(model_dir, 'encoder_quant.onnx')
            decoder_model_file = os.path.join(model_dir, 'decoder_quant.onnx')
        if not os.path.exists(encoder_model_file) or not os.path.exists(decoder_model_file):
            print(".onnx is not exist, begin to export onnx")
            from funasr.export.export_model import ModelExport
            export_model = ModelExport(
                cache_dir=cache_dir,
                onnx=True,
                device="cpu",
                quant=quantize,
            )
            export_model.export(model_dir)
        config_file = os.path.join(model_dir, 'config.yaml')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        config = read_yaml(config_file)

        self.converter = TokenIDConverter(config['token_list'])
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(
            cmvn_file=cmvn_file,
            **config['frontend_conf']
        )

        self.pe = SinusoidalPositionEncoderOnline()
        self.ort_encoder_infer = OrtInferSession(encoder_model_file, device_id,
                                                 intra_op_num_threads=intra_op_num_threads)
        self.ort_decoder_infer = OrtInferSession(decoder_model_file, device_id,
                                                 intra_op_num_threads=intra_op_num_threads)

        self.encoder_output_size = config["encoder_conf"]["output_size"]
        self.feats_dims = config["frontend_conf"]["n_mels"] * config["frontend_conf"]["lfr_m"]
        self.cif_threshold = config["predictor_conf"]["threshold"]
        self.tail_threshold = config["predictor_conf"]["tail_threshold"]
        self.batch_size = batch_size
        self.pred_bias = pred_bias
        self.chunk_size = chunk_size

    def prepare_cache(self):
        cache = {}
        cache["encoder_outputs"] = None
        cache["alphas"] = None
        cache["last_chunk"] = False
        cache["feats"] = np.zeros((self.chunk_size[0] + self.chunk_size[2], self.feats_dims)).astype(np.float32)
        return cache

    def __call__(self, wav_content: Union[str, np.ndarray, List[str]], **kwargs) -> List:
        data_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        id_list = [item[0] for item in data_list]
        waveform_list = [item[1] for item in data_list]
        waveform_nums = len(waveform_list)
        asr_res = []
        feats_list, feats_len_list = self.extract_feat(waveform_list)

        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            max_chunk_num = 0
            cache = []
            id_list_chunk = []
            for i in range(end_idx - beg_idx):
                cache_tmp = self.prepare_cache()
                cache_tmp["chunk_num"] = math.ceil(feats_len_list[beg_idx+i] / self.chunk_size[1])
                if cache_tmp["chunk_num"] > max_chunk_num:
                    max_chunk_num = cache_tmp["chunk_num"]
                cache.append(cache_tmp)
                id_list_chunk.append(id_list[i])
            try:
                feats = feats_list[beg_idx:end_idx]
                # forward encoder
                for i in range(max_chunk_num):
                    feats_chunk = []
                    feats_chunk_lens = []
                    for j in range(end_idx - beg_idx):
                        if i+1 > cache[j]["chunk_num"]:
                            cache[j]["last_chunk"] = True
                            continue
                        else:
                            feats_tmp = feats[j][i*self.chunk_size[1]:(i+1)*self.chunk_size[1]]
                            feats_tmp = np.concatenate((cache[j]["feats"], feats_tmp), axis=0)
                            cache[j]["feats"] = feats_tmp[-(self.chunk_size[0] + self.chunk_size[2]):]
                            if i+1 == cache[j]["chunk_num"]:
                                cache[j]["last_chunk"] = True

                            feats_chunk.append(feats_tmp)
                            feats_chunk_lens.append(feats_tmp.shape[0])

                    feats_chunk = self.pad_feats(feats_chunk, max(feats_chunk_lens))
                    feats_chunk_lens = np.array(feats_chunk_lens).astype(np.int32)
                    enc, enc_lens, alphas = self.encode_chunk(feats_chunk, feats_chunk_lens)

                    index = 0
                    for j in range(end_idx - beg_idx):
                        if i+1 <= cache[j]["chunk_num"]:
                            if cache[j]["last_chunk"]:
                                enc_remove_overlap = enc[index, self.chunk_size[0]:enc_lens[index], :]
                                alpha_remove_overlap = alphas[index, self.chunk_size[0]:enc_lens[index]]
                            else:
                                enc_remove_overlap = enc[index, self.chunk_size[0]:sum(self.chunk_size[:2]), :]
                                alpha_remove_overlap = alphas[index, self.chunk_size[0]:sum(self.chunk_size[:2])]

                            if cache[j]["encoder_outputs"] is None:
                                cache[j]["encoder_outputs"] = enc_remove_overlap
                                cache[j]["alphas"] = alpha_remove_overlap
                            else:
                                cache[j]["encoder_outputs"] = np.concatenate((cache[j]["encoder_outputs"],
                                                                              enc_remove_overlap), axis=0)
                                cache[j]["alphas"] = np.concatenate((cache[j]["alphas"], alpha_remove_overlap), axis=0)
                            index = index + 1

                enc_list = []
                alpha_list = []
                max_lens = 0
                for j in range(end_idx - beg_idx):
                    enc_list.append(cache[j]["encoder_outputs"])
                    alpha_list.append(np.expand_dims(cache[j]["alphas"], axis=1))
                    if cache[j]["encoder_outputs"].shape[0] > max_lens:
                        max_lens = cache[j]["encoder_outputs"].shape[0]
                enc = self.pad_feats(enc_list, max_lens)
                enc_lens = [item.shape[0] for item in enc_list]
                enc_lens = np.array(enc_lens).astype(np.int32)
                alpha = self.pad_feats(alpha_list, max_lens)
                alpha = np.squeeze(alpha, axis=-1)
                # forward predictor
                pre_acoustic_embeds, pre_token_length = self.alpha_search(enc, alpha)
                # forward decoder
                am_scores, _ = self.decode_chunk(enc, enc_lens, pre_acoustic_embeds, pre_token_length)
            except ONNXRuntimeError:
                logging.warning("input wav is silence or noise")
                preds = ['']
            else:
                preds = self.decode(am_scores, pre_token_length)
                for pred in preds:
                    pred = sentence_postprocess(pred)
                    asr_res.append({'preds': pred})
        return asr_res

    def load_data(self,
                  wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [("test", wav_content)]

        if isinstance(wav_content, str):
            if not os.path.exists(wav_content):
                raise TypeError("The file of {} is not exits".format(str))
            if wav_content.endswith(".wav"):
                return [("test", load_wav(wav_content))]
            elif wav_content.endswith(".scp"):
                with open(wav_content, "r") as file:
                    lines = file.readlines()
                return [(item.strip().split()[0], load_wav(item.strip().split()[1])) for item in lines]

        if isinstance(wav_content, list):
            return [("test", load_wav(path)) for path in wav_content]
        raise TypeError(
            f'The type of {wav_content} is not in [str, np.ndarray]')

    def extract_feat(self,
                     waveform_list: List[np.ndarray]
                     ) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feat *= self.encoder_output_size ** 0.5
            feat = self.pe.forward(np.expand_dims(feat, axis=0))
            feats.append(np.squeeze(feat, axis=0))
            feats_len.append(feat_len)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, 'constant', constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def encode_chunk(self, feats: np.ndarray, feats_len: np.ndarray):
        # encoder forward
        enc_input = [feats, feats_len]
        enc, enc_lens, cif_alphas = self.ort_encoder_infer(enc_input)
        return enc, enc_lens, cif_alphas

    def decode_chunk(self, enc, enc_len, pre_acoustic_embeds, pre_token_length):
        # decoder forward
        dec_input = [enc, enc_len, pre_acoustic_embeds, pre_token_length]
        dec, pre_token_length = self.ort_decoder_infer(dec_input)
        return dec, pre_acoustic_embeds

    def decode(self, am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [self.decode_one(am_score, token_num)
                for am_score, token_num in zip(am_scores, token_nums)]

    def decode_one(self,
                   am_score: np.ndarray,
                   valid_token_num: int) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        # asr_model.sos:1  asr_model.eos:2
        yseq = np.array([1] + yseq.tolist() + [2])
        hyp = Hypothesis(yseq=yseq, score=score)

        # remove sos/eos and get results
        last_pos = -1
        token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x not in (0, 2), token_int))

        # Change integer-ids to tokens
        token = self.converter.ids2tokens(token_int)
        token = token[:valid_token_num - self.pred_bias]
        # texts = sentence_postprocess(token)
        return token

    def alpha_search(self, hidden, alphas):
        batch_size, len_time, hidden_size = hidden.shape
        token_length = []
        list_fires = []
        list_frames = []
        if self.tail_threshold > 0.0:
            tail_hidden = np.zeros((batch_size, 1, hidden_size)).astype(np.float32)
            tail_alphas = np.array([[self.tail_threshold]]).astype(np.float32)
            tail_alphas = np.tile(tail_alphas, (batch_size, 1))
            hidden = np.concatenate((hidden, tail_hidden), axis=1)
            alphas = np.concatenate((alphas, tail_alphas), axis=1)

        len_time = alphas.shape[1]
        for b in range(batch_size):
            integrate = 0.0
            frames = np.zeros(hidden_size).astype(np.float32)
            list_frame = []
            list_fire = []
            for t in range(len_time):
                alpha = alphas[b][t]
                if alpha + integrate < self.cif_threshold:
                    integrate += alpha
                    list_fire.append(integrate)
                    frames += alpha * hidden[b][t]
                else:
                    frames += (self.cif_threshold - integrate) * hidden[b][t]
                    list_frame.append(frames)
                    integrate += alpha
                    list_fire.append(integrate)
                    integrate -= self.cif_threshold
                    frames = integrate * hidden[b][t]

            token_length.append(len(list_frame))
            list_frames.append(list_frame)

        max_token_len = max(token_length)
        list_ls = []
        for b in range(batch_size):
            pad_frames = np.zeros((max_token_len - token_length[b], hidden_size)).astype(np.float32)
            if token_length[b] == 0:
                list_ls.append(pad_frames)
            else:
                list_ls.append(np.concatenate((list_frames[b], pad_frames), axis=0))

        return np.stack(list_ls, axis=0).astype(np.float32), np.stack(token_length, axis=0).astype(np.int32)

