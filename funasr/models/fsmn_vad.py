# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from typeguard import check_argument_types
import numpy as np

from funasr.layers.abs_normalize import AbsNormalize
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.modules.nets_utils import th_accuracy
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.models.encoder.fsmn_encoder import FSMN
from funasr.models.base_model import FunASRModel
import time

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

def load_cmvn(cmvn_file):
    with open(cmvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float)
    vars = np.array(vars_list).astype(np.float)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn


def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    dtype = inputs.dtype
    _, frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

    return inputs.type(torch.float32)


# def apply_lfr(inputs, lfr_m, lfr_n):
    # LFR_inputs = []
    # T = inputs.shape[0]
    # T_lfr = int(np.ceil(T / lfr_n))
    # left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    # inputs = torch.vstack((left_padding, inputs))
    # T = T + (lfr_m - 1) // 2
    # for i in range(T_lfr):
        # if lfr_m <= T - i * lfr_n:
            # LFR_inputs.append((inputs[i * lfr_n:i * lfr_n + lfr_m]).view(1, -1))
        # else:  # process last LFR frame
            # num_padding = lfr_m - (T - i * lfr_n)
            # frame = (inputs[i * lfr_n:]).view(-1)
            # for _ in range(num_padding):
                # frame = torch.hstack((frame, inputs[-1]))
            # LFR_inputs.append(frame)
    # LFR_outputs = torch.vstack(LFR_inputs)
    # return LFR_outputs.type(torch.float32)

def apply_lfr(inputs):
    left_padding = inputs[:, 0:1, :].repeat(1, 2, 1)
    right_padding = inputs[:, -1:, :].repeat(1, 2, 1)
    inputs = torch.cat((left_padding, inputs, right_padding), dim=1)
    outputs = torch.cat((inputs[:, :-4, :], inputs[:, 1:-3, :], inputs[:, 2:-2, :], inputs[:, 3:-1, :], inputs[:, 4:, :]), dim=2)
    return outputs

class FsmnVadModel(FunASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
            self,
            frontend: Optional[AbsFrontend],
            specaug: Optional[AbsSpecAug],
            normalize: Optional[AbsNormalize],
            encoder: FSMN,
            extract_feats_in_collect_stats: bool = True,
            ignore_id: int = -1,
            feature_transform: Dict[str, Union[str, int]] = dict()
    ):
        assert check_argument_types()

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.ignore_id = ignore_id
        self.softmax = torch.nn.Softmax(dim=2)
        self.criterion = nn.CrossEntropyLoss() 
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.feature_transform = feature_transform
        # self.cmvn = None
        # self.lfr_m = 0
        # self.lfr_n = 0
        if self.feature_transform:
            self.cmvn = load_cmvn(feature_transform['cmvn_file'])
            self.lfr_m = feature_transform['lfr_m']
            self.lfr_n = feature_transform['lfr_n']

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out = self.encode(speech, speech_lengths)
        scores = self.softmax(encoder_out)
        acc = th_accuracy(
            scores.view(-1, scores.shape[-1]),
            text,
            ignore_label=self.ignore_id,
        )
        stats = dict()
        min_t = min(encoder_out.shape[1], text.shape[1])
        loss = self.criterion(encoder_out[:, :min_t, :].view(-1, encoder_out.shape[-1]), text[:, :min_t].view(-1))

        # Collect Attn branch stats
        stats["acc"] = acc

        # Collect total loss stats
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Frontend + Encoder. 

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)


        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out = self.encoder(feats, in_cache=dict())

        return encoder_out

    def _extract_feats(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
            if self.feature_transform:
                feats = apply_lfr(feats)
                feats = apply_cmvn(feats, self.cmvn)
                # for idx, feat in enumerate(feats):
                    # feat_lfr = apply_lfr(feat, self.lfr_m, self.lfr_n)
                    # feats_lfr_cmvn.append(apply_cmvn(feat_lfr, self.cmvn))
                # feats = torch.stack(feats_lfr_cmvn)
        return feats, feats_lengths

