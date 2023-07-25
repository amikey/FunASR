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
from typeguard import check_argument_types

from funasr.layers.abs_normalize import AbsNormalize
from funasr.losses.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.modules.add_sos_eos import add_sos_eos
from funasr.modules.e2e_asr_common import ErrorCalculator
from funasr.modules.nets_utils import th_accuracy
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.models.base_model import FunASRModel
from funasr.modules.streaming_utils.chunk_utilis import sequence_mask
from funasr.models.predictor.cif import mae_loss

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class SemanticVADModel(FunASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
            self,
            vocab_size: int,
            token_list: Union[Tuple[str, ...], List[str]],
            frontend: Optional[AbsFrontend],
            specaug: Optional[AbsSpecAug],
            normalize: Optional[AbsNormalize],
            encoder: AbsEncoder,
            decoder: AbsDecoder,
            ctc: CTC,
            ctc_weight: float = 0.5,
            interctc_weight: float = 0.0,
            ignore_id: int = -1,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            report_cer: bool = True,
            report_wer: bool = True,
            sym_space: str = "<space>",
            sym_blank: str = "<blank>",
            extract_feats_in_collect_stats: bool = True,
            preencoder: Optional[AbsPreEncoder] = None,
            postencoder: Optional[AbsPostEncoder] = None,
            vad_output_size: int = 249,
            point_output_size: int = 3,
            point_weight: float = 0.2,
            asr_weight: float = 0.2

    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = 1
        self.eos = 2
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.error_calculator = None


        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        enc_dim = self.encoder.output_size()
        self.point_linear_layer=torch.nn.Linear(enc_dim, enc_dim)
        self.vad_linear_layer=torch.nn.Linear(enc_dim, enc_dim)

        self.leaky_relu=torch.nn.LeakyReLU(0.1)

        self.point_classifier = torch.nn.Linear(enc_dim, point_output_size)
        self.classifier = torch.nn.Linear(enc_dim, vad_output_size)
        
#        self.softmax = torch.nn.Softmax(dim=2)
        self.vad_output_size = vad_output_size
        self.point_output_size = point_output_size
        self.point_weight = point_weight
        self.asr_weight = asr_weight
        self.criterion_vad = LabelSmoothingLoss(
            size=vad_output_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.criterion_point = LabelSmoothingLoss(
            size=point_output_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )


    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            point_text: torch.Tensor,
            point_text_lengths: torch.Tensor,
            vad_text: torch.Tensor,
            vad_text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss + point loss + vad loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            point_text: (Batch, Length)
            point_text_lengths: (Batch, Length)
            vad_text: (Batch, Length)
            vad_text_lengths: (Batch, Length)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
                == point_text.shape[0]
                == vad_text.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape, point_text.shape, point_text_lengths.shape, vad_text.shape, vad_text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        point_text = point_text[:, : point_text_lengths.max()]
        vad_text = vad_text[:, : vad_text_lengths.max()]
        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                               1 - self.interctc_weight
                       ) * loss_ctc + self.interctc_weight * loss_interctc


        # 2b. Attention decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss_asr = loss_att
        elif self.ctc_weight == 1.0:
            loss_asr = loss_ctc
        else:
            loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        point_hid_out = self.point_linear_layer(encoder_out)
        point_out = self.leaky_relu(point_hid_out)
        point_out = self.point_classifier(point_out)
        # point_scores = self.softmax(point_out)
        #assert point_out.shape == point_text.shape, f"point_out shape {point_out.shape} != point_text shape {point_text.shape}"
        acc_point = th_accuracy(
            point_out.view(-1, self.point_output_size),
            point_text,
            ignore_label=self.ignore_id,
        )
        loss_point = self.criterion_point(point_out, point_text)
        stats["acc_point"] = acc_point
        stats["loss_point"] = loss_point

        vad_out = self.vad_linear_layer(encoder_out + point_hid_out)
        vad_out = self.leaky_relu(vad_out)
        vad_out = self.classifier(vad_out)
        #vad_scores = self.softmax(vad_out)
        acc_vad = th_accuracy(
            vad_out.view(-1, self.vad_output_size),
            vad_text,
            ignore_label=self.ignore_id,
        )
        # assert vad_out.shape == vad_text.shape, f"vad_out shape {vad_out.shape} != vad_text shape {vad_text.shape}"
        loss_vad = self.criterion_vad(vad_out, vad_text)
        stats["acc_vad"] = acc_vad
        stats["loss_vad"] = loss_vad

        loss = (1 - self.point_weight - self.asr_weight) * loss_vad + self.point_weight * loss_point + self.asr_weight * loss_asr
        # Collect Attn branch stats
        stats["loss_asr"] = loss_asr.detach() if loss_asr is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
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

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

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
        return feats, feats_lengths

    def nll(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder
        Normally, this function is called in batchify_nll.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
            batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder
        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc


class SemanticVADModelChunkOpt(FunASRModel):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    """

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        stride_conv = None,
        ctc_weight: float = 0.5,
        preencoder: Optional[AbsPreEncoder] = None,
        postencoder: Optional[AbsPostEncoder] = None,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        predictor=None,
        predictor_weight: float = 0.0,
        decoder_attention_chunk_type: str = 'chunk',
        vad_output_size: int = 2,
        point_output_size: int = 3,
        point_weight: float = 0.2,
        asr_weight: float = 0.2,
        enable_maas_finetune: bool = False
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        self.blank_id = 0
        self.sos = 1
        self.eos = 2
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.error_calculator = None

        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        self.step_cur = 0
        if self.encoder.overlap_chunk_cls is not None:
            from funasr.modules.streaming_utils.chunk_utilis import build_scama_mask_for_cross_attention_decoder
            self.build_scama_mask_for_cross_attention_decoder_fn = build_scama_mask_for_cross_attention_decoder
            self.decoder_attention_chunk_type = decoder_attention_chunk_type
        self.stride_conv =  stride_conv

        enc_dim = self.encoder.output_size()
        self.point_linear_layer=torch.nn.Linear(enc_dim, enc_dim)
        self.vad_linear_layer=torch.nn.Linear(enc_dim, enc_dim)

        self.leaky_relu=torch.nn.LeakyReLU(0.1)

        self.point_classifier = torch.nn.Linear(enc_dim, point_output_size)
        self.classifier = torch.nn.Linear(enc_dim, vad_output_size)

#       self.softmax = torch.nn.Softmax(dim=2)    
        self.vad_output_size = vad_output_size
        self.point_weight = point_weight
        self.asr_weight = asr_weight
        self.point_output_size = point_output_size
        self.criterion_vad = LabelSmoothingLoss(
            size=vad_output_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.criterion_point = LabelSmoothingLoss(
            size=point_output_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.enable_maas_finetune = enable_maas_finetune

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        point_text: torch.Tensor = None,
        point_text_lengths: torch.Tensor = None,
        vad_text: torch.Tensor = None,
        vad_text_lengths: torch.Tensor = None,
        decoding_ind: int = None,
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
        if point_text is not None:
            assert (
                    speech.shape[0]
                    == speech_lengths.shape[0]
                    == text.shape[0]
                    == text_lengths.shape[0]
                    == point_text.shape[0]
                    == vad_text.shape[0]
            ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape, point_text.shape, point_text_lengths.shape, vad_text.shape, vad_text_lengths.shape)
        else:
            assert (
                    speech.shape[0]
                    == speech_lengths.shape[0]
                    == text.shape[0]
                    == text_lengths.shape[0]
            ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        if self.stride_conv is not None:
            speech, speech_lengths = self.stride_conv(speech, speech_lengths)
            if point_text is not None:
                stride = self.stride_conv.stride_size()
                point_text = point_text[:, ::stride].contiguous()
                point_text_lengths = (point_text_lengths - 1 ) // stride + 1
                vad_text = vad_text[:, ::stride].contiguous()
                vad_text_lengths = (vad_text_lengths - 1) // stride + 1

        min_frame =  speech_lengths.max() if point_text is None else min(speech_lengths.max(), point_text_lengths.max(), vad_text_lengths.max())
        # for data-parallel
        # speech = speech[:, :speech_lengths.max()]
        speech = speech[:, : min_frame]
        text = text[:, : text_lengths.max()]
        if point_text is not None:
            # point_text = point_text[:, : point_text_lengths.max()]
            point_text = point_text[:, : min_frame].contiguous()
            # vad_text = vad_text[:, : vad_text_lengths.max()]
            vad_text = vad_text[:, : min_frame].contiguous()

        ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
        # 1. Encoder
        if self.enable_maas_finetune:
            with torch.no_grad():
                speech_raw, encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)
        else:
            speech_raw, encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()
        loss_pre = None
        loss_asr = 0.0

        encoder_out_ctc = None
        encoder_out_lens_ctc = None
        # 1. CTC branch
        if self.enable_maas_finetune:
            with torch.no_grad():
                if self.ctc_weight != 0.0:
                    if self.encoder.overlap_chunk_cls is not None:
                        encoder_out_ctc, encoder_out_lens_ctc = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out,
                                                                                                            encoder_out_lens,
                                                                                                            chunk_outs=None)
                    loss_ctc, cer_ctc = self._calc_ctc_loss(
                        encoder_out_ctc, encoder_out_lens_ctc, text, text_lengths
                    )

                    # Collect CTC branch stats
                    stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                    stats["cer_ctc"] = cer_ctc

                # Intermediate CTC (optional)
                loss_interctc = 0.0
                if self.interctc_weight != 0.0 and intermediate_outs is not None:
                    for layer_idx, intermediate_out in intermediate_outs:
                        # we assume intermediate_out has the same length & padding
                        # as those of encoder_out
                        if self.encoder.overlap_chunk_cls is not None:
                            encoder_out_ctc, encoder_out_lens_ctc = \
                                self.encoder.overlap_chunk_cls.remove_chunk(
                                    intermediate_out,
                                    encoder_out_lens,
                                    chunk_outs=None)
                        loss_ic, cer_ic = self._calc_ctc_loss(
                            encoder_out_ctc, encoder_out_lens_ctc, text, text_lengths
                        )
                        loss_interctc = loss_interctc + loss_ic

                        # Collect Intermedaite CTC stats
                        stats["loss_interctc_layer{}".format(layer_idx)] = (
                            loss_ic.detach() if loss_ic is not None else None
                        )
                        stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

                    loss_interctc = loss_interctc / len(intermediate_outs)

                    # calculate whole encoder loss
                    loss_ctc = (
                                1 - self.interctc_weight
                            ) * loss_ctc + self.interctc_weight * loss_interctc

                # 2b. Attention decoder branch
                if self.ctc_weight != 1.0:
                    loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_predictor_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )

                # 3. CTC-Att loss definition
                if self.ctc_weight == 0.0:
                    loss_asr = loss_att + loss_pre * self.predictor_weight
                elif self.ctc_weight == 1.0:
                    loss_asr = loss_ctc
                else:
                    loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight

                # Collect Attn branch stats
                stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                stats["acc"] = acc_att
                stats["cer"] = cer_att
                stats["wer"] = wer_att
                stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
        else:
            if self.encoder.overlap_chunk_cls is not None:
                encoder_out_ctc, encoder_out_lens_ctc = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out, encoder_out_lens, chunk_outs=None)
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out_ctc, encoder_out_lens_ctc, text, text_lengths
                )

                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc

                # Intermediate CTC (optional)
            loss_interctc = 0.0
            if self.interctc_weight != 0.0 and intermediate_outs is not None:
                for layer_idx, intermediate_out in intermediate_outs:
                    # we assume intermediate_out has the same length & padding
                    # as those of encoder_out
                    if self.encoder.overlap_chunk_cls is not None:
                        encoder_out_ctc, encoder_out_lens_ctc = \
                            self.encoder.overlap_chunk_cls.remove_chunk(
                                intermediate_out,
                                encoder_out_lens,
                                chunk_outs=None)
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        encoder_out_ctc, encoder_out_lens_ctc, text, text_lengths
                    )
                    loss_interctc = loss_interctc + loss_ic

                    # Collect Intermedaite CTC stats
                    stats["loss_interctc_layer{}".format(layer_idx)] = (
                        loss_ic.detach() if loss_ic is not None else None
                    )
                    stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

                loss_interctc = loss_interctc / len(intermediate_outs)

                # calculate whole encoder loss
                loss_ctc = (
                            1 - self.interctc_weight
                        ) * loss_ctc + self.interctc_weight * loss_interctc

            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_predictor_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss_asr = loss_att + loss_pre * self.predictor_weight
            elif self.ctc_weight == 1.0:
                loss_asr = loss_ctc
            else:
                loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att
            stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None

        if point_text is not None:
            point_hid_out = self.point_linear_layer(encoder_out_ctc)
            point_out = self.leaky_relu(point_hid_out)
            point_out = self.point_classifier(point_out)
            acc_point = th_accuracy(
                    point_out.view(-1, self.point_output_size),
                    point_text,
                    ignore_label=self.ignore_id,
                    )
            loss_point = self.criterion_point(point_out, point_text)
            stats["acc_point"] = acc_point
            stats["loss_point"] = loss_point
    
            vad_out = self.vad_linear_layer(encoder_out_ctc + point_hid_out)
            vad_out = self.leaky_relu(vad_out) 
            vad_out = self.classifier(vad_out)
            #vad_scores = self.softmax(vad_out)
            acc_vad = th_accuracy(
                    vad_out.view(-1, self.vad_output_size),
                    vad_text,
                    ignore_label=self.ignore_id,
                    )
            loss_vad = self.criterion_vad(vad_out, vad_text)
            stats["acc_vad"] = acc_vad
            stats["loss_vad"] = loss_vad


            loss = self.asr_weight * loss_asr + (1 - self.asr_weight - self.point_weight) * loss_vad + self.point_weight * loss_point
        else:
            loss = loss_asr
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
            featsengths = self._extract_feats(speech, speech_lengths)
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
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, ind: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
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
        speech_raw = feats.clone().to(feats.device)
        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc, ind=ind
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths, ind=ind)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return speech_raw, encoder_out, encoder_out_lens

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
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder
        Normally, this function is called in batchify_nll.
        Args:
                        encoder_out: (Batch, Length, Dim)
                        encoder_out_lens: (Batch,)
                        ys_pad: (Batch, Length)
                        ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder
        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
                        encoder_out: (Batch, Length, Dim)
                        encoder_out_lens: (Batch,)
                        ys_pad: (Batch, Length)
                        ys_pad_lens: (Batch,)
                        batch_size: int, samples each batch contain when computing nll,
                                                                        you may change this to avoid OOM or increase
                                                                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_att_predictor_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        encoder_out_mask = sequence_mask(encoder_out_lens, maxlen=encoder_out.size(1), dtype=encoder_out.dtype,
                                         device=encoder_out.device)[:, None, :]
        mask_chunk_predictor = None
        if self.encoder.overlap_chunk_cls is not None:
            mask_chunk_predictor = self.encoder.overlap_chunk_cls.get_mask_chunk_predictor(None,
                                                                                           device=encoder_out.device,
                                                                                           batch_size=encoder_out.size(
                                                                                               0))
            mask_shfit_chunk = self.encoder.overlap_chunk_cls.get_mask_shfit_chunk(None, device=encoder_out.device,
                                                                                   batch_size=encoder_out.size(0))
            encoder_out = encoder_out * mask_shfit_chunk
        pre_acoustic_embeds, pre_token_length, pre_alphas, _ = self.predictor(encoder_out,
                                                                              ys_out_pad,
                                                                              encoder_out_mask,
                                                                              ignore_id=self.ignore_id,
                                                                              mask_chunk_predictor=mask_chunk_predictor,
                                                                              target_label_length=ys_in_lens,
                                                                              )
        predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(pre_alphas,
                                                                                             encoder_out_lens)

        scama_mask = None
        if self.encoder.overlap_chunk_cls is not None and self.decoder_attention_chunk_type == 'chunk':
            encoder_chunk_size = self.encoder.overlap_chunk_cls.chunk_size_pad_shift_cur
            attention_chunk_center_bias = 0
            attention_chunk_size = encoder_chunk_size
            decoder_att_look_back_factor = self.encoder.overlap_chunk_cls.decoder_att_look_back_factor_cur
            mask_shift_att_chunk_decoder = self.encoder.overlap_chunk_cls.get_mask_shift_att_chunk_decoder(None,
                                                                                                           device=encoder_out.device,
                                                                                                           batch_size=encoder_out.size(
                                                                                                               0))
            scama_mask = self.build_scama_mask_for_cross_attention_decoder_fn(
                predictor_alignments=predictor_alignments,
                encoder_sequence_length=encoder_out_lens,
                chunk_size=1,
                encoder_chunk_size=encoder_chunk_size,
                attention_chunk_center_bias=attention_chunk_center_bias,
                attention_chunk_size=attention_chunk_size,
                attention_chunk_type=self.decoder_attention_chunk_type,
                step=None,
                predictor_mask_chunk_hopping=mask_chunk_predictor,
                decoder_att_look_back_factor=decoder_att_look_back_factor,
                mask_shift_att_chunk_decoder=mask_shift_att_chunk_decoder,
                target_length=ys_in_lens,
                is_training=self.training,
            )
        elif self.encoder.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out, encoder_out_lens,
                                                                                        chunk_outs=None)
        # try:
        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out,
            encoder_out_lens,
            ys_in_pad,
            ys_in_lens,
            chunk_mask=scama_mask,
            pre_acoustic_embeds=pre_acoustic_embeds,

        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        # predictor loss
        loss_pre = self.criterion_pre(ys_in_lens.type_as(pre_token_length), pre_token_length)
        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre

    def calc_predictor_mask(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor = None,
        ys_pad_lens: torch.Tensor = None,
    ):
        # ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # ys_in_lens = ys_pad_lens + 1
        ys_out_pad, ys_in_lens = None, None

        encoder_out_mask = sequence_mask(encoder_out_lens, maxlen=encoder_out.size(1), dtype=encoder_out.dtype,
                                         device=encoder_out.device)[:, None, :]
        mask_chunk_predictor = None
        if self.encoder.overlap_chunk_cls is not None:
            mask_chunk_predictor = self.encoder.overlap_chunk_cls.get_mask_chunk_predictor(None,
                                                                                           device=encoder_out.device,
                                                                                           batch_size=encoder_out.size(
                                                                                               0))
            mask_shfit_chunk = self.encoder.overlap_chunk_cls.get_mask_shfit_chunk(None, device=encoder_out.device,
                                                                                   batch_size=encoder_out.size(0))
            encoder_out = encoder_out * mask_shfit_chunk
        pre_acoustic_embeds, pre_token_length, pre_alphas, _ = self.predictor(encoder_out,
                                                                              ys_out_pad,
                                                                              encoder_out_mask,
                                                                              ignore_id=self.ignore_id,
                                                                              mask_chunk_predictor=mask_chunk_predictor,
                                                                              target_label_length=ys_in_lens,
                                                                              )
        predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(pre_alphas,
                                                                                             encoder_out_lens)

        scama_mask = None
        if self.encoder.overlap_chunk_cls is not None and self.decoder_attention_chunk_type == 'chunk':
            encoder_chunk_size = self.encoder.overlap_chunk_cls.chunk_size_pad_shift_cur
            attention_chunk_center_bias = 0
            attention_chunk_size = encoder_chunk_size
            decoder_att_look_back_factor = self.encoder.overlap_chunk_cls.decoder_att_look_back_factor_cur
            mask_shift_att_chunk_decoder = self.encoder.overlap_chunk_cls.get_mask_shift_att_chunk_decoder(None,
                                                                                                           device=encoder_out.device,
                                                                                                           batch_size=encoder_out.size(
                                                                                                               0))
            scama_mask = self.build_scama_mask_for_cross_attention_decoder_fn(
                predictor_alignments=predictor_alignments,
                encoder_sequence_length=encoder_out_lens,
                chunk_size=1,
                encoder_chunk_size=encoder_chunk_size,
                attention_chunk_center_bias=attention_chunk_center_bias,
                attention_chunk_size=attention_chunk_size,
                attention_chunk_type=self.decoder_attention_chunk_type,
                step=None,
                predictor_mask_chunk_hopping=mask_chunk_predictor,
                decoder_att_look_back_factor=decoder_att_look_back_factor,
                mask_shift_att_chunk_decoder=mask_shift_att_chunk_decoder,
                target_length=ys_in_lens,
                is_training=self.training,
            )
        elif self.encoder.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out, encoder_out_lens,
                                                                                        chunk_outs=None)

        return pre_acoustic_embeds, pre_token_length, predictor_alignments, predictor_alignments_len, scama_mask

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc
