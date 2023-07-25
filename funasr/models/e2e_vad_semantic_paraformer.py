import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import random
import numpy as np
from typeguard import check_argument_types

from funasr.layers.abs_normalize import AbsNormalize
from funasr.losses.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.e2e_asr_common import ErrorCalculator
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.predictor.cif import mae_loss
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.modules.add_sos_eos import add_sos_eos
from funasr.modules.nets_utils import make_pad_mask, pad_list
from funasr.modules.nets_utils import th_accuracy
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.models.base_model import FunASRModel
from funasr.models.predictor.cif import CifPredictorV3

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class SemanticVADParaformer(FunASRModel):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
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
            ctc_weight: float = 0.5,
            interctc_weight: float = 0.0,
            ignore_id: int = -1,
            blank_id: int = 0,
            sos: int = 1,
            eos: int = 2,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            report_cer: bool = True,
            report_wer: bool = True,
            sym_space: str = "<space>",
            sym_blank: str = "<blank>",
            extract_feats_in_collect_stats: bool = True,
            predictor=None,
            predictor_weight: float = 0.0,
            predictor_bias: int = 0,
            sampling_ratio: float = 0.2,
            share_embedding: bool = False,
            preencoder: Optional[AbsPreEncoder] = None,
            postencoder: Optional[AbsPostEncoder] = None,
            use_1st_decoder_loss: bool = False,
            vad_output_size: int = 248,
            point_output_size: int = 3,
            point_weight: float = 0.2,
            asr_weight: float = 0.2
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = vocab_size - 1 if sos is None else sos
        self.eos = vocab_size - 1 if eos is None else eos
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
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        self.step_cur = 0

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None

        self.use_1st_decoder_loss = use_1st_decoder_loss

        enc_dim = self.encoder.output_size()
        self.point_linear_layer=torch.nn.Linear(enc_dim, enc_dim)
        self.vad_linear_layer=torch.nn.Linear(enc_dim, enc_dim)
        self.leaky_relu=torch.nn.LeakyReLU(0.1)

        self.point_classifier = torch.nn.Linear(enc_dim, point_output_size)
        self.classifier = torch.nn.Linear(enc_dim, vad_output_size)

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
        self.step_cur += 1
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
        min_frame =  speech_lengths.max() if point_text is None else min(speech_lengths.max(), point_text_lengths.max(), vad_text_lengths.max())
        # for data-parallel
        # speech = speech[:, :speech_lengths.max()]
        speech = speech[:, : min_frame].contiguous()
        text = text[:, : text_lengths.max()].contiguous()
        if point_text is not None:
            # point_text = point_text[:, : point_text_lengths.max()]
            point_text = point_text[:, : min_frame].contiguous()
            # vad_text = vad_text[:, : vad_text_lengths.max()]
            vad_text = vad_text[:, : min_frame].contiguous()
        # 1. Encoder
        if hasattr(self.encoder, "overlap_chunk_cls"):
            ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, pre_loss_att, acc_att, cer_att, wer_att = None, None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_pre = None
        stats = dict()
        loss_asr = 0.0

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
            loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss_asr = loss_att + loss_pre * self.predictor_weight
        elif self.ctc_weight == 1.0:
            loss_asr = loss_ctc
        else:
            loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight

        if self.use_1st_decoder_loss and pre_loss_att is not None:
            loss_asr = loss_asr + (1 - self.ctc_weight ) * pre_loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None

        loss = 0.0
        if point_text is not None:
            point_hid_out = self.point_linear_layer(encoder_out)
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

            vad_out = self.vad_linear_layer(encoder_out + point_hid_out)
            vad_out = self.leaky_relu(vad_out)
            vad_out = self.classifier(vad_out)
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

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            if hasattr(self.encoder, "overlap_chunk_cls"):
                encoder_out, encoder_out_lens, _ = self.encoder(
                    feats, feats_lengths, ctc=self.ctc, ind=ind
                )
                encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out,
                                                                                            encoder_out_lens,
                                                                                            chunk_outs=None)
            else:
                encoder_out, encoder_out_lens, _ = self.encoder(
                    feats, feats_lengths, ctc=self.ctc
                )
        else:
            if hasattr(self.encoder, "overlap_chunk_cls"):
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths, ind=ind)
                encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out,
                                                                                            encoder_out_lens,
                                                                                            chunk_outs=None)
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

    def calc_predictor(self, encoder_out, encoder_out_lens):

        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(encoder_out, None, encoder_out_mask,
                                                                                  ignore_id=self.ignore_id)
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index

    def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens):

        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
        )
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

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
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(encoder_out, ys_pad, encoder_out_mask,
                                                                                  ignore_id=self.ignore_id)

        # 0. sampler
        decoder_out_1st = None
        pre_loss_att = None
        if self.sampling_ratio > 0.0:
            if self.step_cur < 2:
                logging.info("enable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
            if self.use_1st_decoder_loss:
                sematic_embeds, decoder_out_1st, pre_loss_att = self.sampler_with_grad(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                                                               pre_acoustic_embeds)
            else:
                sematic_embeds, decoder_out_1st = self.sampler(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                                                               pre_acoustic_embeds)
        else:
            if self.step_cur < 2:
                logging.info("disable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
        )
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att

    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds):

        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        with torch.no_grad():
            decoder_outs = self.decoder(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens
            )
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def sampler_with_grad(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds):
        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens
        )
        pre_loss_att = self.criterion_att(decoder_outs[0], ys_pad)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]
        pred_tokens = decoder_out.argmax(-1)
        nonpad_positions = ys_pad.ne(self.ignore_id)
        seq_lens = (nonpad_positions).sum(1)
        same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
        input_mask = torch.ones_like(nonpad_positions)
        bsz, seq_len = ys_pad.size()
        for li in range(bsz):
            target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
            if target_num > 0:
                input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
        input_mask = input_mask.eq(1)
        input_mask = input_mask.masked_fill(~nonpad_positions, False)
        input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)

        return sematic_embeds * tgt_mask, decoder_out * tgt_mask, pre_loss_att

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


class SemanticVADParaformerOnline(SemanticVADParaformer):
    """
    Author: Speech Lab, Alibaba Group, China
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
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
            ctc_weight: float = 0.5,
            interctc_weight: float = 0.0,
            ignore_id: int = -1,
            blank_id: int = 0,
            sos: int = 1,
            eos: int = 2,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            report_cer: bool = True,
            report_wer: bool = True,
            sym_space: str = "<space>",
            sym_blank: str = "<blank>",
            extract_feats_in_collect_stats: bool = True,
            predictor=None,
            predictor_weight: float = 0.0,
            predictor_bias: int = 0,
            sampling_ratio: float = 0.2,
            decoder_attention_chunk_type: str = 'chunk',
            share_embedding: bool = False,
            preencoder: Optional[AbsPreEncoder] = None,
            postencoder: Optional[AbsPostEncoder] = None,
            use_1st_decoder_loss: bool = False,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            blank_id=blank_id,
            sos=sos,
            eos=eos,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            predictor=predictor,
            predictor_weight=predictor_weight,
            predictor_bias=predictor_bias,
            sampling_ratio=sampling_ratio,
        )
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = vocab_size - 1 if sos is None else sos
        self.eos = vocab_size - 1 if eos is None else eos
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
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        self.step_cur = 0
        self.scama_mask = None
        if hasattr(self.encoder, "overlap_chunk_cls") and self.encoder.overlap_chunk_cls is not None:
            from funasr.modules.streaming_utils.chunk_utilis import build_scama_mask_for_cross_attention_decoder
            self.build_scama_mask_for_cross_attention_decoder_fn = build_scama_mask_for_cross_attention_decoder
            self.decoder_attention_chunk_type = decoder_attention_chunk_type

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None

        self.use_1st_decoder_loss = use_1st_decoder_loss

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
        self.step_cur += 1
        # for data-parallel
        min_frame =  speech_lengths.max() if point_text is None else min(speech_lengths.max(), point_text_lengths.max(), vad_text_lengths.max())
        # speech = speech[:, :speech_lengths.max()]
        speech = speech[:, : min_frame].contiguous()
        text = text[:, : text_lengths.max()].contiguous()
        if point_text is not None:
            # point_text = point_text[:, : point_text_lengths.max()]
            point_text = point_text[:, : min_frame].contiguous()
            # vad_text = vad_text[:, : vad_text_lengths.max()]
            vad_text = vad_text[:, : min_frame].contiguous()

        # 1. Encoder
        if hasattr(self.encoder, "overlap_chunk_cls"):
            ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_pre = None
        stats = dict()
        loss_asr = 0.0
        # 1. CTC branch
        if self.ctc_weight != 0.0:
            if hasattr(self.encoder, "overlap_chunk_cls"):
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
                if hasattr(self.encoder, "overlap_chunk_cls"):
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
            loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att = self._calc_att_predictor_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att + loss_pre * self.predictor_weight
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight

        if self.use_1st_decoder_loss and pre_loss_att is not None:
            loss = loss + pre_loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None

        if point_text is not None:
            point_hid_out = self.point_linear_layer(encoder_out)
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

            vad_out = self.vad_linear_layer(encoder_out + point_hid_out)
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

        return encoder_out, encoder_out_lens

    def encode_chunk(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor, cache: dict = None
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
            encoder_out, encoder_out_lens, _ = self.encoder.forward_chunk(
                feats, feats_lengths, cache=cache["encoder"], ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder.forward_chunk(feats, feats_lengths, cache=cache["encoder"])
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, torch.tensor([encoder_out.size(1)])

    def _calc_att_predictor_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
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
                                                                              ys_pad,
                                                                              encoder_out_mask,
                                                                              ignore_id=self.ignore_id,
                                                                              mask_chunk_predictor=mask_chunk_predictor,
                                                                              target_label_length=ys_pad_lens,
                                                                              )
        predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(pre_alphas,
                                                                                             encoder_out_lens)

        scama_mask = None
        if self.encoder.overlap_chunk_cls is not None and self.decoder_attention_chunk_type == 'chunk':
            encoder_chunk_size = self.encoder.overlap_chunk_cls.chunk_size_pad_shift_cur
            attention_chunk_center_bias = 0
            attention_chunk_size = encoder_chunk_size
            decoder_att_look_back_factor = self.encoder.overlap_chunk_cls.decoder_att_look_back_factor_cur
            mask_shift_att_chunk_decoder = self.encoder.overlap_chunk_cls.\
                get_mask_shift_att_chunk_decoder(None,
                                                 device=encoder_out.device,
                                                 batch_size=encoder_out.size(0)
                                                 )
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
                target_length=ys_pad_lens,
                is_training=self.training,
            )
        elif self.encoder.overlap_chunk_cls is not None:
            encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out,
                                                                                        encoder_out_lens,
                                                                                        chunk_outs=None)
        # 0. sampler
        decoder_out_1st = None
        pre_loss_att = None
        if self.sampling_ratio > 0.0:
            if self.step_cur < 2:
                logging.info("enable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
            if self.use_1st_decoder_loss:
                sematic_embeds, decoder_out_1st, pre_loss_att = \
                    self.sampler_with_grad(encoder_out, encoder_out_lens, ys_pad,
                                           ys_pad_lens, pre_acoustic_embeds, scama_mask)
            else:
                sematic_embeds, decoder_out_1st = \
                    self.sampler(encoder_out, encoder_out_lens, ys_pad,
                                 ys_pad_lens, pre_acoustic_embeds, scama_mask)
        else:
            if self.step_cur < 2:
                logging.info("disable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, scama_mask
        )
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att

    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds, chunk_mask=None):

        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        with torch.no_grad():
            decoder_outs = self.decoder(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens, chunk_mask
            )
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def sampler_with_grad(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds, chunk_mask=None):
        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens, chunk_mask
        )
        pre_loss_att = self.criterion_att(decoder_outs[0], ys_pad)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]
        pred_tokens = decoder_out.argmax(-1)
        nonpad_positions = ys_pad.ne(self.ignore_id)
        seq_lens = (nonpad_positions).sum(1)
        same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
        input_mask = torch.ones_like(nonpad_positions)
        bsz, seq_len = ys_pad.size()
        for li in range(bsz):
            target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
            if target_num > 0:
                input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
        input_mask = input_mask.eq(1)
        input_mask = input_mask.masked_fill(~nonpad_positions, False)
        input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)

        return sematic_embeds * tgt_mask, decoder_out * tgt_mask, pre_loss_att

    def calc_predictor(self, encoder_out, encoder_out_lens):

        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        mask_chunk_predictor = None
        if self.encoder.overlap_chunk_cls is not None:
            mask_chunk_predictor = self.encoder.overlap_chunk_cls.get_mask_chunk_predictor(None,
                                                                                           device=encoder_out.device,
                                                                                           batch_size=encoder_out.size(
                                                                                               0))
            mask_shfit_chunk = self.encoder.overlap_chunk_cls.get_mask_shfit_chunk(None, device=encoder_out.device,
                                                                                   batch_size=encoder_out.size(0))
            encoder_out = encoder_out * mask_shfit_chunk
        pre_acoustic_embeds, pre_token_length, pre_alphas, pre_peak_index = self.predictor(encoder_out,
                                                                                           None,
                                                                                           encoder_out_mask,
                                                                                           ignore_id=self.ignore_id,
                                                                                           mask_chunk_predictor=mask_chunk_predictor,
                                                                                           target_label_length=None,
                                                                                           )
        predictor_alignments, predictor_alignments_len = self.predictor.gen_frame_alignments(pre_alphas,
                                                                                             encoder_out_lens+1 if self.predictor.tail_threshold > 0.0 else encoder_out_lens)

        scama_mask = None
        if self.encoder.overlap_chunk_cls is not None and self.decoder_attention_chunk_type == 'chunk':
            encoder_chunk_size = self.encoder.overlap_chunk_cls.chunk_size_pad_shift_cur
            attention_chunk_center_bias = 0
            attention_chunk_size = encoder_chunk_size
            decoder_att_look_back_factor = self.encoder.overlap_chunk_cls.decoder_att_look_back_factor_cur
            mask_shift_att_chunk_decoder = self.encoder.overlap_chunk_cls.\
                get_mask_shift_att_chunk_decoder(None,
                                                 device=encoder_out.device,
                                                 batch_size=encoder_out.size(0)
                                                 )
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
                target_length=None,
                is_training=self.training,
            )
        self.scama_mask = scama_mask

        return pre_acoustic_embeds, pre_token_length, pre_alphas, pre_peak_index

    def calc_predictor_chunk(self, encoder_out, cache=None):

        pre_acoustic_embeds, pre_token_length = \
            self.predictor.forward_chunk(encoder_out, cache["encoder"])
        return pre_acoustic_embeds, pre_token_length

    def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens):
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, self.scama_mask
        )
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def cal_decoder_with_predictor_chunk(self, encoder_out, sematic_embeds, cache=None):
        decoder_outs = self.decoder.forward_chunk(
            encoder_out, sematic_embeds, cache["decoder"]
        )
        decoder_out = decoder_outs
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out
