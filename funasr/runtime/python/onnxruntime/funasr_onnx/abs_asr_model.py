import logging
import os

from funasr_onnx.beam_search.batch_beam_search import BatchBeamSearch
from funasr_onnx.beam_search.beam_search import BeamSearch
from funasr_onnx.beam_search.beam_search_transducer import \
    BeamSearchTransducer
from funasr_onnx.model.decoder import get_decoder
from funasr_onnx.model.encoder import get_encoder
from funasr_onnx.model.joint_network import JointNetwork
from funasr_onnx.model.lm import get_lm
from funasr_onnx.scorer.ctc_prefix_scorer import CTCPrefixScorer
from funasr_onnx.scorer.interface import BatchScorerInterface
from funasr_onnx.scorer.length_bonus import LengthBonus
from funasr_onnx.utils.abs_model import AbsModel

class AbsASRModel(AbsModel):
    def _check_flags(self, use_quantized):
        if use_quantized and "quantized_model_path" not in self.config.encoder.keys():
            # check if quantized model config is defined.
            raise RuntimeError("Configuration for quantized model is not defined.")

    def _build_beam_search(self, scorers, weights):
        if self.config.transducer.use_transducer_decoder:
            self.beam_search = BeamSearchTransducer(
                self.config.beam_search,
                self.config.token,
                scorers=scorers,
                weights=weights,
            )
        else:
            self.beam_search = BeamSearch(
                self.config.beam_search,
                self.config.token,
                scorers=scorers,
                weights=weights,
            )
            non_batch = [
                k
                for k, v in self.beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                print("here is the batch beam search is selected.")
                self.beam_search.__class__ = BatchBeamSearch
                logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )

    def _build_model(self, providers, use_quantized):

        #print("self.config.encoder {}".format(self.config.encoder))

        self.encoder = get_encoder(self.config.encoder, providers, use_quantized)
        decoder = get_decoder(self.config.decoder, providers, use_quantized)
        scorers = {"decoder": decoder}
        weights = {}
        if not self.config.transducer.use_transducer_decoder:
            ctc = CTCPrefixScorer(
                self.config.ctc, self.config.token.eos, providers, use_quantized
            )
            scorers.update(
                ctc=ctc, length_bonus=LengthBonus(len(self.config.token.list))
            )
            weights.update(
                decoder=self.config.weights.decoder,
                ctc=self.config.weights.ctc,
                length_bonus=self.config.weights.length_bonus,
            )
        else:
            joint_network = JointNetwork(
                self.config.joint_network, providers, use_quantized
            )
            scorers.update(joint_network=joint_network)

        lm = get_lm(self.config, providers, use_quantized)
        if lm is not None:
            scorers.update(lm=lm)
            weights.update(lm=self.config.weights.lm)

        self._build_beam_search(scorers, weights)
        self._build_tokenizer()
        self._build_token_converter()
        self.scorers = scorers
        self.weights = weights
