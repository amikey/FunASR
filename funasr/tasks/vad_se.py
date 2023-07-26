import argparse
import logging
import os
from pathlib import Path
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from funasr.models.base_model import FunASRModel

import numpy as np
import torch
import yaml
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.datasets.collate_fn import CommonCollateFn
from funasr.datasets.preprocessor import CommonPreprocessor
from funasr.layers.abs_normalize import AbsNormalize
from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.models.ctc import CTC
from funasr.models.e2e_vad import E2EVadModel
from funasr.models.fsmn_vad import FsmnVadModel
from funasr.models.e2e_vad_semantic import SemanticVADModel, SemanticVADModelChunkOpt
from funasr.models.e2e_vad_semantic_paraformer import SemanticVADParaformer
from funasr.models.encoder.fsmn_encoder import FSMN, DFSMN
from funasr.models.encoder.conformer_encoder import ConformerEncoder, ConformerChunkEncoder
from funasr.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.decoder.sanm_decoder import FsmnDecoderSCAMAOpt, ParaformerSANMDecoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend, WavFrontendOnline
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.specaug.specaug import SpecAug
from funasr.models.specaug.specaug import SpecAugLFR
from funasr.modules.subsampling import Conv1dSubsampling
from funasr.models.predictor.cif import CifPredictor, CifPredictorV2, CifPredictorV3
from funasr.tasks.abs_task import AbsTask
from funasr.text.phoneme_tokenizer import g2p_choices
from funasr.torch_utils.initialize import initialize
from funasr.train.class_choices import ClassChoices
from funasr.train.trainer import Trainer
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.types import float_or_none
from funasr.utils.types import int_or_none
from funasr.utils.types import str_or_none
from funasr.utils.types import str2bool
from funasr.utils.nested_dict_action import NestedDictAction

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        wav_frontend=WavFrontend,
        wav_frontend_online=WavFrontendOnline,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
        specaug_lfr=SpecAugLFR,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        e2evad=E2EVadModel,
        fsmnvad=FsmnVadModel,
        semanticvad=SemanticVADModel,
        semanticvad_chunkopt=SemanticVADModelChunkOpt,
        semanticvad_paraformer=SemanticVADParaformer
    ),
    type_check=object,
    default="semanticvad_chunkopt",
)

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        fsmn=FSMN,
        dfsmn=DFSMN,
        conformer=ConformerEncoder,
        sanm=SANMEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        chunk_conformer=ConformerChunkEncoder,
    ),
    type_check=torch.nn.Module,
    default="dfsmn",
)

decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        fsmn_scama_opt=FsmnDecoderSCAMAOpt,
        paraformer_decoder_sanm=ParaformerSANMDecoder,
    ),
    default="fsmn_scama_opt",
)

predictor_choices = ClassChoices(
    name="predictor",
    classes=dict(
        cif_predictor=CifPredictor,
        ctc_predictor=None,
        cif_predictor_v2=CifPredictorV2,
        cif_predictor_v3=CifPredictorV3,
    ),
    default="cif_predictor",
    optional=True,
)

stride_conv_choices = ClassChoices(
    name="stride_conv",
    classes=dict(
        stride_conv1d=Conv1dSubsampling
    ),
    default="stride_conv1d",
    optional=True,
)

class VADTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --model and --model_conf
        model_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --predictor and --predictor_conf
        predictor_choices,
        # --stride_conv and --stride_conv_conf
        stride_conv_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")
        # required += ["token_list"]
        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--split_with_space",
            type=str2bool,
            default=True,
            help="whether to split text using <space>",
        )
        group.add_argument(
            "--seg_dict_file",
            type=str,
            default=None,
            help="seg_dict_file for text processing",
        )
        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--joint_network_conf",
            action=NestedDictAction,
            default=None,
            help="The keyword arguments for joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            default=None,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group = parser.add_argument_group(description="Preprocess related")
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--cmvn_file",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        parser.add_argument(
            "--feature_transform",
            action=NestedDictAction,
            default=dict(),
            help=f"The feature transform for original feature, include cmvn and lfr",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols if hasattr(args, "non_linguistic_symbols") else None,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                split_with_space=args.split_with_space if hasattr(args, "split_with_space") else False,
                seg_dict_file=args.seg_dict_file if hasattr(args, "seg_dict_file") else None,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            #retval = ("speech", "text", "point_text", "vad_text")
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        #retval = ()
        retval = ("point_text", "vad_text")
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace):
        assert check_argument_types()
        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            if args.frontend == 'wav_frontend':
                frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
            else:
                frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size
            feature_transform = args.feature_transform

        if  args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None
        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(**args.encoder_conf)

        # 5. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("e2evad")
        if args.model == "e2evad":
            encoder.eval()


        if encoder.training:
            model = model_class(frontend=frontend, specaug=specaug, normalize=normalize, encoder=encoder, feature_transform=feature_transform)
        else:
            model = model_class(encoder=encoder, vad_post_args=args.vad_post_conf, frontend=frontend)

        return model

    @classmethod
    def build_vad_semantic_model(cls, args):
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")

        # frontend
        if args.input_size is None:
            frontend_class = frontend_choices.get_class(args.frontend)
            if args.frontend == 'wav_frontend':
                frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
            else:
                frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # decoder
        decoder_class = decoder_choices.get_class(args.decoder)
        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder.output_size(),
            **args.decoder_conf,
        )

        # ctc
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder.output_size(), **args.ctc_conf
        )

        if args.model in ["semanticvad"]:
            model_class = model_choices.get_class(args.model)
            model = model_class(
                vocab_size=vocab_size,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                token_list=token_list,
                **args.model_conf,
            )
        elif args.model in ["semanticvad_chunkopt"]:
            stride_conv_class = stride_conv_choices.get_class(args.stride_conv)
            stride_conv = stride_conv_class(**args.stride_conv_conf, idim=input_size, odim=input_size)
            predictor_class = predictor_choices.get_class(args.predictor)
            predictor = predictor_class(**args.predictor_conf)
            model_class = model_choices.get_class(args.model)
            model = model_class(
                vocab_size=vocab_size,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                encoder=encoder,
                decoder=decoder,
                stride_conv=stride_conv,
                ctc=ctc,
                token_list=token_list,
                predictor=predictor,
                **args.model_conf,
            )
        elif args.model in ["semanticvad_paraformer"]:
            #stride_conv_class = stride_conv_choices.get_class(args.stride_conv)
            #stride_conv = stride_conv_class(**args.stride_conv_conf, idim=input_size, odim=input_size)
            predictor_class = predictor_choices.get_class(args.predictor)
            predictor = predictor_class(**args.predictor_conf)
            model_class = model_choices.get_class(args.model)
            model = model_class(
                vocab_size=vocab_size,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                token_list=token_list,
                predictor=predictor,
                **args.model_conf,
            )
        else:
            raise NotImplementedError("Not supported model: {}".format(args.model))

        # initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
            cls,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            device: str = "cpu",
            cmvn_file: Union[Path, str] = None,
    ):
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        # if cmvn_file is not None:
        args["cmvn_file"] = cmvn_file
        if "input_size" not in args:
            args["input_size"] = None
        if "frontend" not in args:
            args["frontend"] = None
            args["frontend_conf"] = None
        if "specaug" not in args:
            args["specaug"] = None 
        if "normalize" not in args:
            args["normalize"] = None
        if "feature_transform" not in args:
            args["feature_transform"] = None

        args = argparse.Namespace(**args)
        model = None
        if args.model in ["fsmnvad"]:
            model = cls.build_model(args)
        else:
            model = cls.build_vad_semantic_model(args)
        model.to(device)
        model_dict = dict()
        model_name_pth = None
        if model_file is not None:
            logging.info("model_file is {}".format(model_file))
            if device == "cuda":
                device = f"cuda:{torch.cuda.current_device()}"
            model_dir = os.path.dirname(model_file)
            model_name = os.path.basename(model_file)
            model_dict = torch.load(model_file, map_location=device)
        #model.encoder.load_state_dict(model_dict)
        model.load_state_dict(model_dict)

        return model, args
