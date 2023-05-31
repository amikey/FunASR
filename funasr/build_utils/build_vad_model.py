import torch

from funasr.datasets.collate_fn import CommonCollateFn
from funasr.layers.abs_normalize import AbsNormalize
from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.models.e2e_vad import E2EVadModel
from funasr.models.fsmn_vad import FsmnVadModel
from funasr.models.encoder.fsmn_encoder import FSMN
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend, WavFrontendOnline
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.specaug.specaug import SpecAug
from funasr.models.specaug.specaug import SpecAugLFR
from funasr.tasks.abs_task import AbsTask
from funasr.train.class_choices import ClassChoices
from funasr.train.trainer import Trainer
from funasr.utils.types import float_or_none
from funasr.utils.types import int_or_none
from funasr.utils.types import str_or_none
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
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        fsmn=FSMN,
    ),
    type_check=torch.nn.Module,
    default="fsmn",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        e2evad=E2EVadModel,
        fsmnvad=FsmnVadModel,
    ),
    default="e2evad",
)

class_choices_list = [
    # --frontend and --frontend_conf
    frontend_choices,
    # --specaug and --specaug_conf
    specaug_choices,
    # --normalize and --normalize_conf
    normalize_choices,
    # --encoder and --encoder_conf
    encoder_choices,
    # --model and --model_conf
    model_choices,
]


def build_vad_model(args):
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
        feature_transform = args.feature_transform

    if args.specaug is not None:
        specaug_class = specaug_choices.get_class(args.specaug)
        specaug = specaug_class(**args.specaug_conf)
    else:
        specaug = None

    if args.normalize is not None:
        normalize_class = normalize_choices.get_class(args.normalize)
        normalize = normalize_class(**args.normalize_conf)
    else:
        normalize = None
    # encoder
    encoder_class = encoder_choices.get_class(args.encoder)
    encoder = encoder_class(**args.encoder_conf)

    model_class = model_choices.get_class(args.model)
    if encoder.training:
        model = model_class(frontend=frontend, specaug=specaug, normalize=normalize, encoder=encoder, feature_transform=feature_transform)
    else:
        model = model_class(encoder=encoder, vad_post_args=args.vad_post_conf, frontend=frontend)

    # initialize
    if args.init is not None:
        initialize(model, args.init)

    return model
