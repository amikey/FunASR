import librosa
from funasr_onnx.asr_model import Speech2Text

wav_file = "/mnt/nas_sg/mit_sg/ni.chongjia/PAI_ASR_Training/20230801FunASR_onnx/test.wav"
y, sr = librosa.load(wav_file, sr=16000)
model_dir = "/mnt/nas_sg/mit_sg/ni.chongjia/PAI_ASR_Training/20230801FunASR_onnx/english_conformer_model_onnx"
speech2text = Speech2Text(model_dir=model_dir)
nbest = speech2text(y)
print(nbest[0][0])

