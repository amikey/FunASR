from funasr_torch import Paraformer_dashscope as Paraformer


model_dir = "/home/haoneng.lhn/FunASR/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

model = Paraformer(model_dir, batch_size=1)  # cpu
#model = Paraformer(model_dir, batch_size=1, device_id=0)  # gpu

wav_path = "/home/haoneng.lhn/FunASR/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"

result = model(wav_path)
print(result)
