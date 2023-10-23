[//]: # (<div align="left"><img src="docs/images/funasr_logo.jpg" width="400"/></div>)

(简体中文|[English](./README.md))

# FunASR: A Fundamental End-to-End Speech Recognition Toolkit
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

FunASR希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过发布工业级语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的研究和生产，并推动语音识别生态的发展。让语音识别更有趣！

<div align="center">  
<h4>
 <a href="#核心功能"> 核心功能 </a>   
｜<a href="#最新动态"> 最新动态 </a>
｜<a href="#安装教程"> 安装 </a>
｜<a href="#快速开始"> 快速开始 </a>
｜<a href="https://alibaba-damo-academy.github.io/FunASR/en/index.html"> 教程文档 </a>
｜<a href="./docs/model_zoo/modelscope_models.md"> 模型仓库 </a>
｜<a href="./funasr/runtime/readme_cn.md"> 服务部署 </a>
｜<a href="#联系我们"> 联系我们 </a>
</h4>
</div>

<a name="核心功能"></a>
## 核心功能
- FunASR是一个基础语音识别工具包，提供多种功能，包括语音识别（ASR）、语音端点检测（VAD）、标点恢复、语言模型、说话人验证、说话人分离和多人对话语音识别等。FunASR提供了便捷的脚本和教程，支持预训练好的模型的推理与微调。
- 我们在[ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition)与[huggingface](https://huggingface.co/FunAudio)上发布了大量开源数据集或者海量工业数据训练的模型，可以通过我们的[模型仓库](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)了解模型的详细信息。代表性的[Paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)非自回归端到端语音识别模型具有高精度、高效率、便捷部署的优点，支持快速构建语音识别服务，详细信息可以阅读([服务部署文档](funasr/runtime/readme_cn.md))。

<a name="最新动态"></a>
## 最新动态
- 20223/10/17: 英文离线文件转写服务一键部署的CPU版本发布，详细信息参阅([一键部署文档](funasr/runtime/docs/SDK_tutorial_en_zh.md))
- 2023/10/13: [SlideSpeech](https://slidespeech.github.io/): 一个大规模的多模态音视频语料库，主要是在线会议或者在线课程场景，包含了大量与发言人讲话实时同步的幻灯片。
- 2023.10.10: [Paraformer-long-Spk](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr_vad_spk/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/demo.py)模型发布，支持在长语音识别的基础上获取每句话的说话人标签。
- 2023.10.07: [FunCodec](https://github.com/alibaba-damo-academy/FunCodec): FunCodec提供开源模型和训练工具，可以用于音频离散编码，以及基于离散编码的语音识别、语音合成等任务。
- 2023.09.01: 中文离线文件转写服务2.0 CPU版本发布，新增ffmpeg、时间戳与热词模型支持，详细信息参阅([一键部署文档](funasr/runtime/docs/SDK_tutorial_zh.md))
- 2023.08.07: 中文实时语音听写服务一键部署的CPU版本发布，详细信息参阅([一键部署文档](funasr/runtime/docs/SDK_tutorial_online_zh.md))
- 2023.07.17: BAT一种低延迟低内存消耗的RNN-T模型发布，详细信息参阅（[BAT](egs/aishell/bat)）
- 2023.06.26: ASRU2023 多通道多方会议转录挑战赛2.0完成竞赛结果公布，详细信息参阅（[M2MeT2.0](https://alibaba-damo-academy.github.io/FunASR/m2met2_cn/index.html)）

<a name="安装教程"></a>
## 安装教程
FunASR安装教程请阅读（[Installation](https://alibaba-damo-academy.github.io/FunASR/en/installation/installation.html)）

## 模型仓库

FunASR开源了大量在工业数据上预训练模型，您可以在[模型许可协议](./MODEL_LICENSE)下自由使用、复制、修改和分享FunASR模型，下面列举代表性的模型，更多模型请参考[模型仓库]()


|                                                                                                         模型名字                                                                                                         |        任务详情        |     训练数据     | 参数量  |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------:|:------------:|:----:|
| speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch ([🤗]() [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) ) |  语音识别，带时间戳输出，非实时   |  60000小时，中文  | 220M |
|                            speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn ([🤗]() [⭐](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary) )                            | 分角色语音识别，带时间戳输出，非实时 |  60000小时，中文  | 220M |
|           speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020 ([🤗]() [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary) )           |      语音识别，非实时      |  50000小时，英文  | 220M |
|                                 speech_conformer_asr-en-16k-vocab4199-pytorch ([🤗]() [⭐](https://modelscope.cn/models/damo/speech_conformer_asr-en-16k-vocab4199-pytorch/summary) )                                 |      语音识别，非实时      |  50000小时，英文  | 220M |
|      speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online ([🤗]() [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) )                    |      语音识别，实时       |  60000小时，中文  | 220M | 
|      punc_ct-transformer_cn-en-common-vocab471067-large ([🤗]() [⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary) )                    |      标点恢复，非实时      |  100M，中文与英文  | 1.1G | 
|      speech_fsmn_vad_zh-cn-8k-common ([🤗]() [⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-8k-common/summary) )                    |     语音端点检测，实时      | 5000小时，中文与英文 | 0.4M | 
|      speech_timestamp_prediction-v1-16k-offline ([🤗]() [⭐](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary) )                    |      字级别时间戳预测      |  50000小时，中文  | 38M  | 





<a name="快速开始"></a>
## 快速开始
FunASR支持数万小时工业数据训练的模型的推理和微调，详细信息可以参阅（[modelscope_egs](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/quick_start.html)）；也支持学术标准数据集模型的训练和微调，详细信息可以参阅（[egs](https://alibaba-damo-academy.github.io/FunASR/en/academic_recipe/asr_recipe.html)）。

下面为快速上手教程
### step.1 加载头定义和下载音频文件
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 中文测试音频
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav
# 英文测试音频
wget 
```
### step.2 定义推理pipeline

```python
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)
```
其中，
- `task`参数为任务类型，支持有：语音识别（`Tasks.auto_speech_recognition`），标点恢复（`Tasks.punctuation`），语音端点检测（`Tasks.voice_activity_detection`），时间戳预测（`Tasks.speech_timestamp`）。
- `model`参数为具体模型名，与`task`参数配合使用，支持[模型仓库]中任意模型。

### step.3 推理音频
分为非实时模型与实时模型，非实时模型输入为整句音频/文本，实时模型输入为音频流或者片段

#### 非实时模型
```python
rec_result = inference_pipeline(audio_in='./vad_example.wav') #语音输入
# rec_result = inference_pipeline(text_in='我们都是木头人不会讲话不会动') #文本输入
```

#### 实时模型
```python
import soundfile
speech, sample_rate = soundfile.read("./vad_example.wav")

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention
param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size,
              "encoder_chunk_look_back": encoder_chunk_look_back, "decoder_chunk_look_back": decoder_chunk_look_back}
chunk_stride = chunk_size[1] * 960 # 600ms、480ms
# first chunk, 600ms
speech_chunk = speech[0:chunk_stride]
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
# next chunk, 600ms
speech_chunk = speech[chunk_stride:chunk_stride+chunk_stride]
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
```

更多详细用法（[新人文档](https://alibaba-damo-academy.github.io/FunASR/en/funasr/quick_start_zh.html)）


<a name="服务部署"></a>
## 服务部署
FunASR支持预训练或者进一步微调的模型进行服务部署。目前支持以下几种服务部署：

- 中文离线文件转写服务（CPU版本），已完成
- 中文流式语音识别服务（CPU版本），已完成
- 英文离线文件转写服务（CPU版本），已完成
- 中文离线文件转写服务（GPU版本），进行中
- 更多支持中

详细信息可以参阅([服务部署文档](funasr/runtime/readme_cn.md))。


<a name="社区交流"></a>
## 联系我们

如果您在使用中遇到问题，可以直接在github页面提Issues。欢迎语音兴趣爱好者扫描以下的钉钉群或者微信群二维码加入社区群，进行交流和讨论。

|                                  钉钉群                                  |                          微信                           |
|:---------------------------------------------------------------------:|:-----------------------------------------------------:|
| <div align="left"><img src="docs/images/dingding.jpg" width="250"/>   | <img src="docs/images/wechat.png" width="215"/></div> |

## 社区贡献者

| <div align="left"><img src="docs/images/nwpu.png" width="260"/> | <img src="docs/images/China_Telecom.png" width="200"/> </div>  | <img src="docs/images/RapidAI.png" width="200"/> </div> | <img src="docs/images/aihealthx.png" width="200"/> </div> | <img src="docs/images/XVERSE.png" width="250"/> </div> |
|:---------------------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------------------:|:------------------------------------------------------:|

贡献者名单请参考（[致谢名单](./Acknowledge.md)）


## 许可协议
项目遵循[The MIT License](https://opensource.org/licenses/MIT)开源协议，模型许可协议请参考（[模型协议](./MODEL_LICENSE)）


## 论文引用

``` bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and Zerui Li and Jiaming Wang and Haoneng Luo and Xian Shi and Mengzhe Chen and Yabin Li and Lingyun Zuo and Zhihao Du and Zhangyu Xiao and Shiliang Zhang},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  year={2023},
  booktitle={INTERSPEECH},
}
@inproceedings{An2023bat,
  author={Keyu An and Xian Shi and Shiliang Zhang},
  title={BAT: Boundary aware transducer for memory-efficient and low-latency ASR},
  year={2023},
  booktitle={INTERSPEECH},
}
@inproceedings{gao22b_interspeech,
  author={Zhifu Gao and ShiLiang Zhang and Ian McLoughlin and Zhijie Yan},
  title={{Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={2063--2067},
  doi={10.21437/Interspeech.2022-9996}
}
```
