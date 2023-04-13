import onnxruntime
import numpy as np

def test_encoder(onnx_path):
    opts = onnxruntime.SessionOptions()
    providers = ['CPUExecutionProvider']
    sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = [nd.name for nd in sess.get_inputs()]
    output_name = [nd.name for nd in sess.get_outputs()]
    
    def _get_feed_dict(feats_length):
        return {'speech': np.random.rand(1, feats_length, 560).astype(np.float32),
                'speech_lengths': np.array([feats_length, ], dtype=np.int32),
                }
    
    def _run(feed_dict):
        output = sess.run(output_name, input_feed=feed_dict)
        for name, value in zip(output_name, output):
            print('{}: {}'.format(name, value.shape))
    
    _run(_get_feed_dict(100))
    _run(_get_feed_dict(200))


def test_decoder(onnx_path):
    opts = onnxruntime.SessionOptions()
    providers = ['CPUExecutionProvider']
    sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = [nd.name for nd in sess.get_inputs()]
    output_name = [nd.name for nd in sess.get_outputs()]
    
    def _get_feed_dict(feats_length=100):
        enc = np.random.rand(1, feats_length, 320).astype(np.float32)
        enc_len = np.array([feats_length, ], dtype=np.int32)
        acoustic_embeds = np.random.rand(1, feats_length//2, 320).astype(np.float32)
        acoustic_embeds_len = np.array([feats_length//2, ], dtype=np.int32)
        cache_num = 12
        cache = [
            np.zeros((1, 320, 10)).astype(np.float32)
            for _ in range(cache_num)
        ]
        ret = {'enc': enc,
                'enc_len': enc_len,
                'acoustic_embeds': acoustic_embeds,
                'acoustic_embeds_len': acoustic_embeds_len
                }
        ret.update(
            {
                'in_cache_%d' % d: cache[d]
                for d in range(cache_num)
            }
        )
        return ret
    
    def _run(feed_dict):
        output = sess.run(output_name, input_feed=feed_dict)
        for name, value in zip(output_name, output):
            print('{}: {}'.format(name, value.shape))
    
    _run(_get_feed_dict(100))
    _run(_get_feed_dict(200))

if __name__ == '__main__':
    onnx_path = "export/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/encoder.onnx"
    test_encoder(onnx_path)
    
    onnx_path = "export/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/decoder.onnx"
    test_decoder(onnx_path)
    