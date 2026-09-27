import os
import string
import sys

import scipy
import soundfile as sf
import torch
import zhconv
from funasr import AutoModel
from jiwer import compute_measures
from tqdm import tqdm
from transformers import Wav2Vec2Processor, HubertForCTC
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from zhon.hanzi import punctuation


punctuation_all = punctuation + string.punctuation + '-'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EN_WHISPER_MODEL = os.environ.get(
    "EN_ASR_MODEL_ID",
    os.path.join(PROJECT_ROOT, "ckpt", "whisper-large-v3"),
)
EN_HUBERT_MODEL = os.environ.get(
    "EN_HUBERT_MODEL_ID",
    os.path.join(PROJECT_ROOT, "hubert-large-ls960-ft"),
)
ZH_ASR_MODEL = os.environ.get(
    "ZH_ASR_MODEL",
    os.path.join(PROJECT_ROOT, "ckpt", "funasr", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
)
ZH_ASR_VAD_MODEL = os.environ.get(
    "ZH_ASR_VAD_MODEL",
    os.path.join(PROJECT_ROOT, "ckpt", "funasr", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
)
ZH_ASR_PUNC_MODEL = os.environ.get(
    "ZH_ASR_PUNC_MODEL",
    os.path.join(PROJECT_ROOT, "ckpt", "funasr", "punc_ct-transformer_cn-en-common-vocab471067-large"),
)

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3]  # zh 或 en
device = os.environ.get("ASR_DEVICE", "cuda:0")

def load_en_model_hubert():
    dev = torch.device(device)
    processor = Wav2Vec2Processor.from_pretrained(EN_HUBERT_MODEL)
    model = HubertForCTC.from_pretrained(EN_HUBERT_MODEL).to(dev)
    return processor, model

def load_en_model():
    dev = torch.device(device)
    model_id = EN_WHISPER_MODEL
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(dev)
    print(f'模型加载完毕')
    return processor, model

def load_zh_model():

    model = AutoModel(
        model=ZH_ASR_MODEL,
        vad_model=ZH_ASR_VAD_MODEL,
        punc_model=ZH_ASR_PUNC_MODEL,
        model_hub="ms",
        disable_pbar=True,
        device=device
    )

    return model

def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    # 评测前去掉标点，中文按字切分，英文按词切分。
    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')
    truth = truth.replace('-', ' ')
    hypo = hypo.replace('-', ' ')
    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
        raw_truth = raw_truth.replace('.', '。')
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, res_path):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    # import pdb;pdb.set_trace()
    if lang == "en":
        processor, model = load_en_model()
    elif lang == "zh":
        model = load_zh_model()
    else:
        raise NotImplementedError(f"Unsupported lang: {lang}")

    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        if len(line.split('|')) == 2:
            wav_res_path, text_ref = line.split('|')
        elif len(line.split('|')) == 3:
            wav_res_path, wav_ref_path, text_ref = line.split('|')
        elif len(line.split('|')) == 4:
            wav_res_path, _, text_ref, wav_ref_path = line.split('|')
        else:
            raise NotImplementedError

        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, text_ref))
    fout = open(res_path, "w")
    f_gen = open(res_path+'_gen', "w")
    f_gt = open(res_path+'_gt', "w")

    for wav_res_path, text_ref in tqdm(params):
        if lang == "en":
            wav, sr = sf.read(wav_res_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif lang == "zh":
            try:
                res = model.generate(input=wav_res_path,
                        batch_size_s=300)
                transcription = res[0]["text"]
                transcription = zhconv.convert(transcription, 'zh-cn')
            except:
                transcription = ""
                print("空音频：",wav_res_path)
                
        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref)
        fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
        f_gen.write(f"{wav_res_path}\t{raw_hypo}\n")
        f_gt.write(f"{wav_res_path}\t{raw_truth}\n")
        fout.flush()


run_asr(wav_res_text_path, res_path)
