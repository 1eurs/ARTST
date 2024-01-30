import os
import torch
import numpy as np
import os.path as op
import pyarabic.araby as araby
from artst.tasks.artst import ArTSTTask
from transformers import SpeechT5HifiGan
from artst.models.artst import ArTSTTransformerModel
from fairseq.tasks.hubert_pretraining import LabelEncoder
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform 
import soundfile as sf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('ckpts/CLARTTS_ArTSTstar_TTS.pt')
checkpoint['cfg']['task'].t5_task = 't2s'
checkpoint['cfg']['task'].bpe_tokenizer = "utils/tts_spm.model"
checkpoint['cfg']['task'].data = "utils/"
checkpoint['cfg']['model'].mask_prob = 0.5
checkpoint['cfg']['task'].mask_prob = 0.5
task = ArTSTTask.setup_task(checkpoint['cfg']['task'])

emb_path='embs/clartts.npy'
model = ArTSTTransformerModel.build_model(checkpoint['cfg']['model'], task)
model.load_state_dict(checkpoint['model'])

checkpoint['cfg']['task'].bpe_tokenizer = task.build_bpe(checkpoint['cfg']['model'])
tokenizer = checkpoint['cfg']['task'].bpe_tokenizer

processor = LabelEncoder(task.dicts['text'])

vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(device)

def get_embs(emb_path):
    spkembs = get_features_or_waveform(emb_path)
    spkembs = torch.from_numpy(spkembs).float().unsqueeze(0)
    return spkembs

def process_text(text):
    text = araby.strip_diacritics(text)
    return processor(tokenizer.encode(text)).reshape(1, -1)

net_input = {}

def inference(text, spkr=emb_path):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))
    net_input['src_tokens'] = process_text(text)
    net_input['spkembs'] = get_embs(spkr)
    outs, _, attn = task.generate_speech(
            [model], 
            net_input,
        )
    with torch.no_grad():
        gen_audio = vocoder(outs.to(device))
    speech = (gen_audio.cpu().numpy() * 32767).astype(np.int16)
    return (16000,speech)