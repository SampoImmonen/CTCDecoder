import librosa
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor 
import re

from SpeechRecognizer import SpeechRecognizer, CTCDecoder

#setup model

xlsl_aapo = "aapot/wav2vec2-large-xlsr-53-finnish"
voxpopuli1 = "data/voxpopuli-finetuned/"
voxpopuli_hugginface = "facebook/wav2vec2-base-10k-voxpopuli-ft-fi"

#Setup model and decoder
model = SpeechRecognizer(model_dir = voxpopuli1)

labels, blank = model.get_labels()
lm_path = "data/model2.bin"
decoder = CTCDecoder(labels, lm_path=lm_path, alpha=1.5, beta=0.8, blank_id=blank, beam_width=256, cutoff_top_n=15)


#aapo fair 5 gram
#0.9, 1.2, 24.0168
#1.2, 1.2  22.527
#1.8  1.2  22.75
#1.2  0.6  22.374


#voxpopuli1 fair 5 gram
#1.2 0.6 18.44
#1 0.6 18.9
#1.5 0.8 18.32
#1.5 0.8 18.02 (beam_width = 256, top_n = 15)


#voxpopuli netti 2ngam 1.5, 0.8 10.6


#voxpopuli model2.bin 0.
#1.5, 0.8, 8.89



def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(sampling_rate, speech_array).squeeze()
    return batch


def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

def custom_evaluate(batch):
    """
    evaluation function made for custom speechrecognizer class with separate decoder
    """
    #inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        pred, logits = model(batch['speech']) 
    probs = logits.softmax(dim=2).cpu()
    
    text = decoder.decode(probs)
    batch["pred_strings"] = [text]
    #print(text)
    return batch


if __name__ == "__main__":

    test_dataset = load_dataset("common_voice", "fi", split="test")
    wer = load_metric("wer")
    cer = load_metric("cer")

    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\...\…\–\é]'
    resampler = lambda sr, y: librosa.resample(y.numpy().squeeze(), sr, 16_000)

    test_dataset = test_dataset.map(speech_file_to_array_fn)
    result = test_dataset.map(custom_evaluate, batched=True, batch_size=1)

    #result = test_dataset.map(evaluate, batched=True, batch_size=1)
    print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))
    print("CER: {:2f}".format(100*cer.compute(predictions=result["pred_strings"], references=result["sentence"])))
