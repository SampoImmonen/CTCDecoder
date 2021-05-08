"""
works only on linux with proper prerequisites

mainly https://github.com/parlance/ctcdecode
"""

import librosa
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import audio2numpy as an


test_dataset = load_dataset("common_voice", "fi", split="test[:2%]")

processor = Wav2Vec2Processor.from_pretrained("aapot/wav2vec2-large-xlsr-53-finnish")
model = Wav2Vec2ForCTC.from_pretrained("aapot/wav2vec2-large-xlsr-53-finnish")

resampler = lambda sr, y: librosa.resample(y.numpy().squeeze(), sr, 16_000)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = an.open_audio(batch["path"])
    speech_array = torch.tensor(speech_array)
    batch["speech"] = resampler(sampling_rate, speech_array).squeeze()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits


    
predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])

class SpeechRecognizer:
    
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        model.to(device)
        self.processor = processor
        self.device = device
        
    def _prepareaudio(self, path : str):
        # load audio
        # regex
        audio, sr = an.open_audio(path)
        try:
            audio = audio[:, 1]
        except:
            pass
        audio = torch.tensor(audio)
        audio = librosa.resample(audio.numpy().squeeze(), sr, 16_000)
        return audio
    
    def decode(self, output: torch.tensor, mode: str = "argmax"):
        
        if mode=="argmax":
            pred_ids = torch.argmax(output, dim=-1)
            
        return pred_ids
    
    
    @torch.no_grad()
    def __call__(self, path: str):
        audio = self._prepareaudio(path)
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        logits = model(inputs.input_values.to(self.device), attention_mask=inputs.attention_mask.to(self.device)).logits
        pred_ids = self.decode(logits)
        prediction = processor.batch_decode(pred_ids)
        return prediction, logits
    
recog = SpeechRecognizer(model, processor, device="cuda")

pred, logits = recog("data/testi.mp3")
logits = logits.squeeze(0)
#pred = torch.argmax(logits[:,:, :], dim=-1)
#pred = processor.batch_decode(pred)
print(pred)

vocab = processor.tokenizer.get_vocab()
print(vocab)

from itertools import groupby
import numpy as np
def invert_dict(dict):
    return {v: k for k, v in dict.items()}

dd = invert_dict(vocab)

vals = sorted(vocab.items(), key = lambda x:x[1])
labels = ([x[0] for x in vals[:-2]])
labels[10] = ' '
print(labels)

def map_to_chars(ids, vocab):
    result = [vocab[i] for i in ids]
    return "".join(result).replace('|', ' ')


if __name__ == '__main__':
    from ctcdecode import CTCBeamDecoder

    pred, logits = recog("data/3bfd7887-4a92-4528-b1c5-0cf94efbc6e3.mp3")

    probs = logits.softmax(dim=2).cpu()

    decoder = CTCBeamDecoder(
        labels,
        model_path="data/fi_3gram_lm.bin",
        alpha=2.3,
        beta=1,
        cutoff_top_n=15,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=30,
        log_probs_input=False
    )

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(probs)
    result = beam_results[0][0][:out_lens[0][0]]
    for i in range(20):
        print(map_to_chars(beam_results[0][i][:out_lens[0][i]].numpy(), dd))
    #print(map_to_chars(result.numpy(), dd))