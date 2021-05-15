import os

import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor 
from ctcdecode import CTCBeamDecoder

class SpeechRecognizer:
    
    def __init__(self, model_dir=None, hub_name=None,device="cuda"):
        
        #if model comes locally
        if model_dir != None:
            self.model = Wav2Vec2ForCTC.from_pretrained(model_dir)
            #currently only this confiqurations
            tokenizer = Wav2Vec2CTCTokenizer(os.path.join(model_dir, "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
            self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        
        #model from huggingface hub
        elif hub_name != None:
            pass

        else:
            print("no model path given")

        self.model.to(device)
        self.device = device
        
    def _prepareaudio(self, path : str):
        # load audio
        # regex
        audio, sr = torchaudio.load(path)
        audio = audio[0]
        audio = librosa.resample(audio.numpy().squeeze(), sr, 16_000)
        return audio
    
    def decode(self, output: torch.tensor, mode: str = "argmax"):
        
        if mode=="argmax":
            pred_ids = torch.argmax(output, dim=-1)
            
        return pred_ids
    
    def get_vocab(self):
        return self.processor.tokenizer.get_vocab()

    def get_labels(self, blank_symbol="[PAD]", word_delimiter="|"):
        """
        returns labels and blank_id required by the decoder
        """
        vocab = self.get_vocab()
        vals = sorted(vocab.items(), key = lambda x:x[1])
        labels = ([x[0] for x in vals])
        blank_id = labels.index(blank_symbol)
        delimiter_index = labels.index('|')
        labels[delimiter_index] = ' '
        return labels, blank_id

    @torch.no_grad()
    def __call__(self, path: str):
        audio = self._prepareaudio(path)
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        logits = self.model(inputs.input_values.to(self.device), attention_mask=inputs.attention_mask.to(self.device)).logits
        pred_ids = self.decode(logits)
        prediction = self.processor.batch_decode(pred_ids)
        return prediction, logits

    def _from_hub(self, name):
        pass


class CTCDecoder:

    def __init__(self, 
            labels, 
            lm_path=None, 
            alpha=1.5, 
            beta=0.8,
            cutoff_top_n=15,
            cutoff_prob=1.0,
            beam_width=256,
            num_processes=4,
            blank_id=31,
            log_probs_input=False):

        print("Initializing Decoder")
        self.decoder = CTCBeamDecoder(
            labels,
            model_path = lm_path,
            alpha=alpha,
            beta=beta,
            cutoff_top_n=cutoff_top_n,
            cutoff_prob=cutoff_prob,
            beam_width=beam_width,
            num_processes=num_processes,
            blank_id=blank_id,
            log_probs_input=log_probs_input
        )

        self.decode_dict = self._dict_from_labels(labels)
        print("Decoder ready")

    def _dict_from_labels(self, labels):
        d = {}
        for i in range(len(labels)):
            d[i] = labels[i]
        return d

    def map_to_chars(self, ids):
        return "".join([self.decode_dict[i] for i in ids])

    def decode(self, probs):
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(probs)
        return self.map_to_chars(beam_results[0][0][:out_lens[0][0]].numpy())


if __name__ == "__main__":

    recog = SpeechRecognizer(model_dir="data/voxpopuli-finetuned/")
    pred, logits = recog("data/5c6fbee2-7595-4367-9170-5e9c594fc73e.mp3")
    print(pred)
    labels, blank = recog.get_labels()
    #print(labels, blank)
    decoder = CTCDecoder(labels, lm_path="data/fi_5gram_lm.bin" ,blank_id=31)
    print(decoder.decode(logits.softmax(dim=2).cpu()))