import torch
from fairseq.models.transformer import TransformerModel
from typing import List
from fairseq.data.data_utils import collate_tokens
import sentencepiece as spm
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer, MosesTruecaser, MosesDetruecaser, MosesDetokenizer
import numpy as np


class Pipeline:
    """
    This deals with text normalization other than word segmentation via Sentencepiece.
    """
    
    def __init__(self, lang, truecaser_model, do_normalize=False, do_tokenize=False, do_truecase=False):
        self.lang = lang
        self.truecaser_model = truecaser_model
        self.do_normalize = do_normalize
        self.do_tokenize = do_tokenize
        self.do_truecase = do_truecase
        
        self.punct = MosesPunctNormalizer(lang=lang) if do_normalize else None
        self.tokenizer = MosesTokenizer(lang=lang) if do_tokenize else None
        self.detokenizer = MosesDetokenizer(lang=lang) if do_tokenize else None
        self.truecaser = MosesTruecaser(truecaser_model) if do_truecase else None
        self.detruecaser = MosesDetruecaser() if do_truecase else None

    def encode(self, text: str) -> str:
        if self.do_normalize:
            text = self.punct.normalize(text)
        if self.do_tokenize:
            text = self.tokenizer.tokenize(text, return_str=True)
        if self.do_truecase:
            text = self.truecaser.truecase(text, return_str=True)
        return text
    
    def decode(self, text: str) -> str:
        if self.do_truecase:
            tokens = self.detruecaser.detruecase(text)
        else:
            tokens = text.split()
        if self.do_tokenize:
            text = self.detokenizer.detokenize(tokens, return_str=True)
        else:
            text = ' '.join(tokens)
        return text
    

class FairseqModel:
    
    def __init__(
        self, model_path: str, bin_path: str, src_spm: str, tgt_spm: str,
        src_pipeline: Pipeline, tgt_pipeline: Pipeline,
        device=torch.device('cpu'),
        checkpoint_file='averaged_model.pt',
        bpe_type='sentencepiece'
    ):        

        # Preprocessing
        self.src_pipeline = src_pipeline
        self.tgt_pipeline = tgt_pipeline
        # Tokenization into segments known by the NMT engine
        self.src_spm = spm.SentencePieceProcessor(model_file=src_spm)
        self.tgt_spm = spm.SentencePieceProcessor(model_file=tgt_spm)        
        # Model
        self.model = TransformerModel.from_pretrained(
            model_path, 
            checkpoint_file=checkpoint_file, 
            data_name_or_path=bin_path, 
            bpe=bpe_type,
            sentencepiece_model=src_spm,
        ).to(device)
        self.model.eval()    
        self.device = device

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
            
    def encode_src(self, text: str, append_eos=True):
        tensor = self.model.src_dict.encode_line(
            ' '.join(self.src_spm.encode(self.src_pipeline.encode(text), out_type=str)), 
            add_if_not_exist=False, 
            append_eos=append_eos
        ).long()
        return tensor
    
    def encode_tgt(self, text: str, append_eos=True):
        tensor = self.model.tgt_dict.encode_line(
            ' '.join(self.tgt_spm.encode(self.tgt_pipeline.encode(text), out_type=str)), 
            add_if_not_exist=False,
            append_eos=append_eos,            
        ).long()
        return tensor
    
    def tokenize_src(self, src: str):
        """Complete pre-processing pipeline and model tokenization into subword units"""
        return self.src_spm.encode(self.src_pipeline.encode(src), out_type=str)
    
    def tokenize_tgt(self, tgt: str):
        """Complete pre-processing pipeline and model tokenization into subword units"""
        return self.tgt_spm.encode(self.tgt_pipeline.encode(tgt), out_type=str)    
    
    def tgt_lengths(self, tgt_corpus):   
        """
        Compute the length of target sentences as seen by the model,
            that is, in tokenized subword units and including EOS.
        
        :param tgt_corpus: list of sentences, each a string
        :returns: lengths
        """
        return torch.tensor([len(self.tgt_spm.encode(self.tgt_pipeline.encode(text))) + 1 for s in tgt_corpus]).long()
    
    def beam_search(self, src, beam_size=1, compute_entropies=False):
        """
        :param text: a sentence (str) or a list of sentences (each a string)
        :param beam_size: 
        :param compute_entropies: this will compute the entropy of each conditional
            due to limitations of hub_utils, this does take an additional forward pass 
        :returns: a dict if type(src) is str, a List[dict] if type(str) is list
            each output dict contains:
            - output: list containing beam_size translations (each a string)
            - surprisal: - log prob(y|x) [beam_size]
            - length: [beam_size] 
            - entropy: H(Y[j]|x,y[:j]) a list of beam_size tensors, each with shape [output length]
                NOTE: there is not need for further masking
            - faiseq object returned by model.generate
        """
        if type(src) is str:
            return self.beam_search([src], beam_size=beam_size, compute_entropies=compute_entropies)[0]
      
        gen_list = self.model.generate([self.encode_src(s) for s in src], beam=beam_size, sampling=False)        

        results = []
        for gen in gen_list:
            outputs = []
            surprisals = []
            lengths = []
            entropies = []
            for i in range(beam_size):
                sample = gen[i]['tokens']
                outputs.append(self.tgt_pipeline.decode(self.model.decode(sample)))
                lengths.append((sample != self.model.tgt_dict.pad_index).sum(-1))
                surprisals.append(-gen[i]['positional_scores'].sum())
                if compute_entropies:
                    # takes an additional forward pass (because of the current hub_utils API)
                    r = self.surprisal(src, outputs[-1])
                    entropies.append(r['entropy'][0])
                else:
                    entropies.append(None)
                
            # [beam_size]
            surprisals = torch.tensor(surprisals)
            # [beam_size]
            lengths = torch.tensor(lengths).long()
                
            results.append(dict(output=outputs, surprisal=surprisals, length=lengths, entropy=entropies, fairseq=gen))
        return results
    
    def ancestral_sampling(self, src: str, num_samples=1, compute_entropies=False):
        """
        :param text: a sentence (str) or a list of sentences (each a string)
        :param num_samples: 
        :param compute_entropies: this will compute the entropy of each conditional
            due to limitations of hub_utils, this does take an additional forward pass 
        :returns: a dict if type(src) is str, or List[dict] if type(src) is list
            each output dict contains:
            - output: list containing num_samples translations (each a string)
            - surprisal: - log prob(y|x) [num_samples]
            - length: [num_samples] 
            - entropy: H(Y[j]|x,y[:j]) a list of num_samples tensors, each with shape [output length]
                NOTE: there is not need for further masking
            - faiseq object returned by model.generate
        """
        if type(src) is str:
            return self.ancestral_sampling([src], num_samples=num_samples, compute_entropies=compute_entropies)[0]

        # ancestral sampling
        gen_list = self.model.generate(
            [self.encode_src(s) for s in src], 
            beam=num_samples, 
            sampling=True, 
            temperature=1.0, 
            sampling_topk=-1, 
            sampling_topp=-1
        )
        results = []
        for gen in gen_list:
            outputs = []
            lengths = []
            surprisals = []
            entropies = []
            for i in range(num_samples):
                sample = gen[i]['tokens']
                outputs.append(self.tgt_pipeline.decode(self.model.decode(sample)))
                lengths.append((sample != self.model.tgt_dict.pad_index).sum(-1))
                surprisals.append(-gen[i]['positional_scores'].sum())
                if compute_entropies:
                    # takes an additional forward pass (because of the current hub_utils API)
                    r = self.surprisal(src, outputs[-1])  # TODO: batch it
                    entropies.append(r['entropy'][0])
                else:
                    entropies.append(None)

            # [num_samples]
            surprisals = torch.tensor(surprisals)
            # [num_samples]
            lengths = torch.tensor(lengths).long()
                
            results.append(dict(output=outputs, surprisal=surprisals, length=lengths, entropy=entropies, fairseq=gen))

        return results
    
    def surprisal(self, src: str, tgt: str):
        return self.batch_surprisal([src], [tgt])
        
    def batch_surprisal(self, src_corpus: List[str], tgt_corpus: List[str]):
        """
        :param src: a list of sentences (each a string)
        :param tgt: a list of sentences (each a string)
        :returns: dict
            - output: tgt_corpus
            - surprisal: [batch_size]
            - legnth: target lengths [batch_size]
            - entropy: entropies [batch_size, max_length]
                (use lengths to mask this along the time dimension before analysing)
            - fairseq: None
        """
        with torch.no_grad():
            src_tokens = collate_tokens(
                values=[self.encode_src(s) for s in src_corpus],
                pad_idx=self.model.src_dict.pad_index,
                eos_idx=None,
                left_pad=False,
                move_eos_to_beginning=False,
                pad_to_length=None,
                pad_to_multiple=1,
            ).to(self.device)
            src_lengths = (src_tokens != self.model.src_dict.pad_index).sum(-1)
            
            tgt_bin = [self.encode_tgt(s) for s in tgt_corpus]
            # this is the target output
            tgt_tokens = collate_tokens(
                values=tgt_bin,
                pad_idx=self.model.tgt_dict.pad_index,
                eos_idx=self.model.tgt_dict.eos_index,
                left_pad=False,
                move_eos_to_beginning=False,
                pad_to_length=None,
                pad_to_multiple=1,
            ).to(self.device)
            tgt_lengths = (tgt_tokens != self.model.tgt_dict.pad_index).sum(-1)
            
            # this is the target input
            prev_tgt_tokens = collate_tokens(
                values=tgt_bin,
                pad_idx=self.model.tgt_dict.pad_index,
                eos_idx=self.model.tgt_dict.eos_index,
                left_pad=False,
                move_eos_to_beginning=True,
                pad_to_length=None,
                pad_to_multiple=1,
            ).to(self.device)

            logits, _ = self.model.models[0](src_tokens, src_lengths, prev_tgt_tokens)
            entropies = torch.distributions.Categorical(logits=logits).entropy()
            surprisals = torch.nn.functional.cross_entropy(
                input=logits.permute(0 ,2, 1),
                target=tgt_tokens,
                ignore_index=self.model.tgt_dict.pad_index,
                reduction="none",
            ).sum(-1)           
            
            return dict(output=tgt_corpus, surprisal=surprisals, length=tgt_lengths, entropy=entropies, fairseq=None)


