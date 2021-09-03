import torch
import numpy as np
from nmt import Pipeline, FairseqModel
from tqdm import tqdm
import numpy as np
from collections import namedtuple, OrderedDict

Return = namedtuple('Return', ['pipeline_x', 'pipeline_y', 'x2y', 'y2x'])


def clean_corpus(corpus_x, corpus_y, tokenizer_x, tokenizer_y, max_length_x=None, max_length_y=None):
    clean_x, clean_y = [], []
    for x, y in zip(corpus_x, corpus_y):
        if max_length_x and len(tokenizer_x(x)) > max_length_x:
            continue
        if max_length_y and len(tokenizer_y(y)) > max_length_y:
            continue
        clean_x.append(x)
        clean_y.append(y)
    return clean_x, clean_y


def load_de_en(np_seed=None, torch_seed=None, device=torch.device('cpu')):

    rng = np.random.RandomState(seed=np_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    pipeline_de = Pipeline(
        "de", "data/pretrained-fairseq/de-en/truecaser/truecase-model.de",
        True, True, True
    )
    pipeline_en = Pipeline(
        "en", "data/pretrained-fairseq/de-en/truecaser/truecase-model.en",
        True, True, True
    )
    deen = FairseqModel(
        model_path='data/pretrained-fairseq/de-en/mle/', 
        bin_path='data/pretrained-fairseq/de-en', 
        src_spm="data/pretrained-fairseq/de-en/sentencepiece.bpe.model",
        tgt_spm="data/pretrained-fairseq/de-en/sentencepiece.bpe.model",
        src_pipeline=pipeline_de,
        tgt_pipeline=pipeline_en,
        device=device
    )
    ende = FairseqModel(
        model_path='data/pretrained-fairseq/en-de/mle/', 
        bin_path='data/pretrained-fairseq/en-de', 
        src_spm="data/pretrained-fairseq/en-de/sentencepiece.bpe.model",
        tgt_spm="data/pretrained-fairseq/en-de/sentencepiece.bpe.model",
        src_pipeline=pipeline_en,
        tgt_pipeline=pipeline_de,
        device=device
    )

    return Return(pipeline_x=pipeline_de, pipeline_y=pipeline_en, x2y=deen, y2x=ende)



class Sampler:
    """
    A mechanism to obtain samples from Y|X=x for a given x.
    """
    
    def __call__(self, src: str):
        """Return a number of samples from Y|X=src"""
        raise NotImplementedError("Implement me!")
        
        
class SampleFromBuffer(Sampler):
    """
    This sampler works for a pre-specified collections of inputs, for each input, the user must
     provide a collection of samples during construnction.
     
    For a known src, this class will return either all available samples (if sample_size=None in the constructor)
     or a sample (with replacement) of a given size.
    """
    
    def __init__(self, samples: dict, sample_size=None):
        """
        Parameters
        
            samples: maps a source sentence to a list of samples
            sample_size: None means using all samples in the dict, any other strictly positive integer means sampling with replacement from the samples in the dict        
        """
        self._samples = samples
        self._sample_size = sample_size
        
    def __call__(self, src: str):
        """Return a list of samples drawn from a fixed pool"""
        try:
            samples = self._samples[src]
        except KeyError:
            raise KeyError(f"You have not provided samples for source sentence: {src}")
        if self._sample_size is None:
            return samples
        else:
            indices = np.random.choice(len(samples), self._sample_size)
            return [samples[i] for i in indices]
        

class SampleFromNMT(Sampler):
    """
    This sampler has access to the NMT model itself, thus it works for any given input x.
    
    Every time we run __call__, a number of samples will be drawn from the NMT model.
    """
    
    def __init__(self, model: FairseqModel, sample_size: int):
        """
        Parameters
            model: a FairseqModel from source to target language
            sample_size: a strictly positive integer            
        """
        self._model = model
        self._sample_size = sample_size
        
    def __call__(self, src: str):
        """Return a list of samples drawn independently from the NMT model"""
        r = self._model.ancestral_sampling(src, num_samples=self._sample_size)
        return r['output']
    
class CachedSampler(Sampler):
    """
    Every time we run __call__ with a new hypothesis, a number of samples will be drawn from the NMT model,
    then either __call__ will return all these samples (if sample_size=None) or a subset drawn with replacement.
    
    This class implements a caching mechanism allowing us to reuse samples for calls that share the same hypothesis.
    """
    
    def __init__(self, sampler: SampleFromNMT, sample_size: int, cache_max_size=1):
        """
        Parameters
            model: a FairseqModel from source to target language
            sample_size: a strictly positive integer            
        """
        self._sampler = sampler
        self._cache = OrderedDict()
        self._sample_size = sample_size
        self._cache_max_size = cache_max_size
        
    def __call__(self, src: str):        
        """Return a list of samples drawn independently from the NMT model"""
        samples = self._cache.get(src, None)
        if samples is None:
            samples = self._sampler(src)
            self._cache[src] = samples
            if self.cache_max_size and len(self._cache) > self._cache_max_size:  # clear cache
                self._cache.popitem(last=False)
        if self._sample_size is None:
            return samples
        else:
            indices = np.random.choice(len(samples), self._sample_size)
            return [samples[i] for i in indices]

    def reset(self):
        self._cache = OrderedDict()
        

class Utility:
    """
    Assign utility to a hypothesis.
    """
    
    def __call__(self, src: str, hyp: str, ref: str) -> float:
        """Return the utility (float) of hyp as a translation of src when ref is the correct/preferred translation)."""
        raise NotImplementedError("Implement me!")
        
    def batch(self, src: str, hyp: str, refs: list):
        """Compute utility against a list of references"""
        return np.array([self(src=src, hyp=hyp, ref=ref) for ref in refs])
                
    
class MBR:
    """
    A simple implementation of MBR decoding. 
    
    This implementation is not optimised for efficiency, rather, for clarity.
    """
    
    def __init__(self, utility: Utility, sampler: Sampler):        
        self.utility = utility
        self.sampler = sampler
               
    def mu(self, src: str, hyp: str):
        """Return an estimate of the expected utility of hyp given src"""
        return np.mean(self.utility.batch(src=src, hyp=hyp, refs=self.sampler(src)))
    
    def mus(self, src: str, hyp_space: list):
        """Return a numpy array as long as hyp_space, each cell is an estimate of the expected utility of the corresponding hypothesis given src."""
        return np.array([self.mu(src, hyp) for hyp in hyp_space])
    
    def decode(self, src: str, hyp_space: list):
        """Estimate the expected utility of each hypothesis in hyp_space given src and return the best hypothesis"""
        mus = self.mus(src, hyp_space)
        idx = np.argmax(mus)
        return hyp_space[idx] 
