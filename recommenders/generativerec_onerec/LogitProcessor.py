from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None,
        prompt_len: int = 0
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        self.prompt_len = prompt_len
        
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3
        

        self.empty_count = 0
        self.total_count = 0

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, -1000000)
        

        MAX_KEY_LEN = 3
            
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):

                generated_len = len(sent) - self.prompt_len
                
                if generated_len == 0:

                    hash_key = sent[-self.prefix_index:]
                else:

                    actual_len = min(generated_len, MAX_KEY_LEN)
                    hash_key = sent[-actual_len:]
                
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                self.total_count += 1
                if len(prefix_allowed_tokens) == 0:
                    self.empty_count += 1

                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    else:

                        mask[batch_id * self._num_beams + beam_id, :] = 0
                    continue 
                
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        scores = scores + mask
        return scores
    
    def get_stats(self):
        """Return debug statistics"""
        return {
            'empty_count': self.empty_count,
            'total_count': self.total_count,
            'empty_ratio': self.empty_count / max(1, self.total_count)
        }