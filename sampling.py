import os
import numpy as np
import math
import torch
from torch import nn
from typing import Dict, Optional, Callable, List, Generator

def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids

def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids

def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids

def sampling_ids(
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        speech_token_size,
        sampling: int,
        ignore_eos: bool = True,
):
    num_trials, max_trials = 0, 100
    while True:
        top_ids = ras_sampling(weighted_scores, decoded_tokens, sampling)
        if (not ignore_eos) or .speech_token_size not in top_ids):
            break
        num_trials += 1
        if num_trials > max_trials:
            raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
    return top_ids
