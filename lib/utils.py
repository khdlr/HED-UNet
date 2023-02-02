import torch
from typing import Iterable

def sample_map(fn):
  def inner(*args):
    out = []
    for sample in zip(*args):
      out.append({k: fn(*[s[k] for s in sample], kind=k) for k in sample[0]})
    return out
  return inner


def mean(items: Iterable[torch.Tensor]) -> torch.Tensor:
  return torch.mean(torch.stack(list(items)))


