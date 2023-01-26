import numpy as np
import torch


class Metrics():
    def __init__(self):
        self.reset()

    def reset(self):
        self.running_agg = {}
        self.running_count = {}

    @torch.no_grad()
    def step(self, **additional_terms):
        for term in additional_terms:
            if term not in self.running_agg:
                self.running_agg[term] = additional_terms[term].detach()
                self.running_count[term] = 1
            else:
                self.running_agg[term] += additional_terms[term].detach()
                self.running_count[term] += 1


    @torch.no_grad()
    def peek(self):
        values = {}
        for key in self.running_agg:
            values[key] = float(self.running_agg[key] / self.running_count[key])
        return values


    @torch.no_grad()
    def evaluate(self):
        values = self.peek()
        self.reset()
        return values
