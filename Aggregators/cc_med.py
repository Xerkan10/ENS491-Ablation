import math

import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng


class cc_med(_BaseAggregator):
    def __init__(self, tau,buck_len=5, buck_rand=False):
        self.tau = tau
        self.buck_len = buck_len
        self.buck_rand = buck_rand
        super(cc_med, self).__init__()
        self.momentum = None

    def bucket_rand(self,inputs):
        n = len(inputs)
        buck_len = self.buck_len
        l = math.ceil(n/buck_len)
        cl_list = np.arange(n)
        bucket = {i:[] for i in range(l)}
        device = inputs[0].get_device()
        for i in range(l):
            avail_cl = buck_len if len(cl_list) > buck_len else len(cl_list)
            rng = default_rng()
            numbers = rng.choice(len(cl_list), size=avail_cl, replace=False)
            selected = cl_list[numbers]
            cl_list = np.delete(cl_list, numbers)
            new_inputs = [ins.detach().clone().cpu() for ins in inputs]
            bucket[i] = np.asarray(new_inputs)[selected]
        for vals in bucket.values():
            [v.to(device) for v in vals]
        [v.to(device) for v in inputs]
        return bucket

    def bucket_cos(self,inputs):
        buck_len = self.buck_len
        n = len(inputs)
        l = math.ceil(n / buck_len)
        cl_list = np.arange(n)
        bucket = {i: [] for i in range(l)}
        cos_sims = [torch.cosine_similarity(self.momentum,i,dim=0).detach().cpu().item() for i in inputs]
        cl_sorted = np.asarray(cos_sims).argsort()
        group_id = [i%l for i in cl_list]
        device = inputs[0].get_device()
        for key in bucket.keys():
            grp = np.asarray(group_id) == key
            new_inputs = [ins.detach().clone().cpu() for ins in inputs]
            new_inputs = np.asarray(new_inputs)
            bucket[key] = new_inputs[cl_sorted[grp]]
        for vals in bucket.values():
            [v.to(device) for v in vals]
        [v.to(device) for v in inputs]
        return bucket

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale


    def __call__(self, inputs):
        flag = 0
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
            flag = 1
        device = inputs[0].get_device()
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        median = (values_upper - values_lower) / 2
        clipped_med = self.clip(median - self.momentum)
        if flag or self.buck_rand:
            bucket = self.bucket_rand(inputs)
        else:
            bucket = self.bucket_cos(inputs)
            flag = 0

        med_refs = [clipped_med.sub(self.momentum).mul(i / (len(bucket.keys()) + 1)).add(self.momentum)
                    for i in range(1, len(bucket.keys()) + 1)]
        orig_ref = self.momentum.detach().clone()
        for i,ins in enumerate(bucket.values()):
            self.momentum = (
                sum(self.clip(v.to(device)-self.clip(v.to(device) - med_refs[i]) - self.momentum)for v in ins) / len(ins)
                + self.momentum
            )
        self.momentum = self.momentum.to(inputs[0])

        return torch.clone(self.momentum).detach()

