import math
from copy import deepcopy
import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng
import random

class Clipping_per(_BaseAggregator):
    def __init__(self, tau, buck_len=5,perm=2,ex_clip=False):
        self.tau = tau
        self.buck_len = buck_len
        self.perm = perm
        self.ex_clip = ex_clip
        super(Clipping_per, self).__init__()
        self.momentum = None

    def bucket_rand(self, inputs):
        n = len(inputs)
        buck_len = self.buck_len
        l = math.ceil(n / buck_len)
        cl_list = np.arange(n)
        bucket = {i: [] for i in range(l)}
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
        return [bucket]

    def get_buckets(self,inputs,inds,l):
        bucket = {i: [] for i in range(l)}
        mask = [torch.ones_like(ind).bool() for ind in inds]
        for buck in bucket.values():
            for i,(ind, m) in enumerate(zip(inds,mask)):
                m_inds = ind[m]
                #print(m_inds)
                if m.sum() > 0:
                    r = random.randint(0,len(m_inds)-1)
                    sel = m_inds[r]
                    sel_ind = int((torch.arange(0,len(ind))[ind == sel]).item())
                    buck.append(inputs[sel])
                    m[sel_ind] = False
                    mask[i] = m
        return bucket

    def bucket_cos(self, inputs):
        device = inputs[0].get_device()
        buck_len = self.buck_len
        n = len(inputs)
        l = math.ceil(n / buck_len)
        cos_sims = torch.empty(len(inputs)).to(device)
        for i,v in enumerate(inputs):
            cos_sims[i] = torch.cosine_similarity(self.momentum,v,dim=0)
        vals, inds = torch.sort(cos_sims)
        inds = inds.chunk(buck_len)
        perm_bucks = [self.get_buckets(inputs, inds, l) for i in range(self.perm)]
        return perm_bucks

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
        if flag:
            buckets = self.bucket_rand(inputs)
        else:
            buckets = self.bucket_cos(inputs)
            flag = 0
        if self.ex_clip:
            orig_ref = deepcopy(self.momentum.detach().clone())
            for bucket in buckets:
                for ins in bucket.values():
                    self.momentum = (
                            sum(self.clip(orig_ref + self.clip(v.to(device) - orig_ref) - self.momentum)
                                for v in ins) / len(ins)
                            + self.momentum
                    )
        else:
            for bucket in buckets:
                for ins in bucket.values():
                    self.momentum = (
                            sum(self.clip(v.to(device) - self.momentum) for v in ins) / len(ins)
                            + self.momentum
                    )
        self.momentum = self.momentum.to(device)

        return torch.clone(self.momentum).detach()



