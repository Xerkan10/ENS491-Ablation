import math

import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng



def bucket_manip(buckets,op='concat'):
    bucket_sizes = [len(b) for b in buckets]
    if 1 in bucket_sizes:
        if op == 'concat':
            buckets[-2].extend(buckets[-1])
            buckets.pop(-1)
        elif op == 'split':
            new_bucket = buckets[-2] + (buckets[-1])
            size = len(new_bucket)
            split = size // 2
            buckets[-2] = new_bucket[:split]
            buckets[-1] = new_bucket[split:]
    return buckets

def shuffle_cluster_inds(cluster):
    new_cluster = []
    for c in cluster:
        p = torch.randperm(len(c))
        new_cluster.append(c[p])
    return cluster

class Clipping_seq_L2(_BaseAggregator):
    def __init__(self, args):
        super(Clipping_seq_L2, self).__init__()
        self.tau = args.tau
        self.buck_len = args.buck_len
        self.buck_len_l2 = args.buck_len_ecc
        self.bucket_op = args.bucket_op
        self.combine_bucket = args.combine_bucket
        self.shuffle =  args.shuffle_bucket_order
        self.momentum = None
        self.mult_ref = args.multi_clip
        self.fixed_ref = args.ref_fixed
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.L2 = torch.nn.PairwiseDistance(p=2)
        self.shuffle_clusters = False
        self.n_iter = args.n_iter


    def buck_rand_sel(self,inputs,ecc=False):
        buck_len = self.buck_len
        cl_list = np.random.choice(len(inputs), len(inputs), replace=False)
        inputs = torch.stack(inputs)
        num_buck = len(inputs) // buck_len
        cl_list = np.array_split(cl_list, num_buck)
        buckets = []
        for cl_buck in cl_list:
            buckets.append(inputs[cl_buck])
        buckets = {i: bucket for i, bucket in enumerate(buckets)}
        return buckets

    def bucket_cos(self, inputs, ecc=False):## last bucket need to fixed non-equal-buckets
        buck_len = self.buck_len
        num_client = len(inputs)
        ref = self.momentum.repeat(num_client,1)
        inputs = torch.stack(inputs)
        sims = torch.cosine_similarity(ref,inputs,dim=1)
        sort = torch.argsort(sims).long()
        #print(sort)
        inputs_sorted = inputs[sort]
        #inputs_sorted = reversed(inputs[sort]) # reversed can be used
        clusters = torch.tensor_split(inputs_sorted, buck_len)
        cls_sizes = torch.tensor([len(c) for c in clusters])
        num_bucket = math.ceil(num_client / buck_len)
        cls_perms = [torch.randperm(s) for s in cls_sizes]
        buckets = [[] for i in range(num_bucket)]
        for perm, clstr in zip(cls_perms, clusters):
            [buckets[p].append(c) for p, c in zip(perm, clstr)]
        buckets = bucket_manip(buckets,self.bucket_op)
        bucket = {i: buckets[i] for i in range(len(buckets))}
        return bucket

    def bucket_L2(self,inputs):
        buck_len = self.buck_len_l2
        num_client = len(inputs)
        mean = sum(v for v in inputs) / len(inputs)
        dists = [self.L2(v, mean) for v in inputs]
        dists2 = [self.L2(v, self.momentum) for v in inputs]
        norms = [round(torch.norm(v).item(),2) for v in inputs]
        dists_ = [round(d.item(),2)for d in dists]
        dists2_ = [round(d.item(),2) for d in dists2]
        print(f"Dists: {dists_}")
        print(f"Dists2: {dists2_}")
        sort = torch.argsort(torch.tensor(dists)).long()
        inputs_sorted = torch.stack(inputs)[sort]
        # inputs_sorted = reversed(inputs[sort]) # reversed can be used
        clusters = torch.tensor_split(inputs_sorted, buck_len)
        cls_sizes = torch.tensor([len(c) for c in clusters])
        num_bucket = math.ceil(num_client / buck_len)
        cls_perms = [torch.randperm(s) for s in cls_sizes]
        buckets = [[] for i in range(num_bucket)]
        for perm, clstr in zip(cls_perms, clusters):
            [buckets[p].append(c) for p, c in zip(perm, clstr)]
        buckets = bucket_manip(buckets, self.bucket_op)
        bucket = {i: buckets[i] for i in range(len(buckets))}
        return bucket

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        flag = 1
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
            flag = 0
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        if flag:
            bucket = self.bucket_cos(inputs)
        else:
            bucket = self.buck_rand_sel(inputs)
        L2_bucket = self.bucket_L2(inputs)
        if self.combine_bucket:
            for ins_cos,ins_l2 in zip(bucket.values(), L2_bucket.values()):
                buck_avg = sum(v.to(device) for v in ins_cos) / len(ins_cos)
                l2_buck_avg = sum(v.to(device) for v in ins_l2) / len(ins_l2)
                all_avg = sum([*ins_cos,*ins_l2]) / (len(ins_cos) + len(ins_l2))
                self.momentum = (
                        self.clip(all_avg - self.momentum)
                        + self.momentum
                )
        else:
            if not self.shuffle:
                for val in bucket.values():
                    self.momentum = (
                            self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                            + self.momentum
                    )
                for val in L2_bucket.values():
                    self.momentum = (
                            self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                            + self.momentum
                    )
            else: # Randomized Order
                all_buckets = [*bucket.values(), *L2_bucket.values()]
                np.random.shuffle(all_buckets)
                for val in all_buckets:
                    self.momentum = (
                            self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                            + self.momentum
                    )
        self.momentum = self.momentum.to(inputs[0])
        return torch.clone(self.momentum).detach()