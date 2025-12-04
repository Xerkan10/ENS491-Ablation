import math
from copy import deepcopy
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
        else:
            pass
    return buckets

def shuffle_cluster_inds(cluster):
    new_cluster = []
    for c in cluster:
        p = torch.randperm(len(c))
        new_cluster.append(c[p])
    return cluster

class Clipping_ultimate(_BaseAggregator):
    def __init__(self, tau=1.0, buck_avg = True ,buck_len=3, buck_len_l2=3, sequantial_update = False,bucket_op=None,apply_TM=False):
        super(Clipping_ultimate, self).__init__()
        self.tau = tau
        self.buck_len = buck_len
        self.buck_len_l2 = buck_len_l2
        self.bucket_op = bucket_op
        self.buck_avg = buck_avg
        self.momentum = None
        self.sequantial_update = sequantial_update
        self.apply_TM = apply_TM
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.L2 = torch.nn.PairwiseDistance(p=2)


    def buck_rand_sel(self,inputs):
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

    def bucket_cos(self, inputs):## last bucket need to fixed non-equal-buckets
        buck_len = self.buck_len 
        num_client = len(inputs)
        ref = self.momentum.repeat(num_client,1)
        inputs = torch.stack(inputs)
        sims = torch.cosine_similarity(ref,inputs,dim=1)
        sort = torch.argsort(sims).long()
        inputs_sorted = inputs[sort]
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

    def bucket_L2_rand(self,inputs):
        buck_len = self.buck_len_l2
        num_client = len(inputs)
        dists = [self.L2(v, self.momentum) for v in inputs]
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
    
    def bucket_L2_ordered(self,inputs):
        buck_len = self.buck_len_l2
        num_client = len(inputs)
        dists = [self.L2(v, self.momentum) for v in inputs]
        sort = torch.argsort(torch.tensor(dists)).long()
        inputs_sorted = torch.stack(inputs)[sort]
        # inputs_sorted = reversed(inputs[sort]) # reversed can be used
        num_clustes = num_client // buck_len
        num_bucket = math.ceil(num_client / buck_len)
        #buckets = torch.tensor_split(inputs_sorted, num_clustes) # option 1 
        buckets = torch.split(inputs_sorted, buck_len) # option 2
        listed_buckets = []
        for b in buckets:
            listed_buckets.append([v for v in b])
        #print(len(buckets))
        buckets = bucket_manip(listed_buckets, self.bucket_op)
        bucket = {i: buckets[i] for i in range(len(buckets))}
        return bucket
    
    def TM(self, inputs, b):
        stacked = torch.stack(inputs, dim=0)
        largest, _ = torch.topk(stacked, b, 0)
        neg_smallest, _ = torch.topk(-stacked, b, 0)
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= len(inputs) - 2 * b
        return new_stacked


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
            momentum_cos= deepcopy(self.momentum)
            momentum_l2 = deepcopy(self.momentum)
            cos_buckets = self.bucket_cos(inputs)
            l2_buckets = self.bucket_L2_rand(inputs)
            cos_clipped = []
            l2_clipped = []
            if self.sequantial_update:
                for cos,l2 in zip(cos_buckets.values(), l2_buckets.values()):
                    if self.buck_avg:
                        buck_avg_cos = sum(v.to(device) for v in cos) / len(cos)
                        buck_avg_l2 = sum(v.to(device) for v in l2) / len(l2)
                        momentum_cos = (
                                self.clip(buck_avg_cos - momentum_cos)
                                + momentum_cos
                        )
                        momentum_l2 = (
                                self.clip(buck_avg_l2 - momentum_l2)
                                + momentum_l2
                        )
                        cos_clipped.append(deepcopy(momentum_cos))
                        l2_clipped.append(deepcopy(momentum_l2))
                    else:
                        cos_buck = [self.clip(v.to(device) - momentum_cos) for v in cos]
                        l2_buck = [self.clip(v.to(device) - momentum_l2) for v in l2]
                        momentum_cos = sum(cos_buck) / len(cos_buck) + momentum_cos
                        momentum_l2 = sum(l2_buck) / len(l2_buck) + momentum_l2
                        cos_clipped.append(deepcopy(momentum_cos))
                        l2_clipped.append(deepcopy(momentum_l2))
                if self.apply_TM:
                    new_inputs = cos_clipped + l2_clipped
                    tm_aggr = self.TM(new_inputs, b=2)
                    self.momentum = tm_aggr
                else:
                    self.momentum = (momentum_cos + momentum_l2) / 2   
            else:
                for cos,l2 in zip(cos_buckets.values(),l2_buckets.values()): # Sequential reference update
                    if self.buck_avg:
                        cos_buck = (
                                    self.clip(sum(v.to(device) for v in cos) / len(cos) - self.momentum)
                            )
                        l2_buck = ( 
                            self.clip(sum(v.to(device) for v in l2) / len(l2) - self.momentum)  
                        )
                    else:
                        cos_buck_list = [self.clip(v.to(device) - self.momentum) for v in cos]
                        cos_buck = sum(cos_buck_list) / len(cos_buck_list)
                        l2_buck_list = [self.clip(v.to(device) - self.momentum) for v in l2]
                        l2_buck = sum(l2_buck_list) / len(l2_buck_list)
                    cos_clipped.append(cos_buck)
                    l2_clipped.append(l2_buck)

                if self.apply_TM:
                    new_inputs = cos_clipped + l2_clipped
                    tm_aggr = self.TM(new_inputs, b=2)
                    self.momentum = tm_aggr + self.momentum
                else:
                    cos_clipped = sum(cos_clipped) / len(cos_clipped)
                    l2_clipped = sum(l2_clipped) / len(l2_clipped)
                    self.momentum = (cos_clipped + l2_clipped) / 2 + self.momentum
        else:
            bucket = self.buck_rand_sel(inputs)
            clipped = []
            for val in bucket.values():
                    buck_cliped = self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                    clipped.append(buck_cliped)
            self.momentum = sum(clipped) / len(clipped) + self.momentum
        self.momentum = self.momentum.to(inputs[0])
        return torch.clone(self.momentum).detach()
    
if __name__ == "__main__": 
    # Test the trimmed mean aggregator
    def test_aggr(clients=25,dim=6e5):
        # Create sample inputs
        #print(torch.sort(torch.stack(inputs,dim=0),dim=0)[0])
        inputs = [torch.randn(int(dim),device='cpu') for _ in range(clients)]
        # Initialize trimmed mean with b=1 (trim 1 from each end)
        aggr = Clipping_ultimate(tau=1.0, buck_len=3)
        
        # Test aggregation
        result = aggr(inputs)
        results = aggr(inputs)
        print(f"AGGR  result: {result}")
        print(f"AGGR result2: {results}")
        

    test_aggr()