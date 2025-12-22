import math
import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng



def bucket_quotas(num_client,buck_size): ## random not filled bucket
    total_bucket = math.ceil(num_client/buck_size)
    full_buckets = num_client // buck_size
    buck_quotas = np.zeros(total_bucket)
    full_inds = np.random.choice(total_bucket,full_buckets,replace=False)
    buck_quotas[full_inds] = buck_size
    left_w = int(num_client - (full_buckets*buck_size))
    if left_w >0:
        buck_quotas[buck_quotas==0] = left_w
    return buck_quotas

def bucket_quotas2(num_client,buck_size): ## last bucket may not be filled
    total_bucket = math.ceil(num_client/buck_size)
    full_buckets = num_client // buck_size
    buck_quotas = np.zeros(total_bucket)
    buck_quotas[:full_buckets] = buck_size
    left_w = int(num_client - (full_buckets*buck_size))
    if left_w >0:
        buck_quotas[-1] = left_w
    return buck_quotas

def bucket_quotas3(num_client, buck_size): ## oversized buckets can happen
    full_buckets = num_client // buck_size
    buck_quotas = np.ones(full_buckets) * buck_size
    left_w = int(num_client - (full_buckets*buck_size))
    if left_w >0:
        np.random.choice(full_buckets,left_w,replace=False)
        buck_quotas[left_w]+=1
    return buck_quotas

def bucket_quotas4(num_client, buck_size): ## balanced
    num_buck = math.ceil(num_client/buck_size)
    buck = np.array_split(np.zeros(num_client),num_buck)
    buck_quotas = np.zeros(num_buck)
    for i,l in enumerate(buck):
        buck_quotas[i] = len(l)
    return buck_quotas

def shuffle_cluster_inds(cluster):
    new_cluster = []
    for c in cluster:
        p = torch.randperm(len(c))
        new_cluster.append(c[p])
    return cluster

class Clipping_rand(_BaseAggregator):
    def __init__(self, tau, buck_len=3):
        self.tau = tau
        self.buck_len = buck_len
        super(Clipping_rand, self).__init__()
        self.momentum = None
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.shuffle_clusters = False


    def buck_rand_sel(self,inputs):
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        buck_len = 3
        cl_list = np.random.choice(len(inputs), len(inputs), replace=False)
        inputs = torch.stack(inputs)
        num_buck = len(inputs) // buck_len
        cl_list = np.array_split(cl_list, num_buck)
        buckets = []
        for cl_buck in cl_list:
            buckets.append(inputs[cl_buck])
        buckets = {i: bucket for i, bucket in enumerate(buckets)}
        return buckets


    def bucket_cos(self, inputs): ### Non-collusion asserted
        buck_len = self.buck_len
        n = len(inputs)
        l = math.ceil(n / buck_len)
        cl_list = np.arange(n)
        bucket = {i: [] for i in range(l)}
        cos_sims = [torch.cosine_similarity(self.momentum, i, dim=0).detach().cpu().item() for i in inputs]
        cl_sorted = np.asarray(cos_sims).argsort()
        group_id = [i % l for i in cl_list]
        #device = inputs[0].get_device()
        for key in bucket.keys():
            grp = np.asarray(group_id) == key
            inds = cl_sorted[grp]
            #bucket[key] = new_inputs[cl_sorted[grp]]
            bucket[key] = [inputs[i] for i in inds]
        #for vals in bucket.values():
        #    [v.to(device) for v in vals]
        #[v.to(device) for v in inputs]
        return bucket

    def bucket_cos_(self,inputs):## last bucket need to fixed non-equal-buckets
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
        bucket = {i: buckets[i] for i in range(len(buckets))}
        return bucket

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale
    
    def clip_and_update(self, grads: list, ref:torch.Tensor,avg:bool=False):
        if avg:
            mean_grad = sum(grads) / len(grads)
            grad_clipped = self.clip(mean_grad - ref)
            new_ref = ref + grad_clipped
        else:
            clipped_grads = [self.clip(g - ref) for g in grads]
            new_ref = ref + sum(clipped_grads) / len(clipped_grads)
        return new_ref
    
    def clip_and_update2(self, grads: list, ref:torch.Tensor,avg:bool=False): # this give new locations
        if avg:
            mean_grad = sum(grads) / len(grads)
            grad_clipped = self.clip(mean_grad - ref)
            new_ref = ref + grad_clipped
        else:
            clipped_grads = [self.clip(g - ref) for g in grads]
            new_ref = [ref + g for g in clipped_grads]
        return new_ref

    def __call__(self, inputs):
        flag = 1
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
            flag = 0
        #device = inputs[0].get_device()
        #device = device if device > -1 else "cpu"
        if flag:
            bucket = self.bucket_cos_(inputs)
            bucket2 = self.bucket_cos_(inputs)
        else:
            bucket = self.buck_rand_sel(inputs)
            bucket2 = self.buck_rand_sel(inputs)
        orig_ref = self.momentum.detach().clone()
        clipped_buckets = [self.clip(sum(v for v in val) / len(val) - self.momentum)
                    + self.momentum for val in bucket.values()]
        final_clip = []
        for i, val in enumerate(bucket2.values()): # option 1
            final_clip.append(self.clip_and_update(val, clipped_buckets[i]))
        self.momentum = sum(final_clip) / len(final_clip) # option 1
        #for i, val in enumerate(bucket2.values()): # option 2
            #final_clip.append(self.clip(sum(v.to(device) for v in val) / len(val) - clipped_buckets[i]))
        #self.momentum = orig_ref + sum(final_clip) / len(final_clip) # option 2 
        self.momentum = self.momentum.to(inputs[0])

        return torch.clone(self.momentum).detach()


    



if __name__ == "__main__": 
    # Test the trimmed mean aggregator
    def test_aggr(clients=25,dim=6e5):
        # Create sample inputs
        #print(torch.sort(torch.stack(inputs,dim=0),dim=0)[0])
        inputs = [torch.randn(int(dim),device='cpu') for _ in range(clients)]
        # Initialize trimmed mean with b=1 (trim 1 from each end)
        aggr = Clipping_rand(tau=1.0, buck_len=3)
        
        # Test aggregation
        result = aggr(inputs)
        results = aggr(inputs)
        print(f"AGGR  result: {result}")
        print(f"AGGR result2: {results}")
        

    test_aggr()
