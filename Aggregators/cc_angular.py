import torch
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
from .base import _BaseAggregator

class Clipping_angular(_BaseAggregator):
    def __init__(self, tau, buck_len=3,buck_avg=False,bucket_op='concat',cosine=0):
        self.tau = tau
        self.buck_len = buck_len
        self.buck_avg = buck_avg
        super(Clipping_angular, self).__init__()
        self.momentum = None
        self.bucket_op = bucket_op  # Default operation for bucket manipulation
        self.cosine = cosine
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.L2 = torch.nn.PairwiseDistance(p=2)
        self.shuffle_clusters = False

    def bucket_manip(self, buckets,op='concat'):
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

    def bucket_L2_rand(self,inputs):
        buck_len = self.buck_len
        num_client = len(inputs)
        dists = [self.L2(v, self.momentum) for v in inputs]
        #print(dists)
        sort = torch.argsort(torch.tensor(dists)).long()
        #print(sort)
        inputs_sorted = torch.stack(inputs)[sort]
        # inputs_sorted = reversed(inputs[sort]) # reversed can be used
        clusters = torch.tensor_split(inputs_sorted, buck_len)
        cls_sizes = torch.tensor([len(c) for c in clusters])
        num_bucket = math.ceil(num_client / buck_len)
        cls_perms = [torch.randperm(s) for s in cls_sizes]
        buckets = [[] for i in range(num_bucket)]
        for perm, clstr in zip(cls_perms, clusters):
            [buckets[p].append(c) for p, c in zip(perm, clstr)]
        buckets = self.bucket_manip(buckets, self.bucket_op)
        bucket = {i: buckets[i] for i in range(len(buckets))}
        return bucket
    
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
        buckets = self.bucket_manip(buckets,self.bucket_op)
        bucket = {i: buckets[i] for i in range(len(buckets))}
        return bucket


    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale
    
    def clip_and_correct(self,v,cos):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        corrected = self.set_cosine_similarity(v*scale,self.momentum,cos)
        return corrected

    def __call__(self, inputs):
        flag = 1
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
            flag = 0
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        if flag:
            #avg_cos,median_cos = self.average_cos_similarity(inputs,self.momentum)
            #cosine_corrected = [self.set_cosine_similarity(v,self.momentum,avg_cos) for v in inputs]
            #self.average_cos_similarity(cosine_corrected,self.momentum)
            # norms = [round(torch.norm(v-self.momentum).item(),2) for v in cosine_corrected
            #          ]
            # print('norms')
            # print(norms)
            # dists = [round(self.L2(self.momentum,v).item(),2) for v in cosine_corrected]
            # print("dists")
            # print(dists)
            clipped_vals = [self.clip(v.to(device) - self.momentum) for v in inputs]
            avg_cos,med_cos = self.average_cos_similarity(clipped_vals,self.momentum)
            bucket = self.bucket_L2_rand(inputs)
            if self.buck_avg:
                for val in bucket.values():
                    self.momentum = (
                        self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                        + self.momentum
                    )
            else:
                for ins in bucket.values():
                    self.momentum = (
                        sum(self.clip_and_correct(v.to(device) - self.momentum,avg_cos) for v in ins) / len(ins)
                        + self.momentum
                    )
        else:
            bucket = self.buck_rand_sel(inputs)
            avgs = [sum(val) / len(val) for val in bucket.values()]
            self.momentum = sum([self.clip(avg-self.momentum) for avg in avgs]) / len(avgs) + self.momentum
        self.momentum = self.momentum.to(inputs[0])

        return torch.clone(self.momentum).detach()

    def set_cosine_similarity(self, vec: torch.Tensor, ref: torch.Tensor, desired_cos: float) -> torch.Tensor:
        # Ensure ref is a unit vector
        ref_u = ref / ref.norm()
        # Remove the component of vec along ref
        proj = torch.dot(vec, ref_u) * ref_u
        ortho = vec - proj
        ortho_norm = ortho.norm()
        if ortho_norm < 1e-8:
            # vec is parallel to ref, pick any orthogonal direction
            ortho = torch.randn_like(ref_u)
            ortho = ortho - torch.dot(ortho, ref_u) * ref_u
            ortho = ortho / ortho.norm()
            ortho_norm = 1.0
        ortho_u = ortho / ortho_norm
        # Construct new vector with desired cosine similarity
        vec_norm = vec.norm()
        new_vec = desired_cos * vec_norm * ref_u + (vec_norm * (1 - desired_cos**2)**0.5) * ortho_u
        return new_vec
    
    def average_cos_similarity(self, inputs: list ,ref: torch.Tensor):
        # Calculate the average cosine similarity of all vectors in inputs
        cos_sims = []
        for vec in inputs:
            cos_sim = round(self.cos(vec, ref).item(),2)
            cos_sims.append(cos_sim)
        median = np.median(cos_sim)
        print(f"Average Cosine Similarity: {sum(cos_sims) / len(cos_sims)}")
        degrees = [self.cosine_to_degree(d) for d in cos_sims]
        print(f"Cosine Similarities: {cos_sims}")
        print(f"degree: {degrees}")
        return sum(cos_sims) / len(cos_sims), float(median)
    

    def outlier_sanitaziation(self, inputs,b):
        stacked = torch.stack(inputs, dim=0)
        sorted_grads,_ = torch.sort(stacked,dim=0)
        mean = torch.mean(stacked,dim=0)
        std = torch.std(stacked)
        locs = sorted_grads.abs() > std

    def cosine_to_degree(self,cos_sim):
    # Clamp value to valid range to avoid numerical errors
        cos_sim = max(min(cos_sim, 1.0), -1.0)
        angle_rad = math.acos(cos_sim)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def degree_to_cosine(self,degree):
    # Convert degree to radians
        rad = math.radians(degree)
        return math.cos(rad)
