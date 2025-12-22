import torch

class TCS(object):
    def __init__(self, g_mask=0.01,local_mask=0.001,*args):
        super(TCS, self).__init__()
        self.global_mask = None
        self.local_aggr_locs = None
        self.all_locs = None
        self.global_mask_rate = g_mask
        self.local_mask_rate = local_mask
        self.device = None
        self.model_size = None

    def __call__(self, inputs):
        if self.global_mask is None:
            self.make_globa_mask(inputs)
            aggr,locs = self.global_mask_aggr(inputs)
            self.all_locs = locs
            #aggr_inputs = [input[self.all_locs] for input in inputs]
            aggr_inputs = [torch.zeros_like(input).to(self.device) for input in inputs]
            for x, input in zip(aggr_inputs,inputs):
                x[self.all_locs] = input[self.all_locs]
            return aggr_inputs
        else:
            aggr_global, locs_global = self.global_mask_aggr(inputs)
            aggr_local,loc_local = self.local_mask_aggr(inputs)
            aggr = aggr_global + aggr_local
            self.all_locs = torch.cat([locs_global,loc_local],dim=0)
            aggr_inputs = [torch.zeros_like(input).to(self.device) for input in inputs]
            for x, input in zip(aggr_inputs,inputs):
                x[self.all_locs] = input[self.all_locs]
            self.make_globa_mask(inputs)
            return aggr_inputs


    def __refactor__(self,aggr):
        vec = torch.zeros(self.model_size).to(self.device)
        vec[self.all_locs] = aggr
        return vec

    def make_globa_mask(self,inputs):
        self.device = inputs[0].device
        if self.global_mask is None:
            size = int(inputs[0].numel() * self.global_mask_rate + inputs[0].numel() * self.local_mask_rate)
            self.model_size = inputs[0].numel()
        else:
            size = int(inputs[0].numel() * self.global_mask_rate)
        cat = torch.stack(inputs, dim=0)
        _, inds = torch.topk(cat.abs(),k=size,dim=1)
        mask = torch.zeros_like(cat[0]).to(self.device)
        for ind in inds:
            mask[ind] = 1
        size_final = int(inputs[0].numel() * self.global_mask_rate)
        _,f_inds = torch.topk(mask,k=size_final)
        mask *=0
        mask[f_inds] = 1
        self.global_mask = mask
        return

    def local_mask_aggr(self,inputs):
        device = self.device
        size = int(self.model_size * self.local_mask_rate)
        cat = torch.stack(inputs, dim=0)
        global_mask = self.global_mask.unsqueeze(0).repeat(cat.size(0),1)
        _, inds = torch.topk((cat*1-global_mask).abs(),k=size,dim=1)
        vec = torch.zeros_like(cat[0]).to(device)
        for ind,input in zip(inds,inputs):
            vec[ind] += input[ind]
        vec *= 1/len(inputs)
        locs = vec.nonzero().squeeze()
        self.local_aggr_locs = locs
        return vec,locs

    def global_mask_aggr(self,inputs):
        device = inputs[0].device
        vec = torch.zeros_like(inputs[0]).to(device)
        for input in inputs:
            vec[self.global_mask.bool()] += input[self.global_mask.bool()]
        vec = vec * 1/len(inputs)
        locs = self.global_mask.nonzero().squeeze()
        return vec,locs


if __name__ == '__main__':
    dummy = [torch.rand(10000) for _ in range(10)]
    tcs = TCS()
    vec = tcs.__call__(dummy)
    vec = sum(vec) / len(vec)
    aggr_vec = tcs.__refactor__(vec)
    print(aggr_vec.shape)
    dummy = [torch.rand(10000) for _ in range(10)]
    vec = tcs.__call__(dummy)
    vec = sum(vec) / len(vec)
    aggr_vec = tcs.__refactor__(vec)
    print(aggr_vec.shape)