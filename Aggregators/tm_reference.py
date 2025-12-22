import torch
from .base import _BaseAggregator
import numpy as np

class TM_refence(_BaseAggregator):
    def __init__(self, b):
        self.b = b
        super(TM_refence, self).__init__()
        self.tm_bypassed = 0
        self.tm_locs = None
        self.rounds = 0
        self.last_aggregated = None

    def __call__(self, inputs):
        if len(inputs) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError
        byz = len(inputs) - b
        stacked = torch.stack(inputs, dim=0)
        sorted_grads,_ = torch.sort(stacked,dim=0)
        outlier_discarded = sorted_grads[:][b:-b]
        #print(outlier_discarded)
        mean = torch.mean(outlier_discarded,dim=0)
        if self.last_aggregated is None:
            res = mean
        else:
            prev_update = torch.Tensor.repeat(self.last_aggregated,(len(inputs),1))
            dists = stacked - prev_update
            _, new_sort_inds = torch.sort(dists,dim=0)
            selected_inds = new_sort_inds[b:-b][:]
            col_indices = torch.arange(len(inputs[0])).expand(len(selected_inds), len(inputs[0]))
            selected = stacked[selected_inds,col_indices] 
            res = torch.mean(selected,dim=0)
        self.last_aggregated = res.clone()
        self.rounds +=1
        return res
    
if __name__ == "__main__": 
    # Test the trimmed mean aggregator
    def test_trimmed_mean():
        # Create sample inputs
        inputs = [
            torch.tensor([0, 2.0, 3.0, 4.0]),
            torch.tensor([-2.0, 3.0, 4.0, 5.0]),
            torch.tensor([0.5, 1.5, 2.5, 3.5]),
            torch.tensor([3.0, 4.0, -5.0, 6.0]),
            torch.tensor([1.5, 2.5, 3.5, 4.5]),
            torch.tensor([2.0, -3.0, 4.0, 5.0]),
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
        ]
        #print(torch.sort(torch.stack(inputs,dim=0),dim=0)[0])
        
        # Initialize trimmed mean with b=1 (trim 1 from each end)
        tm = TM_refence(b=1)
        
        # Test aggregation
        result = tm(inputs)
        results = tm(inputs)
        print(f"Trimmed mean result: {result}")
        print(f"Trimmed mean result2: {results}")
        print(f"Rounds completed: {tm.rounds}")
    test_trimmed_mean()
        
