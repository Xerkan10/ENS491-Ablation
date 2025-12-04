from sklearn.cluster import HDBSCAN
import torch
from .base import _BaseAggregator
import sklearn.metrics.pairwise as smp
import numpy as np
import math
class Flame(_BaseAggregator):
    def __init__(self,):
        super(Flame, self).__init__()
        self.momentum = None

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
        #inputs_np = [v.cpu().detach().numpy() for v in inputs]
        inputs_np = torch.stack(inputs).detach().cpu().numpy()
        cd = smp.cosine_distances(inputs_np)
        # = HDBSCAN(min_cluster_size=self.num_cluster).fit(cd)
        cluster = HDBSCAN(min_cluster_size=
                                  int(len(inputs) / 2 + 1),
                                  min_samples=1,  # gen_min_span_tree=True,
                                  allow_single_cluster=True, metric='precomputed').fit(cd)
        cluster_labels = (cluster.labels_).tolist()

        st = np.median(ed)
        for i in range(self.params.fl_no_models):
            if cluster_labels[i] == -1:
                continue

            update_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(update_name)
            if st / ed[i] < 1:
                for name, data in loaded_params.items():
                    if self.check_ignored_weights(name):
                        continue
                    data.mul_(st / ed[i])
            self.accumulate_weights(weight_accumulator, loaded_params)

        # Add noise
        for name, data in weight_accumulator.items():
            if 'running' in name or 'tracked' in name:
                continue
            self.add_noise(data, sigma=self.lamda * st)

        return torch.clone(self.momentum).detach()

    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if 'tracked' in name or 'running' in name:
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def add_noise(self, sum_update_tensor: torch.Tensor, sigma):
        noised_layer = torch.FloatTensor(sum_update_tensor.shape)
        noised_layer = noised_layer.to(self.device)
        noised_layer.normal_(mean=0, std=sigma)
        sum_update_tensor.add_(noised_layer)


