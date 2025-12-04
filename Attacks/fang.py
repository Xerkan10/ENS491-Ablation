from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np

class fang(_BaseByzantine): ## This uses too much memory and computationally heavy.
    def __init__(self,n,m,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        deviation = torch.sign(mu).to(self.device)
        stack2 = torch.stack(benign_gradients, 0)
        m = self.get_malicious_updates_fang(stack2, mu,deviation, self.m)
        self.adv_momentum = m


    def local_step(self,batch):
        return None

    def train_(self, embd_momentum=None):
        return None

    def compute_lambda_fang(self,all_updates, model_re, n_attackers):

        distances = []
        n_benign, d = all_updates.shape
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1)
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances[distances == 0] = 10000
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
        min_score = torch.min(scores)
        term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
        max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

        return (term_1 + max_wre_dist)

    def get_malicious_updates_fang(self,all_updates, model_re, deviation, n_attackers):

        lamda = self.compute_lambda_fang(all_updates, model_re, n_attackers)
        threshold = 1e-5

        mal_updates = []
        while lamda > threshold:
            mal_update = (- lamda * deviation)

            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads, krum_candidate = self.multi_krum(mal_updates, n_attackers, multi_k=False)

            if krum_candidate < n_attackers:
                return mal_updates

            lamda *= 0.5

        if not len(mal_updates):
            print(lamda, threshold)
            mal_update = (model_re - lamda * deviation)

        return mal_update

    def multi_krum(self,all_updates, n_attackers, multi_k=False):

        candidates = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(remaining_updates) > 2 * n_attackers + 2:
            torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat(
                (candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not multi_k:
                break
        # print(len(remaining_updates))

        aggregate = torch.mean(candidates, dim=0)

        return aggregate, np.array(candidate_indices)
