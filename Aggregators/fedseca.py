import torch
import torch.nn.functional as F
import numpy as np
from .base import _BaseAggregator


class FedSECA(_BaseAggregator):
    """
    FedSECA: Federated Sign-based Entropy Concordance Aggregation
    
    Implements CRISE (Clipping + Reputation + Sign) and ROCA (Reputation-Oriented 
    Concordance Aggregation) steps from the original FedSECA paper.
    
    Key components:
    1. Norm clipping & median clamping
    2. Top-k sparsification based on magnitude quantile
    3. Reputation scoring via sign-cosine similarity
    4. Sign-voted mean aggregation
    5. Momentum-based smoothing
    """
    
    def __init__(self, mom_beta=0.9, tm_gamma=0.1, device='cpu'):
        """
        Args:
            mom_beta: Momentum coefficient for aggregated gradient smoothing
            tm_gamma: Quantile threshold for top-k sparsification (0.0-1.0)
            device: Computation device
        """
        super(FedSECA, self).__init__()
        self.mom_beta = mom_beta
        self.tm_gamma = tm_gamma
        self.device = device
        
        # Momentum buffer (initialized on first call)
        self.mom_deltawvec = None
        self.repute_scores = None
        self.rounds = 0
    
    def safe_divide(self, nu, de, fill=1.0):
        """Safe division handling zero denominators."""
        res_like = de if (sum(nu.shape) < sum(de.shape)) else nu
        res = torch.full_like(res_like, fill_value=fill)
        
        try:
            nu, de = torch.broadcast_tensors(nu, de)
        except Exception as err:
            raise Exception(f"Incompatible shapes {de.shape}, {nu.shape} -- \n{err}")
        
        mask = (de != 0.0)
        res[mask] = torch.div(nu[mask], de[mask])
        return res
    
    def tensor_quantile(self, tnsr, q, dim=1):
        """Compute quantile along dimension (handles large tensors)."""
        numpy_tensor = tnsr.cpu().numpy()
        result = np.quantile(numpy_tensor, q, axis=dim)
        tnsr_result = torch.tensor(result).to(tnsr.device)
        return tnsr_result
    
    def _locwise_grad_clamper(self, xs):
        """
        Norm Clipping & Median Clamping
        
        Args:
            xs: Gradient vectors of shape [K, num_params]
        
        Returns:
            Clipped and clamped gradients
        """
        # Norm clipping: scale down vectors with norm > median norm
        norms = torch.norm(xs, dim=1).view(-1, 1)
        med_norm, _ = torch.median(norms, dim=0)
        norm_clip = med_norm / norms
        norm_clip[norm_clip > 1.0] = 1.0
        xs_clipped = xs * norm_clip.view(-1, 1)
        
        # Median clamping: clamp each coordinate to [-median_abs, +median_abs]
        med_mag, _ = torch.median(torch.abs(xs_clipped), dim=0)
        vs = torch.clamp(xs_clipped, max=med_mag, min=-med_mag)
        
        return vs
    
    def _grad_rating_score(self, sign_x):
        """
        Compute reputation scores based on sign-cosine similarity.
        
        For each client i:
            score_i = mean_j( sign( cosine_similarity(sign_x[i], sign_x[j]) ) )
        
        Then clamp to [0, inf) to gate negative (anti-correlated) clients.
        
        Args:
            sign_x: Sign vectors of gradients [K, num_params]
        
        Returns:
            Reputation scores [K, 1]
        """
        score_list = []
        for i in range(sign_x.shape[0]):
            # Cosine similarity between client i and all clients
            score = F.cosine_similarity(sign_x, sign_x[i].view(1, -1), dim=1)
            # Sign of similarity -> {-1, 0, +1}, then mean
            s = torch.sign(score).mean()
            score_list.append(s)
        
        current_repute = torch.vstack([s.view(1, 1) for s in score_list])
        repute = torch.clamp(current_repute, min=0)
        
        return repute
    
    def sign_voted_mean(self, dw):
        """
        Sign-voted mean aggregation with reputation weighting.
        
        Steps:
        1. Clip gradients (norm + median clamping)
        2. Top-k sparsification based on magnitude quantile
        3. Compute reputation from sign-cosine similarity
        4. Vote on sign direction weighted by reputation
        5. Select entries matching voted sign and average them
        
        Args:
            dw: Stacked gradient vectors [K, num_params]
        
        Returns:
            Aggregated gradient [1, num_params], reputation scores
        """
        x_raw = dw
        
        # Step 1: Clamp the gradients
        x = self._locwise_grad_clamper(x_raw)
        
        # Step 2: Top-k sparsification
        magn_xraw = torch.abs(x_raw)
        ql = self.tensor_quantile(magn_xraw, self.tm_gamma, dim=1)
        ql = ql.view(-1, 1)
        x[magn_xraw < ql] = 0.0
        
        # Step 3: Compute reputation from signs
        sign_xraw = torch.sign(x_raw)
        repute = self._grad_rating_score(sign_xraw)
        
        # Step 4: Vote on sign direction weighted by reputation
        sign_x = torch.sign(x)
        voted_sign = (sign_x * repute.view(-1, 1)).sum(dim=0).view(1, -1)
        
        # Step 5: Select entries matching voted sign
        disjoint_select = (0 < (x * voted_sign)).int()
        disjoint_x = x * disjoint_select
        disjoint_divisor = disjoint_select.sum(dim=0)
        
        # Compute mean of selected entries
        mean_dwvec = self.safe_divide(disjoint_x.sum(dim=0), disjoint_divisor, fill=0.0)
        
        return mean_dwvec.view(1, -1), repute.flatten().tolist()
    
    def __call__(self, inputs):
        """
        Aggregate gradients using FedSECA algorithm.
        
        Args:
            inputs: List of gradient tensors from clients
        
        Returns:
            Aggregated gradient tensor
        """
        # Stack inputs
        stacked = torch.stack(inputs, dim=0).to(self.device)
        
        # Compute sign-voted mean
        voted_mean, repute_info = self.sign_voted_mean(stacked)
        self.repute_scores = repute_info
        
        # Initialize momentum buffer on first call
        if self.mom_deltawvec is None:
            self.mom_deltawvec = torch.zeros_like(voted_mean).to(self.device)
        
        # Apply momentum
        self.mom_deltawvec = (1 - self.mom_beta) * voted_mean + \
                             self.mom_beta * self.mom_deltawvec
        
        self.rounds += 1
        
        return self.mom_deltawvec.view(-1)
    
    def get_attack_stats(self):
        """Return reputation scores for analysis."""
        return {
            'repute_scores': self.repute_scores,
            'rounds': self.rounds
        }


class FedSECA_NoMomentum(FedSECA):
    """FedSECA variant without momentum (for ablation studies)."""
    
    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0).to(self.device)
        voted_mean, repute_info = self.sign_voted_mean(stacked)
        self.repute_scores = repute_info
        self.rounds += 1
        return voted_mean.view(-1)
