"""
Implementation of Mimic attacks from "Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing"
Paper: https://arxiv.org/abs/2006.09365

The Mimic attack exploits the heterogeneity in datasets by identifying and mimicking 
the gradient pattern of a specific benign client. 

Implements:
1. MimicAttack - Simple mimic attack that copies a target client's gradient  
2. MimicVariantAttack - Advanced mimic attack that learns attack direction and targets optimal clients
"""
import torch
import numpy as np
from .base import _BaseByzantine
from utils import *


class MimicAttack(_BaseByzantine):
    """
    Simple Mimic attack that copies the gradient of a specific target client.
    
    This attack identifies a target benign client and copies their gradient updates.
    Effective in heterogeneous settings where mimicking a legitimate client
    can help the attack evade detection.
    """
    
    def __init__(self, n, m, target_rank=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.target_rank = target_rank if target_rank is not None else 0
        self._target_gradient = None
        
    def omniscient_callback(self, benign_gradients):
        """Copy the gradient of the target worker."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        # If target rank is beyond available benign gradients, use the first one
        target_idx = min(self.target_rank, len(benign_gradients) - 1)
        
        # Simply copy the target worker's gradient
        self._target_gradient = benign_gradients[target_idx].clone()
        self.adv_momentum = self._target_gradient.to(self.device)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None


class MimicVariantAttack(_BaseByzantine):
    """
    Advanced Mimic attack that learns an attack direction and targets optimal workers.
    
    This attack maintains a learned direction vector z and targets the benign worker
    whose gradient has the maximum (or minimum) inner product with this direction.
    The attack direction is updated over time to maximize effectiveness.
    """
    
    def __init__(self, n, m, warmup=10, z=None, eps=None, argmax=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.warmup = warmup  # Number of rounds to learn attack direction
        self.t = 0  # Current round counter
        self.mu = None  # Running average of benign gradients
        self.z = None  # Attack direction vector
        self.argmax = argmax  # True: maximize inner product, False: minimize
        self.target_rank = 0
        self._gradient = None
        
        # Initialize attack direction if provided
        if z is not None:
            self.initial_z_scale = z
        else:
            self.initial_z_scale = 1.0
            
    def _init_callback(self, benign_gradients, curr_avg):
        """Initialize the attack direction z using gradient information."""
        self.mu = curr_avg.clone()
        
        # Initialize z as a random direction
        device = curr_avg.device
        torch.manual_seed(0)  # For reproducibility
        self.z = torch.randn_like(curr_avg, device=device)
        self.z = self.z / (self.z.norm() + 1e-8)
        
        # Refine z based on gradient patterns
        cumu = torch.zeros_like(curr_avg)
        for g in benign_gradients:
            w = (g - self.mu).dot(self.z)
            cumu += w * (g - self.mu)
            
        if cumu.norm() > 1e-8:
            self.z = cumu / cumu.norm()
        
    def _warmup_callback(self, benign_gradients, curr_avg):
        """Update the attack direction during warmup phase."""
        # Update running average
        alpha = self.t / (1 + self.t)
        self.mu = alpha * self.mu + (1 - alpha) * curr_avg
        
        # Update attack direction
        cumu = torch.zeros_like(curr_avg)
        for g in benign_gradients:
            w = (g - self.mu).dot(self.z)
            cumu += w * (g - self.mu)
            
        if cumu.norm() > 1e-8:
            z_update = cumu / cumu.norm()
            self.z = alpha * self.z + (1 - alpha) * z_update
            self.z = self.z / (self.z.norm() + 1e-8)
        
    def _attack_callback(self, benign_gradients):
        """Select the target worker based on inner product with attack direction."""
        best_value = None
        best_idx = 0
        best_gradient = None
        
        for i, g in enumerate(benign_gradients):
            inner_product = g.dot(self.z)
            
            if self.argmax:
                # Maximize inner product
                if best_value is None or inner_product > best_value:
                    best_value = inner_product
                    best_idx = i
                    best_gradient = g
            else:
                # Minimize inner product  
                if best_value is None or inner_product < best_value:
                    best_value = inner_product
                    best_idx = i
                    best_gradient = g
                    
        return best_value, best_idx, best_gradient
        
    def omniscient_callback(self, benign_gradients):
        """Generate malicious updates using the mimic variant strategy."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        # Compute current average
        curr_avg = torch.mean(torch.stack(benign_gradients), dim=0)
        
        # Update attack direction based on current phase
        if self.t == 0:
            self._init_callback(benign_gradients, curr_avg)
        elif self.t < self.warmup:
            self._warmup_callback(benign_gradients, curr_avg)
            
        # Select target and generate attack
        if self.t < self.warmup:
            # During warmup: learn and attack simultaneously
            best_value, best_idx, target_gradient = self._attack_callback(benign_gradients)
            self._gradient = target_gradient.clone()
        else:
            # After warmup: use fixed target or continue learning
            best_value, best_idx, target_gradient = self._attack_callback(benign_gradients)
            self._gradient = target_gradient.clone()
            
        self.adv_momentum = self._gradient.to(self.device)
        self.t += 1
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None


class AdaptiveMimicAttack(_BaseByzantine):
    """
    Adaptive Mimic attack that dynamically adjusts its strategy based on aggregation patterns.
    
    This variant of the mimic attack adapts its target selection and mimicking strategy
    based on the observed effectiveness and aggregation patterns.
    """
    
    def __init__(self, n, m, z=None, eps=None, adaptation_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_good = n - m
        self.m = m
        self.adaptation_rate = adaptation_rate
        self.target_weights = None  # Weights for targeting different workers
        self.success_history = []
        self.current_target = 0
        
    def _initialize_target_weights(self, num_workers):
        """Initialize uniform target weights."""
        if self.target_weights is None:
            self.target_weights = torch.ones(num_workers) / num_workers
            
    def _update_target_weights(self, benign_gradients, selected_idx):
        """Update target weights based on gradient diversity and selection success."""
        if self.target_weights is None:
            return
            
        # Compute gradient diversity for each worker
        avg_grad = torch.mean(torch.stack(benign_gradients), dim=0)
        diversities = []
        
        for g in benign_gradients:
            diversity = torch.norm(g - avg_grad).item()
            diversities.append(diversity)
            
        # Update weights: favor workers with higher diversity
        diversities = torch.tensor(diversities)
        diversity_weights = torch.softmax(diversities / (diversities.std() + 1e-8), dim=0)
        
        # Blend with current weights
        self.target_weights = (1 - self.adaptation_rate) * self.target_weights + \
                             self.adaptation_rate * diversity_weights
                             
    def omniscient_callback(self, benign_gradients):
        """Generate adaptive malicious updates."""
        if len(benign_gradients) == 0:
            self.adv_momentum = torch.zeros_like(self.momentum)
            return
            
        num_workers = len(benign_gradients)
        self._initialize_target_weights(num_workers)
        
        # Select target based on current weights
        target_probs = self.target_weights[:num_workers]
        target_probs = target_probs / target_probs.sum()  # Normalize
        
        # Sample target worker
        self.current_target = torch.multinomial(target_probs, 1).item()
        
        # Use the selected target's gradient
        target_gradient = benign_gradients[self.current_target]
        self.adv_momentum = target_gradient.clone().to(self.device)
        
        # Update target weights for next round
        self._update_target_weights(benign_gradients, self.current_target)
        
    def local_step(self, batch):
        return None
        
    def train_(self, embd_momentum=None):
        return None
