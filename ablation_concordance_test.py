import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path
import argparse
from collections import defaultdict
from mapper import Mapper
from fl import FL
import time
import csv

import torch
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

class ConcordanceAnalyzer:
    def __init__(self, args):
        self.args = args
        # Ensure the dictionary structure matches what run_ablation_test expects
        self.results = defaultdict(lambda: {"rounds": [], "config": {}})

    def compute_pairwise_omega(self, g1: torch.Tensor, g2: torch.Tensor) -> float:
        g1_flat = g1.flatten()
        g2_flat = g2.flatten()
        sign_prod = torch.sign(g1_flat) * torch.sign(g2_flat)
        return sign_prod.mean().item()

    def calculate_concordance_metrics(self, all_grads: List[torch.Tensor], num_benign: int) -> Dict[str, Any]:
        K = len(all_grads)
        if K == 0: return {}

        omega_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(i, K):
                val = self.compute_pairwise_omega(all_grads[i], all_grads[j])
                omega_matrix[i][j] = val
                omega_matrix[j][i] = val

        individual_rhos = []
        for k in range(K):
            sign_sum = np.sum(np.sign(omega_matrix[k]))
            rho_k = max(0.0, (1.0 / K) * sign_sum)
            individual_rhos.append(rho_k)

        benign_rhos = individual_rhos[:num_benign]
        byzantine_rhos = individual_rhos[num_benign:]

        # Fixed keys to match the logging requirements of run_ablation_test
        return {
            "all_rhos": individual_rhos,
            "benign_rhos": {
                "values": benign_rhos,
                "mean": float(np.mean(benign_rhos)) if benign_rhos else 0.0,
                "std": float(np.std(benign_rhos)) if benign_rhos else 0.0,
                "min": float(np.min(benign_rhos)) if benign_rhos else 0.0,
                "max": float(np.max(benign_rhos)) if benign_rhos else 0.0,
            },
            "byzantine_rhos": {
                "values": byzantine_rhos,
                "mean": float(np.mean(byzantine_rhos)) if byzantine_rhos else 0.0,
                "std": float(np.std(byzantine_rhos)) if byzantine_rhos else 0.0,
                "min": float(np.min(byzantine_rhos)) if byzantine_rhos else 0.0,
                "max": float(np.max(byzantine_rhos)) if byzantine_rhos else 0.0,
            }
        }

    def analyze_round(self, fl_coordinator, round_num: int, attack_name: str) -> Dict:
        all_grads = fl_coordinator.train() 
        num_benign = len(fl_coordinator.benign_clients)
        
        metrics = self.calculate_concordance_metrics(all_grads, num_benign)
        
        round_results = {
            "round": round_num,
            "epoch": float(fl_coordinator.epoch),
            "attack": attack_name,
            "avg_train_loss": float(fl_coordinator.avg_train_loss),
            "client_concordance_ratios": metrics,
            # This key is required by your print loop in run_ablation_test
            "benign_intra_concordance": {"mean": metrics["benign_rhos"]["mean"]},
        }

        # Add byzantine field if traitors exist so the print loop doesn't skip it
        if len(all_grads) > num_benign:
            round_results["byzantine_intra_concordance"] = {"mean": metrics["byzantine_rhos"]["mean"]}

        fl_coordinator.aggregate(all_grads)
        fl_coordinator.update_global_model()
        round_results["test_accuracy"] = float(fl_coordinator.evaluate_accuracy() * 100)

        return round_results

    def save_results(self, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results_file = os.path.join(output_dir, "concordance_results.json")
        with open(results_file, 'w') as f:
            json.dump(dict(self.results), f, indent=2)
        
        summary_file = os.path.join(output_dir, "concordance_summary.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Attack", "Avg_Benign_Rho", "Avg_Byzantine_Rho", "Final_Accuracy"])
            for name, data in self.results.items():
                if not data["rounds"]: continue
                last = data["rounds"][-1]
                writer.writerow([
                    name,
                    f"{last['client_concordance_ratios']['benign_rhos']['mean']:.4f}",
                    f"{last['client_concordance_ratios']['byzantine_rhos']['mean']:.4f}",
                    f"{last['test_accuracy']:.2f}"
                ])

    def print_summary(self, attack_name: str):
        if attack_name not in self.results or not self.results[attack_name]["rounds"]: return
        last = self.results[attack_name]["rounds"][-1]
        m = last["client_concordance_ratios"]
        print(f"\n--- {attack_name} Final Stats ---")
        print(f"Benign Rho: {m['benign_rhos']['mean']:.4f}")
        print(f"Byzantine Rho: {m['byzantine_rhos']['mean']:.4f}")
        print(f"Accuracy: {last['test_accuracy']:.2f}%\n")


def extract_scalar_from_args(args):
    """
    Extract scalar values from list parameters in args.
    Similar to how main.py handles grid search parameters.
    
    Args:
        args: Namespace with potentially list-valued parameters
        
    Returns:
        Modified args with scalar values
    """
    excluded_args = ['excluded_gpus', 'lr_decay']
    
    for arg in vars(args):
        if arg not in excluded_args:
            val = getattr(args, arg)
            if isinstance(val, list) and len(val) > 0:
                # Take first element from list
                setattr(args, arg, val[0])
    
    return args


def run_ablation_test(args: argparse.Namespace, attack_configs: List[Dict]) -> ConcordanceAnalyzer:
    """
    Run ablation test across multiple attack configurations.
    """
    analyzer = ConcordanceAnalyzer(args)
    
    # Set device based on availability
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id > -1 else 'cpu')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    for attack_config in attack_configs:
        print(f"\n{'#'*60}")
        print(f"Testing Attack: {attack_config['name']}")
        print(f"Configuration: {attack_config}")
        print(f"{'#'*60}")
        
        # Create a copy of args and update with attack-specific config
        test_args = argparse.Namespace(**vars(args))
        for key, value in attack_config.items():
            if key != 'name':
                setattr(test_args, key, value)
        
        # Initialize FL
        mapper = Mapper(test_args, device)
        fl = mapper.initialize_FL()
        
        num_benign = len(fl.benign_clients)
        num_malicious = len(fl.malicious_clients)
        
        print(f"Clients: {num_benign} Benign, {num_malicious} Malicious")
        print(f"Aggregator: {test_args.aggr}, Attack: {test_args.attack}")
        print("-" * 60)
        
        attack_results = []
        target_epochs = test_args.global_epoch
        
        for epoch in range(target_epochs):
            epoch_rounds = []
            target_epoch = int(fl.epoch) + 1
            
            # Train until we complete the epoch
            while int(fl.epoch) < target_epoch:
                round_num = len(attack_results)
                # This calls our new logic in analyze_round
                round_results = analyzer.analyze_round(fl, round_num, attack_config['name'])
                epoch_rounds.append(round_results)
                attack_results.append(round_results)
            
            # Print epoch summary
            if len(epoch_rounds) > 0:
                # Calculate epoch averages from the rounds
                avg_benign_rho = np.mean([r["client_concordance_ratios"]["benign_rhos"]["mean"] for r in epoch_rounds])
                last_acc = epoch_rounds[-1]["test_accuracy"]
                last_loss = epoch_rounds[-1]["avg_train_loss"]
                
                info_parts = [
                    f"Epoch {epoch + 1:3d}",
                    f"Acc: {last_acc:5.1f}%",
                    f"Loss: {last_loss:.4f}",
                    f"Avg_ρ_Benign: {avg_benign_rho:.4f}"
                ]
                
                # Check if there are Byzantine clients to report on
                if num_malicious > 0:
                    avg_byz_rho = np.mean([r["client_concordance_ratios"]["byzantine_rhos"]["mean"] for r in epoch_rounds])
                    info_parts.append(f"Avg_ρ_Byz: {avg_byz_rho:.4f}")
                
                print(" | ".join(info_parts))
            
            # Learning rate decay
            if epoch + 1 in test_args.lr_decay:
                fl.__update_lr__()
        
        # Store results for this attack
        analyzer.results[attack_config['name']]["rounds"] = attack_results
        analyzer.results[attack_config['name']]["config"] = attack_config
        
        # Print final summary for this attack configuration
        analyzer.print_summary(attack_config['name'])
    
    return analyzer


if __name__ == "__main__":
    from parameters import args_parser
    
    args = args_parser()
    
    # Extract scalar values using the same logic as main.py
    args = extract_scalar_from_args(args)
    
    # Check for available devices
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
        print('Using MPS (Metal Performance Shaders) device')
    else:
        device_type = 'cpu'
        print('No GPU found, using CPU device')
    
    # Override with ablation-specific defaults if needed
    args.trials = 1  # Single trial for ablation
    args.global_epoch = 10  # Fewer epochs for quick testing
    args.dataset_dist = 'iid'  # Set data distribution (fixed: was data_dist)
    args.dataset_name = "mnist"
    args.aggr = "avg"
    
    # Define attack configurations to test
    # Note: For baseline without attack, use traitor=0.0 with any attack type
    # (the attack won't be applied if there are no traitors)
    attack_configs = [
        #{"name": "NoAttack_Baseline", "attack": "label_flip", "traitor": 0.0},  # Baseline: no Byzantine clients
        {"name": "alie", "attack": "alie", "traitor": 0.2},
        {"name": "label_flip", "attack": "label_flip", "traitor": 0.2},
        {"name": "ipm", "attack": "ipm", "traitor": 0.2},
    ]
    
    print("Starting Ablation Test for Concordance Analysis")
    print(f"Device: {device_type.upper()}")
    print(f"Dataset: {args.dataset_name}, Model: {args.nn_name}")
    print(f"Aggregator: {args.aggr}")
    print(f"Number of attacks to test: {len(attack_configs)}")
    
    # Run ablation test
    analyzer = run_ablation_test(args, attack_configs)
    
    # Save results
    output_dir = f"./ablation_results/{args.dataset_name}_{args.nn_name}_{args.aggr}"
    analyzer.save_results(output_dir)
    
    print(f"\nAblation test complete! Results saved to {output_dir}")