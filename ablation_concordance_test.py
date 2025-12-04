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

class ConcordanceAnalyzer:
    """
    Analyzer for computing concordance ratios between client gradients.
    
    Concordance ratio measures how similar gradient directions are between clients,
    useful for detecting Byzantine behavior and analyzing attack effectiveness.
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results = defaultdict(lambda: defaultdict(list))
        
    def compute_pairwise_concordance(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """
        Compute sign concordance ω between two gradient vectors.
        
        Formula (4) from paper: ω(g1, g2) = (1/D) * Σ sgn(g1^j) · sgn(g2^j)
        
        This measures the fraction of dimensions where gradients have the same sign.
        
        Args:
            grad1: First gradient tensor
            grad2: Second gradient tensor
            
        Returns:
            Sign concordance ω between -1 and 1
        """
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()
        
        # Compute sign concordance: fraction of dimensions with same sign
        sign_product = torch.sign(grad1_flat) * torch.sign(grad2_flat)
        concordance = sign_product.mean().item()
        
        return concordance
    
    def compute_client_concordance_ratio(self, client_grad: torch.Tensor, all_grads: List[torch.Tensor], client_idx: int) -> float:
        """
        Compute concordance ratio ρ_k for a specific client k.
        
        Formula (5) from paper: ρ_k = max[0, (1/K) * Σ_{ℓ∈K} sgn(ω(g_k, g_ℓ))]
        
        This measures how well client k's gradient aligns with the majority.
        Higher ρ_k indicates honest behavior, lower indicates potential Byzantine.
        
        Args:
            client_grad: Gradient tensor of client k
            all_grads: List of all client gradient tensors
            client_idx: Index of client k in all_grads
            
        Returns:
            Concordance ratio ρ_k between 0 and 1
        """
        K = len(all_grads)
        if K <= 1:
            return 0.0
        
        # Sum of sign(ω(g_k, g_ℓ)) for all other clients ℓ
        sign_sum = 0.0
        for l_idx, grad_l in enumerate(all_grads):
            if l_idx != client_idx:
                omega = self.compute_pairwise_concordance(client_grad, grad_l)
                sign_sum += np.sign(omega)
        
        # ρ_k = max[0, (1/K) * Σ sgn(ω)]
        rho_k = max(0.0, sign_sum / K)
        
        return rho_k
    
    def compute_group_concordance(self, grads: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute pairwise sign concordance ω statistics within a group of gradients.
        
        Args:
            grads: List of gradient tensors
            
        Returns:
            Dictionary with mean, std, min, and max concordance values
        """
        if len(grads) < 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        concordances = []
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                conc = self.compute_pairwise_concordance(grads[i], grads[j])
                concordances.append(conc)
        
        return {
            "mean": float(np.mean(concordances)),
            "std": float(np.std(concordances)),
            "min": float(np.min(concordances)),
            "max": float(np.max(concordances)),
            "count": len(concordances)
        }
    
    def compute_all_client_concordance_ratios(self, all_grads: List[torch.Tensor], 
                                               num_benign: int) -> Dict[str, Any]:
        """
        Compute concordance ratio ρ_k for all clients and categorize by type.
        
        Args:
            all_grads: List of all client gradient tensors (benign first, then Byzantine)
            num_benign: Number of benign clients
            
        Returns:
            Dictionary with per-client ρ values and statistics for benign/Byzantine groups
        """
        all_rhos = []
        for k in range(len(all_grads)):
            rho_k = self.compute_client_concordance_ratio(all_grads[k], all_grads, k)
            all_rhos.append(rho_k)
        
        benign_rhos = all_rhos[:num_benign]
        byzantine_rhos = all_rhos[num_benign:] if num_benign < len(all_grads) else []
        
        result = {
            "all_rhos": all_rhos,
            "benign_rhos": {
                "values": benign_rhos,
                "mean": float(np.mean(benign_rhos)) if benign_rhos else 0.0,
                "std": float(np.std(benign_rhos)) if benign_rhos else 0.0,
                "min": float(np.min(benign_rhos)) if benign_rhos else 0.0,
                "max": float(np.max(benign_rhos)) if benign_rhos else 0.0,
            }
        }
        
        if byzantine_rhos:
            result["byzantine_rhos"] = {
                "values": byzantine_rhos,
                "mean": float(np.mean(byzantine_rhos)),
                "std": float(np.std(byzantine_rhos)),
                "min": float(np.min(byzantine_rhos)),
                "max": float(np.max(byzantine_rhos)),
            }
        
        return result
    
    def compute_cross_group_concordance(self, benign_grads: List[torch.Tensor], 
                                       byzantine_grads: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute concordance between benign and Byzantine groups.
        
        Args:
            benign_grads: List of benign client gradients
            byzantine_grads: List of Byzantine client gradients
            
        Returns:
            Dictionary with mean, std, min, and max cross-group concordance values
        """
        if len(benign_grads) == 0 or len(byzantine_grads) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        concordances = []
        for benign_grad in benign_grads:
            for byzantine_grad in byzantine_grads:
                conc = self.compute_pairwise_concordance(benign_grad, byzantine_grad)
                concordances.append(conc)
        
        return {
            "mean": float(np.mean(concordances)),
            "std": float(np.std(concordances)),
            "min": float(np.min(concordances)),
            "max": float(np.max(concordances)),
            "count": len(concordances)
        }
    
    def analyze_round(self, fl_coordinator: FL, round_num: int, attack_name: str) -> Dict:
        """
        Analyze concordance for a single FL round.
        
        Args:
            fl_coordinator: FL coordinator instance
            round_num: Current round number
            attack_name: Name of the attack being tested
            
        Returns:
            Dictionary containing concordance analysis results
        """
        # Get gradients from training
        all_grads = fl_coordinator.train()
        
        # Separate benign and Byzantine gradients
        num_benign = len(fl_coordinator.benign_clients)
        benign_grads = all_grads[:num_benign]
        byzantine_grads = all_grads[num_benign:]
        
        # Compute concordance metrics for benign clients
        benign_concordance = self.compute_group_concordance(benign_grads)
        
        # Compute per-client concordance ratios ρ_k (key metric from paper Formula 5)
        client_rhos = self.compute_all_client_concordance_ratios(all_grads, num_benign)
        
        results = {
            "round": round_num,
            "epoch": float(fl_coordinator.epoch),
            "attack": attack_name,
            "benign_intra_concordance": benign_concordance,
            "client_concordance_ratios": client_rhos,  # Per-client ρ_k values
            "avg_train_loss": float(fl_coordinator.avg_train_loss),
            "num_diverged": float(fl_coordinator.num_diverged),
        }
        
        # Compute concordance metrics for Byzantine clients (if any)
        if len(byzantine_grads) > 0:
            byzantine_concordance = self.compute_group_concordance(byzantine_grads)
            cross_concordance = self.compute_cross_group_concordance(benign_grads, byzantine_grads)
            
            results.update({
                "byzantine_intra_concordance": byzantine_concordance,
                "benign_byzantine_cross_concordance": cross_concordance,
            })
        
        # Perform aggregation
        fl_coordinator.aggregate(all_grads)
        fl_coordinator.update_global_model()
        
        # Evaluate accuracy (returns 0-1, convert to percentage)
        test_acc = fl_coordinator.evaluate_accuracy()
        results["test_accuracy"] = float(test_acc * 100) if test_acc is not None else 0.0
        
        return results
    
    def save_results(self, output_dir: str = "./ablation_results"):
        """
        Save concordance analysis results to JSON files.
        
        Args:
            output_dir: Directory to save results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, "concordance_results.json")
        with open(results_file, 'w') as f:
            json.dump(dict(self.results), f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Create a summary CSV for easy analysis
        self.save_summary_csv(output_dir)
        
    def save_summary_csv(self, output_dir: str):
        """
        Save a summary CSV with key metrics per attack.
        
        Args:
            output_dir: Directory to save results
        """
        import csv
        
        summary_file = os.path.join(output_dir, "concordance_summary.csv")
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Attack", "Avg_Benign_Concordance", "Std_Benign_Concordance",
                "Avg_Byzantine_Concordance", "Std_Byzantine_Concordance",
                "Avg_Cross_Concordance", "Std_Cross_Concordance",
                "Final_Accuracy"
            ])
            
            for attack_name, attack_data in self.results.items():
                rounds_data = attack_data["rounds"]
                
                benign_conc = [r["benign_intra_concordance"]["mean"] for r in rounds_data]
                final_acc = rounds_data[-1]["test_accuracy"]
                
                if "byzantine_intra_concordance" in rounds_data[0]:
                    byz_conc = [r["byzantine_intra_concordance"]["mean"] for r in rounds_data]
                    cross_conc = [r["benign_byzantine_cross_concordance"]["mean"] for r in rounds_data]
                    
                    writer.writerow([
                        attack_name,
                        f"{np.mean(benign_conc):.4f}",
                        f"{np.std(benign_conc):.4f}",
                        f"{np.mean(byz_conc):.4f}",
                        f"{np.std(byz_conc):.4f}",
                        f"{np.mean(cross_conc):.4f}",
                        f"{np.std(cross_conc):.4f}",
                        f"{final_acc:.2f}"
                    ])
                else:
                    writer.writerow([
                        attack_name,
                        f"{np.mean(benign_conc):.4f}",
                        f"{np.std(benign_conc):.4f}",
                        "N/A", "N/A", "N/A", "N/A",
                        f"{final_acc:.2f}"
                    ])
        
        print(f"Summary CSV saved to {summary_file}")
        
    def print_summary(self, attack_name: str):
        """
        Print summary statistics for an attack.
        
        Args:
            attack_name: Name of the attack
        """
        if attack_name not in self.results:
            return
        
        rounds_data = self.results[attack_name]["rounds"]
        
        print(f"\n{'='*60}")
        print(f"Attack: {attack_name}")
        print(f"{'='*60}")
        
        # Compute averages across rounds
        benign_conc = [r["benign_intra_concordance"]["mean"] for r in rounds_data]
        test_accs = [r["test_accuracy"] for r in rounds_data]
        
        print(f"Benign Intra-Concordance:")
        print(f"  Mean: {np.mean(benign_conc):.4f} ± {np.std(benign_conc):.4f}")
        print(f"  Range: [{np.min(benign_conc):.4f}, {np.max(benign_conc):.4f}]")
        print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%")
        
        # Print per-client concordance ratio ρ_k statistics (key metric from paper)
        if "client_concordance_ratios" in rounds_data[-1]:
            last_rhos = rounds_data[-1]["client_concordance_ratios"]
            benign_rho_stats = last_rhos["benign_rhos"]
            print(f"\nPer-Client Concordance Ratio ρ_k (Formula 5):")
            print(f"  Benign clients ρ: {benign_rho_stats['mean']:.4f} ± {benign_rho_stats['std']:.4f}")
            print(f"  Benign ρ range: [{benign_rho_stats['min']:.4f}, {benign_rho_stats['max']:.4f}]")
            
            if "byzantine_rhos" in last_rhos:
                byz_rho_stats = last_rhos["byzantine_rhos"]
                print(f"  Byzantine clients ρ: {byz_rho_stats['mean']:.4f} ± {byz_rho_stats['std']:.4f}")
                print(f"  Byzantine ρ range: [{byz_rho_stats['min']:.4f}, {byz_rho_stats['max']:.4f}]")
        
        if len(rounds_data) > 0 and "byzantine_intra_concordance" in rounds_data[0]:
            byz_conc = [r["byzantine_intra_concordance"]["mean"] for r in rounds_data]
            cross_conc = [r["benign_byzantine_cross_concordance"]["mean"] for r in rounds_data]
            
            print(f"\nByzantine Intra-Concordance (ω):")
            print(f"  Mean: {np.mean(byz_conc):.4f} ± {np.std(byz_conc):.4f}")
            print(f"  Range: [{np.min(byz_conc):.4f}, {np.max(byz_conc):.4f}]")
            
            print(f"\nBenign-Byzantine Cross-Concordance (ω):")
            print(f"  Mean: {np.mean(cross_conc):.4f} ± {np.std(cross_conc):.4f}")
            print(f"  Range: [{np.min(cross_conc):.4f}, {np.max(cross_conc):.4f}]")


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
    
    Args:
        args: Base configuration arguments
        attack_configs: List of attack configuration dictionaries
        
    Returns:
        ConcordanceAnalyzer with collected results
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
        
        # Use Mapper to initialize FL (same as in run.py)
        mapper = Mapper(test_args, device)
        fl = mapper.initialize_FL()
        
        print(f"Initialized with {len(fl.benign_clients)} benign and {len(fl.malicious_clients)} malicious clients")
        print(f"Aggregator: {test_args.aggr}, Attack: {test_args.attack}")
        print("-" * 60)
        
        # Run training rounds and collect concordance data
        attack_results = []
        target_epochs = test_args.global_epoch
        
        for epoch in range(target_epochs):
            epoch_rounds = []
            target_epoch = int(fl.epoch) + 1
            
            # Train until we complete the epoch
            while int(fl.epoch) < target_epoch:
                round_num = len(attack_results)
                round_results = analyzer.analyze_round(fl, round_num, attack_config['name'])
                epoch_rounds.append(round_results)
                attack_results.append(round_results)
            
            # Print epoch summary
            if len(epoch_rounds) > 0:
                avg_benign_conc = np.mean([r["benign_intra_concordance"]["mean"] for r in epoch_rounds])
                last_acc = epoch_rounds[-1]["test_accuracy"]
                last_loss = epoch_rounds[-1]["avg_train_loss"]
                
                # Get per-client concordance ratio ρ_k stats (key metric)
                last_rhos = epoch_rounds[-1]["client_concordance_ratios"]
                benign_rho_mean = last_rhos["benign_rhos"]["mean"]
                
                info_parts = [
                    f"Epoch {epoch + 1:3d}",
                    f"Acc: {last_acc:5.1f}%",
                    f"Loss: {last_loss:.4f}",
                    f"ω_benign: {avg_benign_conc:.4f}",
                    f"ρ_benign: {benign_rho_mean:.4f}"
                ]
                
                if "byzantine_intra_concordance" in epoch_rounds[-1]:
                    avg_byz_conc = np.mean([r["byzantine_intra_concordance"]["mean"] for r in epoch_rounds])
                    cross_conc = epoch_rounds[-1]["benign_byzantine_cross_concordance"]["mean"]
                    byz_rho_mean = last_rhos.get("byzantine_rhos", {}).get("mean", 0.0)
                    info_parts.append(f"ω_byz: {avg_byz_conc:.4f}")
                    info_parts.append(f"ρ_byz: {byz_rho_mean:.4f}")
                
                print(" | ".join(info_parts))
            
            # Learning rate decay
            if epoch + 1 in test_args.lr_decay:
                fl.__update_lr__()
        
        analyzer.results[attack_config['name']]["rounds"] = attack_results
        analyzer.results[attack_config['name']]["config"] = attack_config
        
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