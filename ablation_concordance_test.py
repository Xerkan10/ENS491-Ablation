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
import matplotlib.pyplot as plt

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
        
        Computes pairwise concordance of client k with ALL other clients,
        then returns the average.
        
        Args:
            client_grad: Gradient tensor of client k
            all_grads: List of all client gradient tensors
            client_idx: Index of client k in all_grads
            
        Returns:
            Average concordance ratio for client k
        """
        K = len(all_grads)
        if K <= 1:
            return 0.0
        
        # Compute pairwise concordance with all other clients and average
        concordances = []
        for l_idx, grad_l in enumerate(all_grads):
            if l_idx != client_idx:
                omega = self.compute_pairwise_concordance(client_grad, grad_l)
                concordances.append(omega)
        
        # Return average concordance
        return float(np.mean(concordances)) if concordances else 0.0
    
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
        Compute concordance ratio for all clients and categorize by type.
        
        For each client k:
            1. Compute pairwise concordance with ALL other clients
            2. Average to get individual concordance ratio ρ_k
        
        For benign/Byzantine groups:
            Average the individual concordance ratios of clients in that group
        
        Args:
            all_grads: List of all client gradient tensors (benign first, then Byzantine)
            num_benign: Number of benign clients
            
        Returns:
            Dictionary with per-client concordance values and group statistics
        """
        # Step 1: Compute individual client concordance ratios
        # Each client's concordance = average pairwise concordance with all other clients
        all_rhos = []
        for k in range(len(all_grads)):
            rho_k = self.compute_client_concordance_ratio(all_grads[k], all_grads, k)
            all_rhos.append(rho_k)
        
        # Step 2: Separate into benign and Byzantine groups
        benign_rhos = all_rhos[:num_benign]
        byzantine_rhos = all_rhos[num_benign:] if num_benign < len(all_grads) else []
        
        # Step 3: Compute group concordance = average of individual concordance ratios
        benign_group_avg = float(np.mean(benign_rhos)) if benign_rhos else 0.0
        
        result = {
            "all_rhos": all_rhos,
            "benign_rhos": {
                "values": benign_rhos,
                "mean": benign_group_avg,
                "std": float(np.std(benign_rhos)) if benign_rhos else 0.0,
                "min": float(np.min(benign_rhos)) if benign_rhos else 0.0,
                "max": float(np.max(benign_rhos)) if benign_rhos else 0.0,
                "group_avg": benign_group_avg,  # Same as mean - average of individual client concordances
            }
        }
        
        if byzantine_rhos:
            byzantine_group_avg = float(np.mean(byzantine_rhos))
            
            result["byzantine_rhos"] = {
                "values": byzantine_rhos,
                "mean": byzantine_group_avg,
                "std": float(np.std(byzantine_rhos)),
                "min": float(np.min(byzantine_rhos)),
                "max": float(np.max(byzantine_rhos)),
                "group_avg": byzantine_group_avg,  # Same as mean - average of individual client concordances
            }
        
        return result
    
    
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
        
        # Compute concordance ratios for each group
        # Each group's concordance = average pairwise concordance of that group's clients with ALL clients
        client_concordances = self.compute_all_client_concordance_ratios(all_grads, num_benign)
        
        results = {
            "round": round_num,
            "epoch": float(fl_coordinator.epoch),
            "attack": attack_name,
            "client_concordance_ratios": client_concordances,
            "benign_concordance": client_concordances["benign_rhos"]["group_avg"],  # Single value for benign group
            "avg_train_loss": float(fl_coordinator.avg_train_loss),
            "num_diverged": float(fl_coordinator.num_diverged),
        }
        
        # Add Byzantine concordance if there are Byzantine clients
        if "byzantine_rhos" in client_concordances:
            results["byzantine_concordance"] = client_concordances["byzantine_rhos"]["group_avg"]  # Single value for Byzantine group
        
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
                "Attack", "Benign_Concordance", "Byzantine_Concordance", "Final_Accuracy"
            ])
            
            for attack_name, attack_data in self.results.items():
                rounds_data = attack_data["rounds"]
                
                # Get final round concordance values
                final_benign_conc = rounds_data[-1]["benign_concordance"]
                final_acc = rounds_data[-1]["test_accuracy"]
                
                if "byzantine_concordance" in rounds_data[-1]:
                    final_byz_conc = rounds_data[-1]["byzantine_concordance"]
                    writer.writerow([
                        attack_name,
                        f"{final_benign_conc:.4f}",
                        f"{final_byz_conc:.4f}",
                        f"{final_acc:.2f}"
                    ])
                else:
                    writer.writerow([
                        attack_name,
                        f"{final_benign_conc:.4f}",
                        "N/A",
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
        
        # Get final round metrics
        final_round = rounds_data[-1]
        final_acc = final_round["test_accuracy"]
        benign_conc = final_round["benign_concordance"]
        
        print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
        print(f"\nConcordance Ratios (avg pairwise concordance with all clients):")
        print(f"  Benign group: {benign_conc:.4f}")
        
        if "byzantine_concordance" in final_round:
            byz_conc = final_round["byzantine_concordance"]
            print(f"  Byzantine group: {byz_conc:.4f}")
        
        # Print individual client concordance values
        if "client_concordance_ratios" in final_round:
            rhos = final_round["client_concordance_ratios"]
            print(f"\nIndividual Client Concordances:")
            print(f"  Benign clients: {rhos['benign_rhos']['values']}")
            if "byzantine_rhos" in rhos:
                print(f"  Byzantine clients: {rhos['byzantine_rhos']['values']}")
    
    def plot_concordance_distribution(self, output_dir: str = "./ablation_results"):
        """
        Plot the distribution of concordance ratios for all clients.
        Creates separate plots for each attack showing benign vs Byzantine client distributions.
        
        Args:
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for attack_name, attack_data in self.results.items():
            if "rounds" not in attack_data or len(attack_data["rounds"]) == 0:
                continue
            
            # Get the last round's concordance ratios
            last_round = attack_data["rounds"][-1]
            if "client_concordance_ratios" not in last_round:
                continue
            
            rhos_data = last_round["client_concordance_ratios"]
            benign_rhos = rhos_data["benign_rhos"]["values"]
            byzantine_rhos = rhos_data.get("byzantine_rhos", {}).get("values", [])
            all_rhos = rhos_data["all_rhos"]
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Concordance Ratio Distribution - {attack_name}', fontsize=14, fontweight='bold')
            
            # Plot 1: Bar chart showing individual client concordance ratios
            ax1 = axes[0]
            num_benign = len(benign_rhos)
            num_byzantine = len(byzantine_rhos)
            total_clients = num_benign + num_byzantine
            
            colors = ['green'] * num_benign + ['red'] * num_byzantine
            client_indices = list(range(total_clients))
            
            bars = ax1.bar(client_indices, all_rhos, color=colors, alpha=0.7, edgecolor='black')
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Client Index', fontsize=11)
            ax1.set_ylabel('Concordance Ratio', fontsize=11)
            ax1.set_title('Individual Client Concordance Ratios', fontsize=12)
            ax1.set_xticks(client_indices)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', alpha=0.7, edgecolor='black', label=f'Benign ({num_benign})'),
                             Patch(facecolor='red', alpha=0.7, edgecolor='black', label=f'Byzantine ({num_byzantine})')]
            ax1.legend(handles=legend_elements, loc='upper right')
            
            # Add horizontal lines for group averages
            benign_mean = rhos_data["benign_rhos"]["mean"]
            ax1.axhline(y=benign_mean, color='darkgreen', linestyle='-', linewidth=2, 
                       label=f'Benign Avg: {benign_mean:.3f}')
            if byzantine_rhos:
                byz_mean = rhos_data["byzantine_rhos"]["mean"]
                ax1.axhline(y=byz_mean, color='darkred', linestyle='-', linewidth=2,
                           label=f'Byzantine Avg: {byz_mean:.3f}')
            
            # Add text annotations for means
            ax1.text(total_clients - 0.5, benign_mean, f'Benign Avg: {benign_mean:.3f}', 
                    color='darkgreen', fontsize=9, va='bottom')
            if byzantine_rhos:
                ax1.text(total_clients - 0.5, byz_mean, f'Byzantine Avg: {byz_mean:.3f}', 
                        color='darkred', fontsize=9, va='bottom')
            
            # Plot 2: Histogram/Distribution comparison
            ax2 = axes[1]
            
            if byzantine_rhos:
                # Create overlapping histograms
                bins = np.linspace(min(all_rhos) - 0.1, max(all_rhos) + 0.1, 20)
                ax2.hist(benign_rhos, bins=bins, alpha=0.6, color='green', label=f'Benign (n={num_benign})', edgecolor='black')
                ax2.hist(byzantine_rhos, bins=bins, alpha=0.6, color='red', label=f'Byzantine (n={num_byzantine})', edgecolor='black')
                ax2.legend(loc='upper right')
            else:
                bins = np.linspace(min(all_rhos) - 0.1, max(all_rhos) + 0.1, 20)
                ax2.hist(benign_rhos, bins=bins, alpha=0.6, color='green', label=f'Benign (n={num_benign})', edgecolor='black')
                ax2.legend(loc='upper right')
            
            ax2.set_xlabel('Concordance Ratio', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Distribution of Concordance Ratios', fontsize=12)
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add statistics text box
            stats_text = f"Benign: μ={benign_mean:.3f}, σ={rhos_data['benign_rhos']['std']:.3f}"
            if byzantine_rhos:
                byz_mean = rhos_data["byzantine_rhos"]["mean"]
                byz_std = rhos_data["byzantine_rhos"]["std"]
                stats_text += f"\nByzantine: μ={byz_mean:.3f}, σ={byz_std:.3f}"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(output_dir, f"concordance_distribution_{attack_name}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved to {plot_file}")
        
        # Create a combined plot comparing all attacks
        self._plot_combined_comparison(output_dir)
    
    def _plot_combined_comparison(self, output_dir: str):
        """
        Create a combined plot comparing concordance ratios across all attacks.
        
        Args:
            output_dir: Directory to save plots
        """
        if len(self.results) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        attack_names = []
        benign_means = []
        benign_stds = []
        byzantine_means = []
        byzantine_stds = []
        
        for attack_name, attack_data in self.results.items():
            if "rounds" not in attack_data or len(attack_data["rounds"]) == 0:
                continue
            
            last_round = attack_data["rounds"][-1]
            if "client_concordance_ratios" not in last_round:
                continue
            
            rhos_data = last_round["client_concordance_ratios"]
            
            attack_names.append(attack_name)
            benign_means.append(rhos_data["benign_rhos"]["mean"])
            benign_stds.append(rhos_data["benign_rhos"]["std"])
            
            if "byzantine_rhos" in rhos_data:
                byzantine_means.append(rhos_data["byzantine_rhos"]["mean"])
                byzantine_stds.append(rhos_data["byzantine_rhos"]["std"])
            else:
                byzantine_means.append(0)
                byzantine_stds.append(0)
        
        if len(attack_names) == 0:
            return
        
        x = np.arange(len(attack_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, benign_means, width, yerr=benign_stds, 
                       label='Benign', color='green', alpha=0.7, capsize=5)
        bars2 = ax.bar(x + width/2, byzantine_means, width, yerr=byzantine_stds,
                       label='Byzantine', color='red', alpha=0.7, capsize=5)
        
        ax.set_xlabel('Attack Type', fontsize=11)
        ax.set_ylabel('Average Concordance Ratio', fontsize=11)
        ax.set_title('Concordance Ratio Comparison Across Attacks', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attack_names, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, benign_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=8)
        for bar, mean in zip(bars2, byzantine_means):
            if mean != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, "concordance_comparison_all_attacks.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Combined comparison plot saved to {plot_file}")


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
                last_acc = epoch_rounds[-1]["test_accuracy"]
                last_loss = epoch_rounds[-1]["avg_train_loss"]
                benign_conc = epoch_rounds[-1]["benign_concordance"]
                
                info_parts = [
                    f"Epoch {epoch + 1:3d}",
                    f"Acc: {last_acc:5.1f}%",
                    f"Loss: {last_loss:.4f}",
                    f"Benign Conc: {benign_conc:.4f}"
                ]
                
                if "byzantine_concordance" in epoch_rounds[-1]:
                    byz_conc = epoch_rounds[-1]["byzantine_concordance"]
                    info_parts.append(f"Byzantine Conc: {byz_conc:.4f}")
                
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
        #{"name": "label_flip", "attack": "label_flip", "traitor": 0.2},
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
    
    # Generate concordance distribution plots
    analyzer.plot_concordance_distribution(output_dir)
    
    print(f"\nAblation test complete! Results saved to {output_dir}")