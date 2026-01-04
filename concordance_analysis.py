import torch
from typing import List, Dict
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# Concordance logic
# ============================================================

def compute_concordance_matrix(grads: List[torch.Tensor]) -> torch.Tensor:
    G = torch.stack([g.flatten() for g in grads])   # (K, D)
    S = torch.sign(G)
    C = (S @ S.T) / S.shape[1]
    return C


def concordance_statistics(C: torch.Tensor, num_benign: int) -> Dict:
    K = C.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
    client_scores = (C * mask).sum(dim=1) / (K - 1)

    benign = client_scores[:num_benign]
    byzantine = client_scores[num_benign:]

    return {
        "all_rhos": client_scores.cpu().tolist(),
        "benign_mean": float(benign.mean()),
        "byzantine_mean": float(byzantine.mean()) if len(byzantine) > 0 else 0.0,
    }


# ============================================================
# Visualization utilities
# ============================================================

def save_heatmap(C: np.ndarray, round_id: int, out_dir: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"Concordance Matrix – Round {round_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"round_{round_id}_heatmap.png"))
    plt.close()


def save_graph(C: np.ndarray, num_benign: int, round_id: int, out_dir: str):
    K = C.shape[0]
    G = nx.Graph()

    for i in range(K):
        G.add_node(
            i,
            color="blue" if i < num_benign else "red"
        )

    for i in range(K):
        for j in range(i + 1, K):
            w = C[i, j]
            if abs(w) > 0.05:  # visibility threshold
                G.add_edge(i, j, weight=w)

    pos = nx.spring_layout(G, seed=42)

    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    edge_weights = [abs(G[u][v]["weight"]) * 5 for u, v in G.edges]
    edge_colors = [G[u][v]["weight"] for u, v in G.edges]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.RdBu_r,
        width=edge_weights
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f"Concordance Graph – Round {round_id}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"round_{round_id}_graph.png"))
    plt.close()


def save_numeric_matrix(C: np.ndarray, round_id: int, out_dir: str):
    """
    Save concordance matrix in human-readable formats
    """
    # CSV (for Excel / pandas)
    csv_path = os.path.join(out_dir, f"round_{round_id}_concordance.csv")
    np.savetxt(csv_path, C, delimiter=",", fmt="%.4f")

    # TXT (pretty printed)
    txt_path = os.path.join(out_dir, f"round_{round_id}_concordance.txt")
    with open(txt_path, "w") as f:
        f.write(f"Concordance Matrix – Round {round_id}\n")
        f.write("=" * 40 + "\n")
        for row in C:
            f.write(" ".join(f"{v:+.3f}" for v in row) + "\n")


def save_numeric_matrix_png(
    C: np.ndarray,
    num_benign: int,
    round_id: int,
    out_dir: str
):
    """
    Save concordance matrix as a PNG with numeric values in cells.
    """
    K = C.shape[0]

    fig, ax = plt.subplots(figsize=(0.6*K, 0.6*K))

    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)

    # Write numbers in each cell
    for i in range(K):
        for j in range(K):
            ax.text(
                j, i,
                f"{C[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if abs(C[i, j]) < 0.6 else "white"
            )

    # Draw benign / byzantine boundary
    ax.axhline(num_benign - 0.5, color="black", linewidth=2)
    ax.axvline(num_benign - 0.5, color="black", linewidth=2)

    ax.set_title(f"Sign Concordance Matrix – Round {round_id}")
    ax.set_xlabel("Client index")
    ax.set_ylabel("Client index")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sign Concordance")

    plt.tight_layout()

    path = os.path.join(out_dir, f"round_{round_id}_concordance_numeric.png")
    plt.savefig(path, dpi=200)
    plt.close()




# ============================================================
# One-round analysis
# ============================================================

def analyze_concordance_round(fl, round_id: int, out_dir: str) -> Dict:
    all_grads = fl.train()
    num_benign = len(fl.benign_clients)

    C = compute_concordance_matrix(all_grads)
    stats = concordance_statistics(C, num_benign)

    # ---- SAVE MATRIX ----
    C_np = C.cpu().numpy()
    np.save(os.path.join(out_dir, f"round_{round_id}_concordance.npy"), C_np)

    # ---- SAVE NUMERIC MATRICES ----
    save_numeric_matrix(C_np, round_id, out_dir)

    # ---- VISUALIZE ----
    save_heatmap(C_np, round_id, out_dir)
    save_graph(C_np, num_benign, round_id, out_dir)


    save_numeric_matrix_png(
        C=C_np,
        num_benign=num_benign,
        round_id=round_id,
        out_dir=out_dir
    )



    fl.aggregate(all_grads)
    fl.update_global_model()
    acc = fl.evaluate_accuracy()

    stats["accuracy"] = float(acc * 100) if acc is not None else 0.0
    return stats


# ============================================================
# Multi-round analysis
# ============================================================

def analyze_concordance_epochs(fl, epochs: int, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "benign_mean": [],
        "byzantine_mean": [],
        "accuracy": []
    }

    for r in range(epochs):
        stats = analyze_concordance_round(fl, r + 1, out_dir)

        summary["benign_mean"].append(stats["benign_mean"])
        summary["byzantine_mean"].append(stats["byzantine_mean"])
        summary["accuracy"].append(stats["accuracy"])

        print(
            f"🔁 Round {r+1:02d} | "
            f"Benign ω={stats['benign_mean']:.4f} | "
            f"Byz ω={stats['byzantine_mean']:.4f} | "
            f"Acc={stats['accuracy']:.2f}%"
        )

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    from parameters import args_parser
    from mapper import Mapper

    args = args_parser()
    args.trials = 1
    args.global_epoch = 1
    args.dataset_name = "mnist"
    args.dataset_dist = "iid"
    args.aggr = "avg"

    epochs = 10
    out_dir = "./concordance_results"

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    mapper = Mapper(args, device)
    fl = mapper.initialize_FL()

    print("=" * 70)
    print(f"Benign clients: {len(fl.benign_clients)}")
    print(f"Byzantine clients: {len(fl.malicious_clients)}")
    print(f"Epochs: {epochs}")
    print("=" * 70)

    summary = analyze_concordance_epochs(fl, epochs, out_dir)

    print("\n📌 FINAL SUMMARY")
    print(f"Benign mean (avg):    {np.mean(summary['benign_mean']):.4f}")
    print(f"Byzantine mean (avg): {np.mean(summary['byzantine_mean']):.4f}")
    print(f"Final accuracy:       {summary['accuracy'][-1]:.2f}%")
    print(f"\n📁 Results saved in: {out_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
