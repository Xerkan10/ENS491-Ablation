# ENS491-Ablation

## Gradient Similarity Analysis for Byzantine-Robust Federated Learning

This repository contains analysis tools for studying gradient relationships between benign and Byzantine (malicious) clients in Federated Learning settings. The analysis focuses on two key metrics: **Sign Concordance** and **Cosine Similarity**.

---

## Analysis Methods

### 1. Sign Concordance Analysis

Sign concordance measures the agreement in gradient directions between clients by comparing the signs of gradient components.

**Mathematical Definition:**

For gradient vectors $G_i$ and $G_j$ from clients $i$ and $j$:

$$C_{ij} = \frac{\text{sign}(G_i) \cdot \text{sign}(G_j)}{D}$$

where $D$ is the dimension of the gradient vector.

- **Range:** $[-1, 1]$
- **Interpretation:**
  - $C_{ij} = 1$: Perfect agreement (all gradient signs match)
  - $C_{ij} = 0$: Random/uncorrelated gradients
  - $C_{ij} = -1$: Perfect disagreement (all gradient signs opposite)

### 2. Cosine Similarity Analysis

Cosine similarity measures the angular similarity between gradient vectors, capturing both direction and relative magnitude.

**Mathematical Definition:**

$$\text{CosSim}(G_i, G_j) = \frac{G_i \cdot G_j}{\|G_i\| \|G_j\|}$$

- **Range:** $[-1, 1]$
- **Interpretation:**
  - $1$: Identical direction
  - $0$: Orthogonal gradients
  - $-1$: Opposite direction

---

## Federated Learning Setup Parameters

The experiments use the following default configuration (see [parameters.py](parameters.py)):

| Parameter      | Value      | Description                                        |
| -------------- | ---------- | -------------------------------------------------- |
| `num_client`   | 25         | Total number of clients                            |
| `traitor`      | 0.2        | Byzantine client ratio (20% = 5 malicious clients) |
| `global_epoch` | 10         | Number of communication rounds                     |
| `localIter`    | 1          | Local training epochs per round                    |
| `dataset_name` | MNIST      | Dataset used                                       |
| `nn_name`      | mnistnet   | Neural network architecture                        |
| `attack`       | ALIE / IPM | Attack type (varies by experiment)                 |
| `aggr`         | Bulyan     | Aggregation method                                 |
| `dataset_dist` | IID        | Data distribution                                  |
| `bs`           | 32         | Batch size                                         |
| `lr`           | 0.1        | Learning rate                                      |
| `opt`          | SGD        | Optimizer                                          |
| `Lmomentum`    | 0.9        | Local momentum                                     |

---

## Results Location

Visual results (matrices and graphs) are stored in the following directories:

### Sign Concordance Results

| Attack      | Directory                                              |
| ----------- | ------------------------------------------------------ |
| ALIE Attack | [`concordance_results/`](concordance_results/)         |
| IPM Attack  | [`concordance_results_ipm/`](concordance_results_ipm/) |

### Cosine Similarity Results

| Attack      | Directory                                    |
| ----------- | -------------------------------------------- |
| ALIE Attack | [`similarity_results/`](similarity_results/) |

### Output Files Per Round

Each directory contains the following files for each round:

| File                                                                 | Description                                              |
| -------------------------------------------------------------------- | -------------------------------------------------------- |
| `round_X_concordance_numeric.png` / `round_X_similarity_numeric.png` | **Matrix visualization with numeric values**             |
| `round_X_heatmap.png`                                                | Color-coded heatmap visualization                        |
| `round_X_graph.png`                                                  | Network graph visualization (blue=benign, red=Byzantine) |
| `round_X_concordance.csv` / `round_X_similarity.csv`                 | Raw matrix data (CSV format)                             |
| `round_X_concordance.npy` / `round_X_similarity.npy`                 | NumPy array format                                       |
| `round_X_concordance.txt` / `round_X_similarity.txt`                 | Human-readable text format                               |
| `summary.json`                                                       | Aggregated statistics across all rounds                  |

---

## Key Findings

### 1. Byzantine Clients Show Higher Inter-Similarity

Both metrics reveal that Byzantine clients exhibit **higher pairwise similarity** compared to benign clients:

#### ALIE Attack (Sign Concordance)

| Metric         | Round 1   | Round 10  |
| -------------- | --------- | --------- |
| Benign Mean    | 0.186     | 0.468     |
| Byzantine Mean | 0.399     | 0.628     |
| **Gap**        | **0.213** | **0.160** |

#### ALIE Attack (Cosine Similarity)

| Metric         | Round 1   | Round 10  |
| -------------- | --------- | --------- |
| Benign Mean    | 0.309     | 0.696     |
| Byzantine Mean | 0.587     | 0.845     |
| **Gap**        | **0.278** | **0.149** |

### 2. IPM Attack Creates Negative Concordance

The Inner Product Manipulation (IPM) attack produces **negative sign concordance** between Byzantine and benign clients, indicating adversarial gradient directions:

| Metric         | Round 1    | Round 10   |
| -------------- | ---------- | ---------- |
| Benign Mean    | 0.043      | 0.217      |
| Byzantine Mean | **-0.093** | **-0.307** |

This negative correlation is a distinguishing signature of the IPM attack strategy.

### 3. Similarity Increases Over Training

Both benign and Byzantine client similarities increase as training progresses:

- **Benign:** Gradients converge as the model approaches optimum
- **Byzantine:** Attack gradients become more coordinated/focused

### 4. Gap Remains Exploitable

Despite both groups increasing in similarity, the **gap between Byzantine and benign means persists**, suggesting these metrics could be leveraged for Byzantine client detection in aggregation algorithms.

---

## Running the Analysis

```bash
# Run sign concordance analysis
python concordance_analysis.py

# Run cosine similarity analysis
python cosine_similarity_analysis.py
```

---

## Visualization Examples

The numeric matrix visualizations show:

- **Rows/Columns 0-19:** Benign clients
- **Rows/Columns 20-24:** Byzantine clients
- **Black lines:** Boundary between benign and Byzantine regions
- **Color scale:** Red (high similarity) to Blue (low/negative similarity)
