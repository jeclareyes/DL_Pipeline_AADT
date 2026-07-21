"""
Traffic assignment consistency test using CGAME_DataDriven outputs.

This script:
1) Loads CGAME estimated OD/flows from an eval bundle (eval_*.pt).
2) Loads a saved model checkpoint (required argument for run traceability).
3) Runs a basic traffic assignment (Frank-Wolfe + BPR) using the estimated OD.
4) Compares assigned flows against:
   - CGAME reconstructed flows
   - Observed link flows from the processed network graph
5) Saves scatter plots and goodness-of-fit metrics.

Usage:
    python src/test/Traffic_Assignment_Test.py \
      --eval-path outputs/.../eval_model.pt \
      --checkpoint-path outputs/.../model.pt \
      --network Synthetic03
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Keep imports local to project structure even if script is launched from a subfolder.
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.components.models.traditional_TA.frank_wolfe_congestion import FrankWolfeAssignmentCongestion
from src.test._testing_functions import calculate_metrics, get_latest_epoch_data, load_eval_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Traffic Assignment from CGAME estimated OD and compare flows."
    )
    parser.add_argument("--eval-path", required=True, help="Path to eval_*.pt file.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to model checkpoint *.pt file.")
    parser.add_argument("--network", default="Synthetic03", help="Dataset/network name (default: Synthetic03).")
    parser.add_argument(
        "--output-dir",
        default="outputs/runs/traffic_assignment_test",
        help="Output directory for plots and reports.",
    )
    parser.add_argument("--max-iterations", type=int, default=100, help="Max Frank-Wolfe iterations.")
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-3,
        help="Convergence threshold for Frank-Wolfe relative gap.",
    )
    return parser.parse_args()


def _as_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _to_numpy_1d(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required value: {name}")
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _load_processed_artifacts(network: str) -> Tuple[nx.DiGraph, sparse.csr_matrix]:
    proc_dir = PROJECT_ROOT / "data" / "processed" / network
    graph_path = proc_dir / f"{network}_graph.pkl"
    od_path = proc_dir / f"{network}_od_matrix.npz"

    if not graph_path.exists():
        raise FileNotFoundError(f"Processed graph not found: {graph_path}")
    if not od_path.exists():
        raise FileNotFoundError(f"Processed OD matrix not found: {od_path}")

    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)

    od_matrix = sparse.load_npz(od_path).tocsr()
    return graph, od_matrix


def _coerce_nodes_to_int(graph: nx.DiGraph) -> nx.DiGraph:
    mapping: Dict[Any, int] = {}
    changed = False
    for node in graph.nodes():
        if isinstance(node, int):
            mapping[node] = node
            continue
        try:
            new_node = int(node)
            mapping[node] = new_node
            if new_node != node:
                changed = True
        except Exception as exc:
            raise ValueError(f"Graph node '{node}' cannot be converted to int") from exc

    if not changed:
        return graph

    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    return relabeled


def _apply_bpr_fallbacks(graph: nx.DiGraph) -> Dict[str, int]:
    counts = {
        "free_flow_time": 0,
        "capacity": 0,
        "b": 0,
        "power": 0,
    }

    for _, _, data in graph.edges(data=True):
        t0 = data.get("free_flow_time", None)
        if t0 is None or float(t0) <= 0:
            length = data.get("length", 1.0)
            data["free_flow_time"] = float(length) if length is not None and float(length) > 0 else 1.0
            counts["free_flow_time"] += 1

        cap = data.get("capacity", None)
        if cap is None or float(cap) <= 0:
            data["capacity"] = 1000.0
            counts["capacity"] += 1

        b_val = data.get("b", None)
        if b_val is None or float(b_val) <= 0:
            data["b"] = 0.15
            counts["b"] += 1

        p_val = data.get("power", None)
        if p_val is None or float(p_val) <= 0:
            data["power"] = 4.0
            counts["power"] += 1

    return counts


def _make_predicted_od_matrix(
    base_od_matrix: sparse.csr_matrix,
    predicted_od: np.ndarray,
    od_pair_indices: Any | None = None,
) -> sparse.csr_matrix:
    # Preferred path: explicit OD pair mapping saved in eval static_data.
    if od_pair_indices is not None:
        if isinstance(od_pair_indices, torch.Tensor):
            idx = od_pair_indices.detach().cpu().numpy()
        else:
            idx = np.asarray(od_pair_indices)

        if idx.ndim == 2 and idx.shape[1] >= 2 and idx.shape[0] == predicted_od.shape[0]:
            rows = idx[:, 0].astype(int)
            cols = idx[:, 1].astype(int)
            size = int(max(rows.max(), cols.max()) + 1)
            pred_od_sparse = sparse.coo_matrix((predicted_od, (rows, cols)), shape=(size, size))
            return pred_od_sparse.tocsr()

    # Fallback A: preserve sparsity pattern from processed OD matrix.
    base_coo = base_od_matrix.tocoo()
    if predicted_od.shape[0] == base_coo.nnz:
        pred_od_sparse = sparse.coo_matrix(
            (predicted_od, (base_coo.row, base_coo.col)),
            shape=base_od_matrix.shape,
        )
        return pred_od_sparse.tocsr()

    # Fallback B: dense reshape when model predicts full square OD vector.
    total_size = base_od_matrix.shape[0] * base_od_matrix.shape[1]
    if predicted_od.shape[0] == total_size:
        dense = predicted_od.reshape(base_od_matrix.shape)
        return sparse.csr_matrix(dense)

    raise ValueError(
        "Could not reconstruct predicted OD matrix. Length mismatch against all strategies: "
        f"predicted={predicted_od.shape[0]}, base_nnz={base_coo.nnz}, base_shape={base_od_matrix.shape}, "
        f"od_pair_indices_shape={getattr(od_pair_indices, 'shape', None)}"
    )


def _scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=14, color="#1f77b4", edgecolors="none")

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    if max_val <= min_val:
        max_val = min_val + 1.0

    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="y=x")
    plt.xlabel("Reference flow")
    plt.ylabel("TA assigned flow")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.25)

    metrics_text = (
        f"R2={metrics['R2']:.4f}\n"
        f"MAE={metrics['MAE']:.2f}\n"
        f"RMSE={metrics['RMSE']:.2f}\n"
        f"MAPE={metrics['MAPE']:.2f}%"
    )
    plt.text(
        0.98,
        0.02,
        metrics_text,
        transform=plt.gca().transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _validate_lengths(ta: np.ndarray, cgame: np.ndarray, observed: np.ndarray) -> None:
    if not (ta.shape[0] == cgame.shape[0] == observed.shape[0]):
        raise ValueError(
            "Flow vector length mismatch: "
            f"ta={ta.shape[0]}, cgame={cgame.shape[0]}, observed={observed.shape[0]}"
        )

    if not (np.isfinite(ta).all() and np.isfinite(cgame).all() and np.isfinite(observed).all()):
        raise ValueError("At least one flow vector contains NaN or Inf values")


def _read_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    epochs = raw.get("epochs_history", {})
    if not epochs:
        raise ValueError(f"Checkpoint has no 'epochs_history': {checkpoint_path}")

    latest = max(int(k) for k in epochs.keys())
    return {
        "latest_epoch": latest,
        "has_config": "config" in raw,
    }


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("TrafficAssignmentTest")

    eval_path = _as_path(args.eval_path)
    checkpoint_path = _as_path(args.checkpoint_path)
    output_dir = _as_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        raise FileNotFoundError(f"eval-path not found: {eval_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint-path not found: {checkpoint_path}")

    logger.info("Loading eval bundle: %s", eval_path)
    eval_bundle = load_eval_bundle(str(eval_path))
    eval_epoch, epoch_data = get_latest_epoch_data(eval_bundle)

    logger.info("Loading checkpoint metadata: %s", checkpoint_path)
    checkpoint_meta = _read_checkpoint_metadata(checkpoint_path)

    pred_od = _to_numpy_1d(epoch_data.get("pred_od"), "pred_od")
    pred_flows_cgame = _to_numpy_1d(epoch_data.get("pred_flows"), "pred_flows")

    logger.info("Loading processed artifacts for network: %s", args.network)
    graph, base_od = _load_processed_artifacts(args.network)
    graph = _coerce_nodes_to_int(graph)

    bpr_fallback_counts = _apply_bpr_fallbacks(graph)
    logger.info("BPR fallback counts: %s", bpr_fallback_counts)

    static_data = eval_bundle.get("static_data", {})
    pred_od_sparse = _make_predicted_od_matrix(
        base_od_matrix=base_od,
        predicted_od=pred_od,
        od_pair_indices=static_data.get("od_pair_indices"),
    )

    logger.info("Running Frank-Wolfe + BPR traffic assignment...")
    solver = FrankWolfeAssignmentCongestion(
        graph=graph,
        od_matrix=pred_od_sparse,
        cost_function="bpr",
        solution_attr="solution_congestion",
        use_cache=False,
        k_paths=10,
    )
    ta_stats = solver.solve(
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        verbose=True,
    )

    edge_list = list(graph.edges())
    ta_flows = np.array([graph[u][v].get("solution_congestion", 0.0) for u, v in edge_list], dtype=float)
    observed_flows = np.array([graph[u][v].get("volume", 0.0) for u, v in edge_list], dtype=float)

    _validate_lengths(ta_flows, pred_flows_cgame, observed_flows)

    metrics_ta_vs_cgame = calculate_metrics(pred=ta_flows, target=pred_flows_cgame)
    metrics_ta_vs_observed = calculate_metrics(pred=ta_flows, target=observed_flows)

    _scatter_plot(
        y_true=pred_flows_cgame,
        y_pred=ta_flows,
        metrics=metrics_ta_vs_cgame,
        title="TA assigned vs CGAME reconstructed flows",
        output_path=output_dir / "ta_vs_cgame_scatter.png",
    )
    _scatter_plot(
        y_true=observed_flows,
        y_pred=ta_flows,
        metrics=metrics_ta_vs_observed,
        title="TA assigned vs observed flows",
        output_path=output_dir / "ta_vs_observed_scatter.png",
    )

    comparison_df = pd.DataFrame(
        {
            "from_node": [u for u, _ in edge_list],
            "to_node": [v for _, v in edge_list],
            "ta_assigned_flow": ta_flows,
            "cgame_reconstructed_flow": pred_flows_cgame,
            "observed_flow": observed_flows,
            "err_ta_minus_cgame": ta_flows - pred_flows_cgame,
            "err_ta_minus_observed": ta_flows - observed_flows,
        }
    )
    comparison_df.to_csv(output_dir / "link_flow_comparison.csv", index=False)

    metrics_summary = {
        "ta_vs_cgame": metrics_ta_vs_cgame,
        "ta_vs_observed": metrics_ta_vs_observed,
        "traffic_assignment": {
            "converged": bool(ta_stats.get("converged", False)),
            "iterations": int(ta_stats.get("iterations", -1)),
            "final_gap": float(ta_stats.get("final_gap", np.nan)),
            "total_cost": float(ta_stats.get("total_cost", np.nan)),
        },
        "inputs": {
            "network": args.network,
            "eval_path": str(eval_path),
            "checkpoint_path": str(checkpoint_path),
            "eval_latest_epoch": int(eval_epoch),
            "checkpoint_latest_epoch": int(checkpoint_meta["latest_epoch"]),
        },
        "bpr_fallback_counts": bpr_fallback_counts,
    }

    metrics_rows = [
        {"comparison": "ta_vs_cgame", **metrics_ta_vs_cgame},
        {"comparison": "ta_vs_observed", **metrics_ta_vs_observed},
    ]
    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics_summary.csv", index=False)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_summary, fh, indent=2)

    with (output_dir / "run_summary.txt").open("w", encoding="utf-8") as fh:
        fh.write("Traffic Assignment Test Summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Network: {args.network}\n")
        fh.write(f"Eval file: {eval_path}\n")
        fh.write(f"Checkpoint file: {checkpoint_path}\n")
        fh.write(f"Eval latest epoch: {eval_epoch}\n")
        fh.write(f"Checkpoint latest epoch: {checkpoint_meta['latest_epoch']}\n")
        fh.write(f"Converged: {ta_stats.get('converged', False)}\n")
        fh.write(f"Iterations: {ta_stats.get('iterations', -1)}\n")
        fh.write(f"Final gap: {ta_stats.get('final_gap', np.nan)}\n")
        fh.write("\nTA vs CGAME\n")
        for k, v in metrics_ta_vs_cgame.items():
            fh.write(f"  {k}: {v}\n")
        fh.write("\nTA vs Observed\n")
        for k, v in metrics_ta_vs_observed.items():
            fh.write(f"  {k}: {v}\n")

    logger.info("Artifacts saved in: %s", output_dir)
    logger.info("TA vs CGAME metrics: %s", metrics_ta_vs_cgame)
    logger.info("TA vs Observed metrics: %s", metrics_ta_vs_observed)


if __name__ == "__main__":
    main()
