import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any

def plot_loss_curves(epochs_history: List[Dict[str, Any]], output_path: str) -> None:
    """Plots training losses over epochs."""
    df = pd.DataFrame(epochs_history)
    if df.empty or "total_loss" not in df.columns:
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(df.get("epoch", df.index), df["total_loss"], label="Total Loss", linewidth=2)
    
    if "l_flow" in df.columns:
        plt.plot(df.get("epoch", df.index), df["l_flow"], label="Flow Loss", alpha=0.7)
    if "l_od" in df.columns:
        plt.plot(df.get("epoch", df.index), df["l_od"], label="OD Loss", alpha=0.7)
        
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_lr_curve(epochs_history: List[Dict[str, Any]], output_path: str) -> None:
    """Plots learning rates over epochs."""
    df = pd.DataFrame(epochs_history)
    if df.empty:
        return
        
    lr_cols = [c for c in df.columns if c.startswith("lr_") or c == "lr"]
    if not lr_cols:
        return
        
    plt.figure(figsize=(10, 6))
    for col in lr_cols:
        plt.plot(df.get("epoch", df.index), df[col], label=col)
        
    plt.title("Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_loss_and_score(epochs_history: List[Dict[str, Any]], output_path: str) -> None:
    """Plots loss and monitoring score on dual axes."""
    df = pd.DataFrame(epochs_history)
    if df.empty or "total_loss" not in df.columns or "monitor_score" not in df.columns:
        return
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss', color=color)
    ax1.plot(df.get("epoch", df.index), df["total_loss"], color=color, label="Total Loss")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale("log")
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Monitor Score', color=color)
    ax2.plot(df.get("epoch", df.index), df["monitor_score"], color=color, label="Monitor Score (R2)")
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title("Loss vs. Score")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_alpha_beta_evolution(epochs_history: List[Dict[str, Any]], output_path: str) -> None:
    """Plots the evolution of alpha and beta parameters from the forward physics audit."""
    df = pd.DataFrame(epochs_history)
    if df.empty:
        return
        
    # Extract physics dict if nested
    if "forward_physics_audit" in df.columns:
        physics_df = pd.json_normalize(df["forward_physics_audit"])
    else:
        physics_df = df
        
    has_alpha = "alpha_mean" in physics_df.columns
    has_beta = "beta_mean" in physics_df.columns
    
    if not has_alpha and not has_beta:
        return
        
    epochs = df.get("epoch", df.index)
    
    fig, axes = plt.subplots(2 if has_alpha and has_beta else 1, 1, figsize=(10, 8), squeeze=False)
    
    idx = 0
    if has_alpha:
        ax = axes[idx, 0]
        ax.plot(epochs, physics_df["alpha_mean"], 'b-', label="Mean Alpha")
        if "alpha_min" in physics_df.columns and "alpha_max" in physics_df.columns:
            ax.fill_between(epochs, physics_df["alpha_min"], physics_df["alpha_max"], color='b', alpha=0.2, label="Min/Max Range")
        ax.set_title("Alpha Evolution")
        ax.set_ylabel("Alpha")
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1
        
    if has_beta:
        ax = axes[idx, 0]
        ax.plot(epochs, physics_df["beta_mean"], 'r-', label="Mean Beta")
        if "beta_min" in physics_df.columns and "beta_max" in physics_df.columns:
            ax.fill_between(epochs, physics_df["beta_min"], physics_df["beta_max"], color='r', alpha=0.2, label="Min/Max Range")
        ax.set_title("Beta Evolution")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Beta")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
