# ============================================================================
# TRAINING HISTORY VISUALIZATION
# ============================================================================
# Loads training history JSON files and creates comparison plots.
# Usage: python plot_training_history.py
# ============================================================================

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Use a clean, publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')


def load_history(filepath):
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_history_files(checkpoints_dir='checkpoints'):
    """Find all training history JSON files in the checkpoints directory."""
    history_files = []
    if os.path.exists(checkpoints_dir):
        for f in os.listdir(checkpoints_dir):
            if f.endswith('_history.json'):
                history_files.append(os.path.join(checkpoints_dir, f))
    return history_files


def plot_training_comparison(histories, save_path='results/training_comparison.png'):
    """Create side-by-side comparison plot of Training Loss and Validation Accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color palette for different models
    colors = {
        'Baseline': '#D55E00',      # Orange
        'Teacher': '#009E73',        # Green/Teal (Vanilla in your image)
        'Distillation': '#0072B2',   # Blue (AKTP in your image)
    }
    
    # Default colors for unknown models
    default_colors = ['#CC79A7', '#F0E442', '#56B4E9', '#E69F00']
    color_idx = 0
    
    # Plot Training Loss
    ax1 = axes[0]
    for hist in histories:
        model_name = hist.get('model_name', 'Unknown')
        color = colors.get(model_name, default_colors[color_idx % len(default_colors)])
        if model_name not in colors:
            color_idx += 1
        
        epochs = hist['epochs']
        # For Distillation, use supervised loss for fair comparison with other models
        if 'train_loss_sup' in hist:
            train_loss = hist['train_loss_sup']
        elif 'train_loss' in hist:
            train_loss = hist['train_loss']
        elif 'train_loss_total' in hist:
            train_loss = hist['train_loss_total']
        else:
            train_loss = [0] * len(epochs)
        ax1.plot(epochs, train_loss, label=model_name, color=color, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('(a) Training Loss', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Plot Validation Accuracy (Dice Score as percentage)
    ax2 = axes[1]
    for hist in histories:
        model_name = hist.get('model_name', 'Unknown')
        color = colors.get(model_name, default_colors[color_idx % len(default_colors)])
        if model_name not in colors:
            color_idx += 1
        
        epochs = hist['epochs']
        # Convert Dice score to percentage for better visualization
        val_dice_pct = [d * 100 for d in hist['val_dice']]
        ax2.plot(epochs, val_dice_pct, label=model_name, color=color, linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Val Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Validation Accuracy', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_xlim(left=0)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")
    
    # Also save as PDF for publication quality
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()


def plot_individual_metrics(histories, save_dir='results'):
    """Create additional individual plots for detailed analysis."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot IoU comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {
        'Baseline': '#D55E00',
        'Teacher': '#009E73',
        'Distillation': '#0072B2',
    }
    default_colors = ['#CC79A7', '#F0E442', '#56B4E9']
    color_idx = 0
    
    for hist in histories:
        model_name = hist.get('model_name', 'Unknown')
        color = colors.get(model_name, default_colors[color_idx % len(default_colors)])
        if model_name not in colors:
            color_idx += 1
        
        epochs = hist['epochs']
        val_iou_pct = [iou * 100 for iou in hist['val_iou']]
        ax.plot(epochs, val_iou_pct, label=model_name, color=color, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation IoU (%)', fontsize=12)
    ax.set_title('Validation IoU Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"IoU comparison plot saved to: {os.path.join(save_dir, 'iou_comparison.png')}")
    plt.close()


def print_summary(histories):
    """Print a summary table of final metrics for each model."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Final Loss':<12} {'Best Dice':<12} {'Best IoU':<12}")
    print("-"*60)
    
    for hist in histories:
        model_name = hist.get('model_name', 'Unknown')
        # Handle different loss key names
        if 'train_loss_sup' in hist:
            loss_data = hist['train_loss_sup']
        elif 'train_loss' in hist:
            loss_data = hist['train_loss']
        elif 'train_loss_total' in hist:
            loss_data = hist['train_loss_total']
        else:
            loss_data = [0]
        final_loss = loss_data[-1] if loss_data else 0
        best_dice = max(hist['val_dice']) if hist['val_dice'] else 0
        best_iou = max(hist['val_iou']) if hist['val_iou'] else 0
        
        print(f"{model_name:<15} {final_loss:<12.4f} {best_dice*100:<12.2f}% {best_iou*100:<12.2f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training history comparison')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='Directory containing history JSON files')
    parser.add_argument('--output', type=str, default='results/training_comparison.png',
                        help='Output path for the comparison plot')
    parser.add_argument('--history_files', type=str, nargs='*', default=None,
                        help='Specific history files to plot (optional)')
    args = parser.parse_args()
    
    # Find or use specified history files
    if args.history_files:
        history_files = args.history_files
    else:
        history_files = find_history_files(args.checkpoints_dir)
    
    if not history_files:
        print(f"No history files found in {args.checkpoints_dir}/")
        print("Run training scripts first to generate *_history.json files.")
        return
    
    print(f"Found {len(history_files)} history file(s):")
    for f in history_files:
        print(f"  - {f}")
    
    # Load all histories
    histories = []
    for filepath in history_files:
        try:
            hist = load_history(filepath)
            histories.append(hist)
            print(f"Loaded: {hist.get('model_name', 'Unknown')} ({len(hist['epochs'])} epochs)")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not histories:
        print("No valid history files could be loaded.")
        return
    
    # Print summary
    print_summary(histories)
    
    # Create plots
    plot_training_comparison(histories, save_path=args.output)
    plot_individual_metrics(histories, save_dir=os.path.dirname(args.output) or 'results')
    
    print("\nDone! Check the results/ folder for generated plots.")


if __name__ == "__main__":
    main()
