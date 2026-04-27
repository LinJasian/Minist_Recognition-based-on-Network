"""
Experiment 4: Comprehensive Model Comparison (INFERENCE ONLY - NO TRAINING!)
Load saved results from exp1-exp3 and create comparison visualization.

THIS SCRIPT DOES NOT TRAIN ANY MODELS - IT ONLY READS JSON AND CREATES PLOTS
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import numpy as np
from utils.config import RESULTS_DIR


def load_exp_results(exp_num: int) -> dict:
    """Load results from exp1, exp2, or exp3 - FAST JSON ONLY"""
    results_file = os.path.join(RESULTS_DIR, f"exp{exp_num}_results.json")
    with open(results_file, 'r') as f:
        return json.load(f)


def main():
    print("\n" + "="*70)
    print("Experiment 4: Comprehensive Model Comparison (INFERENCE ONLY)")
    print("="*70)
    
    # ===== Load Results from Previous Experiments (FAST - JSON only) =====
    print("\n[1/3] Loading saved results from experiments 1-3...")
    
    try:
        exp1_results = load_exp_results(1)
        print("  ✓ Loaded exp1 (BP-Basic)")
    except FileNotFoundError:
        print("  ✗ exp1_results.json not found. Please run exp1 first.")
        return None
    
    try:
        exp2_results = load_exp_results(2)
        print("  ✓ Loaded exp2 (BP-Improved)")
    except FileNotFoundError:
        print("  ✗ exp2_results.json not found. Please run exp2 first.")
        return None
    
    try:
        exp3_results = load_exp_results(3)
        print("  ✓ Loaded exp3 (CNN)")
    except FileNotFoundError:
        print("  ✗ exp3_results.json not found. Please run exp3 first.")
        return None
    
    # ===== Extract Key Metrics =====
    print("\n[2/3] Extracting metrics from loaded results...")
    
    # From exp1: BP-Basic
    acc_bp_basic = exp1_results['metrics']['test_acc']
    f1_bp_basic = exp1_results['metrics']['test_f1']
    history_bp_basic = exp1_results['training_history']
    
    # From exp2: BP-Improved
    acc_bp_improved = exp2_results['improved_model']['accuracy']
    f1_bp_improved = exp2_results['improved_model']['f1_score']
    history_bp_improved = exp2_results['improved_model']['training_history']
    
    # From exp3: CNN-ReLU (best from exp3)
    acc_cnn = exp3_results['cnn_relu']['accuracy']
    f1_cnn = exp3_results['cnn_relu']['f1_score']
    history_cnn = exp3_results['cnn_relu']['training_history']
    
    # ===== Print Summary Comparison =====
    print("\n" + "-"*70)
    print("Model Performance Summary:")
    print("-"*70)
    
    models_data = [
        ('BP-Basic', acc_bp_basic, f1_bp_basic),
        ('BP-Improved', acc_bp_improved, f1_bp_improved),
        ('CNN-ReLU', acc_cnn, f1_cnn),
    ]
    
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12}")
    print("-"*70)
    for model_name, acc, f1 in models_data:
        print(f"{model_name:<20} {acc:<12.4f} {f1:<12.4f}")
    
    # Performance ranking
    print("\n" + "-"*70)
    print("Performance Ranking:")
    print("-"*70)
    
    ranked = sorted(models_data, key=lambda x: x[1], reverse=True)
    for rank, (model_name, acc, f1) in enumerate(ranked, 1):
        print(f"  {rank}. {model_name}: {acc:.4f}")
    
    # Performance improvement analysis
    print("\nPerformance Improvement Analysis:")
    improvement_bp_improved = (acc_bp_improved - acc_bp_basic) / acc_bp_basic * 100
    improvement_cnn = (acc_cnn - acc_bp_basic) / acc_bp_basic * 100
    
    print(f"  BP-Improved improvement over BP-Basic: {improvement_bp_improved:+.2f}%")
    print(f"  CNN improvement over BP-Basic: {improvement_cnn:+.2f}%")
    
    # ===== Comprehensive Visualization =====
    print("\n[3/3] Generating comprehensive comparison visualizations...")
    
    # Create figure with larger size and more space
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, top=0.95, bottom=0.08)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. Test Accuracy Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['BP-Basic', 'BP-Improved', 'CNN']
    accuracies = [acc_bp_basic, acc_bp_improved, acc_cnn]
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Set Accuracy', fontsize=13, fontweight='bold')
    
    # Set dynamic y-axis limits based on actual values
    min_acc = min(accuracies) - 0.05
    max_acc = max(accuracies) + 0.05
    ax1.set_ylim([max(0.0, min_acc), min(1.0, max_acc)])
    ax1.tick_params(axis='x', labelsize=11)
    
    # Add value labels on bars with proper spacing
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 2. F1 Score Comparison (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    f1_scores = [f1_bp_basic, f1_bp_improved, f1_cnn]
    bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    ax2.set_ylabel('F1 Score (Weighted)', fontsize=12, fontweight='bold')
    ax2.set_title('Weighted F1 Score', fontsize=13, fontweight='bold')
    
    # Set dynamic y-axis limits
    min_f1 = min(f1_scores) - 0.05
    max_f1 = max(f1_scores) + 0.05
    ax2.set_ylim([max(0.0, min_f1), min(1.0, max_f1)])
    ax2.tick_params(axis='x', labelsize=11)
    
    # Add value labels on bars with proper spacing
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. Training Curves Comparison (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    bp_basic_epochs = len(history_bp_basic['val_accs'])
    bp_improved_epochs = len(history_bp_improved['val_accs'])
    cnn_epochs = len(history_cnn['val_accs'])
    
    ax3.plot(range(bp_basic_epochs), history_bp_basic['val_accs'],
            label='BP-Basic', linewidth=2.5, color='#FF6B6B', marker='o', markersize=3,
            markevery=max(1, bp_basic_epochs//10))
    ax3.plot(range(bp_improved_epochs), history_bp_improved['val_accs'],
            label='BP-Improved', linewidth=2.5, color='#4ECDC4', marker='s', markersize=3,
            markevery=max(1, bp_improved_epochs//10))
    ax3.plot(range(cnn_epochs), history_cnn['val_accs'],
            label='CNN-ReLU', linewidth=2.5, color='#45B7D1', marker='^', markersize=3,
            markevery=max(1, cnn_epochs//8))
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Validation Accuracy During Training', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax3.grid(True, alpha=0.3)
    
    # 4. Train vs Validation (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    train_accs = [
        history_bp_basic['train_accs'][-1],
        history_bp_improved['train_accs'][-1],
        history_cnn['train_accs'][-1]
    ]
    val_accs = [
        history_bp_basic['val_accs'][-1],
        history_bp_improved['val_accs'][-1],
        history_cnn['val_accs'][-1]
    ]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars_train = ax4.bar(x - width/2, train_accs, width, label='Training Acc',
                        color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
    bars_val = ax4.bar(x + width/2, val_accs, width, label='Validation Acc',
                      color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels with proper spacing
    for bar in bars_train:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars_val:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Train vs Validation Accuracy', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, fontsize=11)
    ax4.legend(fontsize=11, loc='lower right')
    
    # Set dynamic y-axis limits
    min_acc_all = min(train_accs + val_accs) - 0.05
    max_acc_all = max(train_accs + val_accs) + 0.05
    ax4.set_ylim([max(0.0, min_acc_all), min(1.0, max_acc_all)])
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Save figure
    fig_path = os.path.join(RESULTS_DIR, "exp4_comprehensive_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', format='png',
               facecolor='white', edgecolor='none')
    print(f"\nComprehensive comparison figure saved to: {fig_path}")
    plt.close()
    
    # ===== Save Summary Results =====
    summary_results = {
        'comparison_type': 'Inference Only - No Training',
        'models': {
            'bp_basic': {
                'model_type': 'BP-Basic',
                'accuracy': float(acc_bp_basic),
                'f1_score': float(f1_bp_basic),
            },
            'bp_improved': {
                'model_type': 'BP-Improved',
                'accuracy': float(acc_bp_improved),
                'f1_score': float(f1_bp_improved),
            },
            'cnn_relu': {
                'model_type': 'CNN-ReLU',
                'accuracy': float(acc_cnn),
                'f1_score': float(f1_cnn),
            }
        },
        'improvements': {
            'bp_improved_vs_basic_%': float(improvement_bp_improved),
            'cnn_vs_basic_%': float(improvement_cnn),
        },
        'ranking': [
            {'rank': i+1, 'model': name, 'accuracy': float(acc)}
            for i, (name, acc, _) in enumerate(ranked)
        ]
    }
    
    results_file = os.path.join(RESULTS_DIR, "exp4_results.json")
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=4)
    print(f"Summary results saved to: {results_file}")
    
    print("\n" + "="*70)
    print("Experiment 4 Complete! (INFERENCE ONLY - No Training)")
    print(f"Total time: <2 seconds (only JSON loading + visualization)")
    print("="*70 + "\n")
    
    return summary_results


if __name__ == "__main__":
    results = main()
