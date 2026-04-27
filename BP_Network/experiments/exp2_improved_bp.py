"""
实验2: 改进BP神经网络
在基本BP网络基础上添加交叉熵损失和L2正则化，与基本模型进行对比
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from dataset.data_loader import LocalDataLoader, DataPreprocessor
from models.basic_bp import BasicBPNetwork
from models.improved_bp import ImprovedBPNetwork
from utils.metrics import Metrics, Visualizer, ModelComparator
from utils.config import *


def main():
    print("\n" + "="*60)
    print("Experiment 2: Improved BP Network (Cross-Entropy + Regularization)")
    print("="*60)
    
    # ===== Data Loading and Preprocessing =====
    print("\n[1/4] Loading and preprocessing data...")
    loader = LocalDataLoader(DATA_ROOT, img_size=IMG_SIZE)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data(
        train_per_class=TRAIN_PER_CLASS,
        val_per_class=VAL_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
        stratified=USE_STRATIFIED_SAMPLING
    )
    
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")
    
    # 数据预处理
    X_train = DataPreprocessor.normalize(X_train, method='minmax')
    X_val = DataPreprocessor.normalize(X_val, method='minmax')
    X_test = DataPreprocessor.normalize(X_test, method='minmax')
    
    y_train_onehot = DataPreprocessor.one_hot_encode(y_train, NUM_CLASSES)
    y_val_onehot = DataPreprocessor.one_hot_encode(y_val, NUM_CLASSES)
    
    # ===== 创建和训练两个模型 =====
    print("\n[2/4] 创建和训练模型...")
    
    # 基本BP网络
    print("\n  2.1) 训练基本BP网络...")
    model_basic = BasicBPNetwork(
        layer_sizes=BP_BASIC['layer_sizes'],
        learning_rate=BP_BASIC['learning_rate']
    )
    
    model_basic.train(
        X_train, y_train_onehot,
        X_val, y_val_onehot,
        epochs=BP_BASIC['epochs'],
        batch_size=BP_BASIC['batch_size'],
        early_stopping_patience=BP_BASIC['early_stopping_patience']
    )
    
    # 改进BP网络
    print("\n  2.2) 训练改进BP网络...")
    model_improved = ImprovedBPNetwork(
        layer_sizes=BP_IMPROVED['layer_sizes'],
        learning_rate=BP_IMPROVED['learning_rate'],
        lambda_reg=BP_IMPROVED['lambda_reg']
    )
    
    model_improved.train(
        X_train, y_train_onehot,
        X_val, y_val_onehot,
        epochs=BP_IMPROVED['epochs'],
        batch_size=BP_IMPROVED['batch_size'],
        early_stopping_patience=BP_IMPROVED['early_stopping_patience'],
        lr_decay=BP_IMPROVED['lr_decay']
    )
    
    # ===== 评估和对比 =====
    print("\n[3/4] 评估和对比模型...")
    
    comparator = ModelComparator()
    
    # 基本模型评估
    y_test_pred_basic = model_basic.predict(X_test)
    acc_basic = Metrics.accuracy(y_test, y_test_pred_basic)
    f1_basic = Metrics.f1_weighted(y_test, y_test_pred_basic)
    
    comparator.add_model('基本BP网络', {
        'accuracy': acc_basic,
        'f1_score': f1_basic,
        'final_train_loss': model_basic.train_losses[-1],
        'final_val_loss': model_basic.val_losses[-1],
        'final_train_acc': model_basic.train_accs[-1],
        'final_val_acc': model_basic.val_accs[-1],
    })
    
    # 改进模型评估
    y_test_pred_improved = model_improved.predict(X_test)
    acc_improved = Metrics.accuracy(y_test, y_test_pred_improved)
    f1_improved = Metrics.f1_weighted(y_test, y_test_pred_improved)
    
    comparator.add_model('改进BP网络', {
        'accuracy': acc_improved,
        'f1_score': f1_improved,
        'final_train_loss': model_improved.train_losses[-1],
        'final_val_loss': model_improved.val_losses[-1],
        'final_train_acc': model_improved.train_accs[-1],
        'final_val_acc': model_improved.val_accs[-1],
    })
    
    # 打印对比结果
    print("\n" + "-"*60)
    comparator.print_comparison()
    
    # 计算性能提升
    acc_improvement = (acc_improved - acc_basic) / acc_basic * 100
    f1_improvement = (f1_improved - f1_basic) / f1_basic * 100
    
    print(f"准确率提升: {acc_improvement:+.2f}%")
    print(f"F1分数提升: {f1_improvement:+.2f}%")
    
    # 过拟合程度对比
    print("\n过拟合程度分析:")
    basic_overfit = model_basic.val_accs[-1] - model_basic.train_accs[-1]
    improved_overfit = model_improved.val_accs[-1] - model_improved.train_accs[-1]
    
    print(f"  基本模型: 训练-验证准确率差异 = {-basic_overfit:.4f} (负值表示过拟合)")
    print(f"  改进模型: 训练-验证准确率差异 = {-improved_overfit:.4f}")
    
    # 混淆矩阵
    cm_basic = Metrics.confusion_matrix_calc(y_test, y_test_pred_basic, NUM_CLASSES)
    cm_improved = Metrics.confusion_matrix_calc(y_test, y_test_pred_improved, NUM_CLASSES)
    
    # ===== Visualization Results =====
    print("\n[4/4] Visualizing results...")
    
    # Draw comparison of training curves for two models
    fig_path = os.path.join(RESULTS_DIR, "exp2_training_comparison.png")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve - use distinct colors and markers
    axes[0, 0].plot(model_basic.train_losses, label='Basic-Train', linewidth=2.5, 
                   color='#1f77b4', marker='o', markersize=3, markevery=max(1, len(model_basic.train_losses)//15))
    axes[0, 0].plot(model_basic.val_losses, label='Basic-Val', linewidth=2.5, 
                   color='#ff7f0e', marker='s', markersize=3, markevery=max(1, len(model_basic.val_losses)//15))
    axes[0, 0].plot(model_improved.train_losses, label='Improved-Train', linewidth=2.5, 
                   color='#2ca02c', marker='^', markersize=3, markevery=max(1, len(model_improved.train_losses)//15), linestyle='--')
    axes[0, 0].plot(model_improved.val_losses, label='Improved-Val', linewidth=2.5, 
                   color='#d62728', marker='v', markersize=3, markevery=max(1, len(model_improved.val_losses)//15), linestyle='--')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Comparison (MSE vs CrossEntropy+L2)', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curve - use distinct colors and markers
    axes[0, 1].plot(model_basic.train_accs, label='Basic-Train', linewidth=2.5,
                   color='#1f77b4', marker='o', markersize=3, markevery=max(1, len(model_basic.train_accs)//15))
    axes[0, 1].plot(model_basic.val_accs, label='Basic-Val', linewidth=2.5,
                   color='#ff7f0e', marker='s', markersize=3, markevery=max(1, len(model_basic.val_accs)//15))
    axes[0, 1].plot(model_improved.train_accs, label='Improved-Train', linewidth=2.5,
                   color='#2ca02c', marker='^', markersize=3, markevery=max(1, len(model_improved.train_accs)//15), linestyle='--')
    axes[0, 1].plot(model_improved.val_accs, label='Improved-Val', linewidth=2.5,
                   color='#d62728', marker='v', markersize=3, markevery=max(1, len(model_improved.val_accs)//15), linestyle='--')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10, loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Basic model confusion matrix with values
    cm_basic_norm = cm_basic.astype('float') / cm_basic.sum(axis=1, keepdims=True)
    im1 = axes[1, 0].imshow(cm_basic_norm, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Basic BP Network Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
    axes[1, 0].set_ylabel('True Label', fontsize=11)
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_yticks(range(10))
    # Add text values
    for i in range(cm_basic.shape[0]):
        for j in range(cm_basic.shape[1]):
            text = f'{cm_basic_norm[i, j]:.2f}'
            axes[1, 0].text(j, i, text, ha='center', va='center',
                           color='white', fontsize=9, fontweight='bold')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Improved model confusion matrix with values
    cm_improved_norm = cm_improved.astype('float') / cm_improved.sum(axis=1, keepdims=True)
    im2 = axes[1, 1].imshow(cm_improved_norm, cmap='Blues', aspect='auto')
    axes[1, 1].set_title('Improved BP Network Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
    axes[1, 1].set_ylabel('True Label', fontsize=11)
    axes[1, 1].set_xticks(range(10))
    axes[1, 1].set_yticks(range(10))
    # Add text values
    for i in range(cm_improved.shape[0]):
        for j in range(cm_improved.shape[1]):
            text = f'{cm_improved_norm[i, j]:.2f}'
            axes[1, 1].text(j, i, text, ha='center', va='center',
                           color='white', fontsize=9, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight',
               format='png', facecolor='white', edgecolor='none')
    print(f"Comparison figure saved to: {fig_path}")
    plt.close()
    
    # Save comparison results
    results = {
        'basic_model': {
            'model_type': 'BP-Basic',
            'accuracy': float(acc_basic),
            'f1_score': float(f1_basic),
            'final_train_loss': float(model_basic.train_losses[-1]),
            'final_val_loss': float(model_basic.val_losses[-1]),
            'training_history': {
                'train_losses': model_basic.train_losses,
                'val_losses': model_basic.val_losses,
                'train_accs': model_basic.train_accs,
                'val_accs': model_basic.val_accs,
            }
        },
        'improved_model': {
            'model_type': 'BP-Improved',
            'accuracy': float(acc_improved),
            'f1_score': float(f1_improved),
            'final_train_loss': float(model_improved.train_losses[-1]),
            'final_val_loss': float(model_improved.val_losses[-1]),
            'training_history': {
                'train_losses': model_improved.train_losses,
                'val_losses': model_improved.val_losses,
                'train_accs': model_improved.train_accs,
                'val_accs': model_improved.val_accs,
            }
        },
        'improvement': {
            'accuracy_improvement_%': acc_improvement,
            'f1_improvement_%': f1_improvement,
        },
        'overfitting': {
            'basic_model_overfit': float(-basic_overfit),
            'improved_model_overfit': float(-improved_overfit),
        }
    }
    
    results_file = os.path.join(RESULTS_DIR, "exp2_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_file}")
    
    print("\n" + "="*60)
    print("实验2完成!")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
