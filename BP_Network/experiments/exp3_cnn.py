"""
实验3: 卷积神经网络（CNN）
实现CNN模型，与BP网络进行对比
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from dataset.data_loader import LocalDataLoader, DataPreprocessor
from models.cnn_pytorch import SimpleCNN, CNNTrainer, create_data_loaders
from utils.metrics import Metrics, Visualizer, ModelComparator
from utils.config import *


def main():
    print("\n" + "="*60)
    print("Experiment 3: Convolutional Neural Network (CNN)")
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
    
    # 预处理
    X_train = DataPreprocessor.normalize(X_train, method='minmax')
    X_val = DataPreprocessor.normalize(X_val, method='minmax')
    X_test = DataPreprocessor.normalize(X_test, method='minmax')
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=CNN_CONFIG['batch_size']
    )
    
    # ===== 创建和训练CNN模型 =====
    print("\n[2/4] 创建和训练CNN模型...")
    
    # ReLU版本
    print("\n  2.1) 训练CNN-ReLU版本...")
    model_relu = SimpleCNN(activation='relu')
    trainer_relu = CNNTrainer(model_relu, learning_rate=CNN_CONFIG['learning_rate'])
    
    trainer_relu.train(
        train_loader, val_loader,
        epochs=CNN_CONFIG['epochs'],
        early_stopping_patience=CNN_CONFIG['early_stopping_patience']
    )
    
    # Sigmoid版本
    print("\n  2.2) 训练CNN-Sigmoid版本...")
    model_sigmoid = SimpleCNN(activation='sigmoid')
    trainer_sigmoid = CNNTrainer(model_sigmoid, learning_rate=CNN_CONFIG['learning_rate'])
    
    trainer_sigmoid.train(
        train_loader, val_loader,
        epochs=CNN_CONFIG['epochs'],
        early_stopping_patience=CNN_CONFIG['early_stopping_patience']
    )
    
    # ===== 评估模型 =====
    print("\n[3/4] 评估模型性能...")
    
    # ReLU版本评估
    y_test_pred_relu = trainer_relu.predict(X_test)
    acc_relu = Metrics.accuracy(y_test, y_test_pred_relu)
    f1_relu = Metrics.f1_weighted(y_test, y_test_pred_relu)
    
    # Sigmoid版本评估
    y_test_pred_sigmoid = trainer_sigmoid.predict(X_test)
    acc_sigmoid = Metrics.accuracy(y_test, y_test_pred_sigmoid)
    f1_sigmoid = Metrics.f1_weighted(y_test, y_test_pred_sigmoid)
    
    # 每类准确率
    per_class_acc_relu = Metrics.per_class_accuracy(y_test, y_test_pred_relu, NUM_CLASSES)
    per_class_acc_sigmoid = Metrics.per_class_accuracy(y_test, y_test_pred_sigmoid, NUM_CLASSES)
    
    # 混淆矩阵
    cm_relu = Metrics.confusion_matrix_calc(y_test, y_test_pred_relu, NUM_CLASSES)
    cm_sigmoid = Metrics.confusion_matrix_calc(y_test, y_test_pred_sigmoid, NUM_CLASSES)
    
    # 打印结果
    print("\n" + "-"*60)
    print("性能指标对比:")
    print("-"*60)
    
    comparator = ModelComparator()
    comparator.add_model('CNN-ReLU', {
        'accuracy': acc_relu,
        'f1_score': f1_relu,
        'final_train_loss': trainer_relu.train_losses[-1],
        'final_val_loss': trainer_relu.val_losses[-1],
        'final_train_acc': trainer_relu.train_accs[-1],
        'final_val_acc': trainer_relu.val_accs[-1],
    })
    comparator.add_model('CNN-Sigmoid', {
        'accuracy': acc_sigmoid,
        'f1_score': f1_sigmoid,
        'final_train_loss': trainer_sigmoid.train_losses[-1],
        'final_val_loss': trainer_sigmoid.val_losses[-1],
        'final_train_acc': trainer_sigmoid.train_accs[-1],
        'final_val_acc': trainer_sigmoid.val_accs[-1],
    })
    comparator.print_comparison()
    
    # 计算激活函数效果对比
    activation_improvement = (acc_relu - acc_sigmoid) / acc_sigmoid * 100
    print(f"\nReLU相对于Sigmoid的准确率改善: {activation_improvement:+.2f}%")
    
    # 收敛速度分析
    print("\n收敛性分析:")
    print(f"  CNN-ReLU - 收敛到最佳验证损失需要 {np.argmin(trainer_relu.val_losses)+1} epochs")
    print(f"  CNN-Sigmoid - 收敛到最佳验证损失需要 {np.argmin(trainer_sigmoid.val_losses)+1} epochs")
    
    # ===== 可视化结果 =====
    print("\n[4/4] 可视化结果...")
    
    # 绘制ReLU版本的训练曲线
    save_path_relu = os.path.join(RESULTS_DIR, "exp3_relu_training_curve.png")
    Visualizer.plot_training_history(
        trainer_relu.train_losses, trainer_relu.val_losses,
        trainer_relu.train_accs, trainer_relu.val_accs,
        save_path=save_path_relu
    )
    
    # 绘制Sigmoid版本的训练曲线
    save_path_sigmoid = os.path.join(RESULTS_DIR, "exp3_sigmoid_training_curve.png")
    Visualizer.plot_training_history(
        trainer_sigmoid.train_losses, trainer_sigmoid.val_losses,
        trainer_sigmoid.train_accs, trainer_sigmoid.val_accs,
        save_path=save_path_sigmoid
    )
    
    # 绘制激活函数对比曲线
    fig_path = os.path.join(RESULTS_DIR, "exp3_activation_comparison.png")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss curve comparison with distinct colors and markers
    axes[0].plot(trainer_relu.val_losses, label='CNN-ReLU', linewidth=2.5, 
                color='#1f77b4', marker='o', markersize=4, 
                markevery=max(1, len(trainer_relu.val_losses)//12))
    axes[0].plot(trainer_sigmoid.val_losses, label='CNN-Sigmoid', linewidth=2.5,
                color='#d62728', marker='s', markersize=4,
                markevery=max(1, len(trainer_sigmoid.val_losses)//12), linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Activation Function Impact on Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve comparison with distinct colors and markers
    axes[1].plot(trainer_relu.val_accs, label='CNN-ReLU', linewidth=2.5,
                color='#1f77b4', marker='o', markersize=4,
                markevery=max(1, len(trainer_relu.val_accs)//12))
    axes[1].plot(trainer_sigmoid.val_accs, label='CNN-Sigmoid', linewidth=2.5,
                color='#d62728', marker='s', markersize=4,
                markevery=max(1, len(trainer_sigmoid.val_accs)//12), linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1].set_title('Activation Function Impact on Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"激活函数对比图已保存到: {fig_path}")
    plt.close()
    
    # 绘制每类准确率对比
    fig_path_per_class = os.path.join(RESULTS_DIR, "exp3_per_class_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(NUM_CLASSES)
    width = 0.35
    
    ax.bar(x - width/2, per_class_acc_relu, width, label='CNN-ReLU', alpha=0.8)
    ax.bar(x + width/2, per_class_acc_sigmoid, width, label='CNN-Sigmoid', alpha=0.8)
    
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy Comparison (ReLU vs Sigmoid)')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_path_per_class, dpi=150, bbox_inches='tight',
               format='png', facecolor='white', edgecolor='none')
    print(f"Per-class accuracy comparison figure saved to: {fig_path_per_class}")
    plt.close()
    
    # Save results
    results = {
        'cnn_relu': {
            'model_type': 'CNN-ReLU',
            'accuracy': float(acc_relu),
            'f1_score': float(f1_relu),
            'final_train_loss': float(trainer_relu.train_losses[-1]),
            'final_val_loss': float(trainer_relu.val_losses[-1]),
            'convergence_epoch': int(np.argmin(trainer_relu.val_losses)+1),
            'per_class_accuracy': per_class_acc_relu.tolist(),
            'training_history': {
                'train_losses': trainer_relu.train_losses,
                'val_losses': trainer_relu.val_losses,
                'train_accs': trainer_relu.train_accs,
                'val_accs': trainer_relu.val_accs,
            }
        },
        'cnn_sigmoid': {
            'model_type': 'CNN-Sigmoid',
            'accuracy': float(acc_sigmoid),
            'f1_score': float(f1_sigmoid),
            'final_train_loss': float(trainer_sigmoid.train_losses[-1]),
            'final_val_loss': float(trainer_sigmoid.val_losses[-1]),
            'convergence_epoch': int(np.argmin(trainer_sigmoid.val_losses)+1),
            'per_class_accuracy': per_class_acc_sigmoid.tolist(),
            'training_history': {
                'train_losses': trainer_sigmoid.train_losses,
                'val_losses': trainer_sigmoid.val_losses,
                'train_accs': trainer_sigmoid.train_accs,
                'val_accs': trainer_sigmoid.val_accs,
            }
        },
        'activation_comparison': {
            'relu_vs_sigmoid_improvement_%': activation_improvement,
        }
    }
    
    results_file = os.path.join(RESULTS_DIR, "exp3_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*60)
    print("实验3完成!")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
