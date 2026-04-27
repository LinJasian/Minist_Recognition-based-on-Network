"""
实验1: 基本BP神经网络 (使用标准MNIST数据集)
训练基本BP网络，评估其在手写数字识别任务上的性能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataset.data_loader import load_dataset, DataPreprocessor
from models.basic_bp import BasicBPNetwork
from utils.metrics import Metrics, Visualizer, ModelComparator
from utils.config import *


def main():
    print("\n" + "="*60)
    print("Experiment 1: Basic BP Neural Network")
    print("="*60)
    
    # ===== Data Loading and Preprocessing =====
    print("\n[1/4] Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(use_mnist=USE_MNIST)
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Data preprocessing
    X_train = DataPreprocessor.normalize(X_train, method='minmax')
    X_val = DataPreprocessor.normalize(X_val, method='minmax')
    X_test = DataPreprocessor.normalize(X_test, method='minmax')
    
    y_train_onehot = DataPreprocessor.one_hot_encode(y_train, NUM_CLASSES)
    y_val_onehot = DataPreprocessor.one_hot_encode(y_val, NUM_CLASSES)
    
    # ===== Create and Train Model =====
    print("\n[2/4] Creating and training basic BP neural network...")
    model = BasicBPNetwork(
        layer_sizes=BP_BASIC['layer_sizes'],
        learning_rate=BP_BASIC['learning_rate']
    )
    
    model.train(
        X_train, y_train_onehot,
        X_val, y_val_onehot,
        epochs=BP_BASIC['epochs'],
        batch_size=BP_BASIC['batch_size'],
        early_stopping_patience=BP_BASIC['early_stopping_patience']
    )
    
    # ===== 评估模型 =====
    print("\n[3/4] Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_acc = Metrics.accuracy(y_train, y_train_pred)
    val_acc = Metrics.accuracy(y_val, y_val_pred)
    test_acc = Metrics.accuracy(y_test, y_test_pred)
    
    test_f1 = Metrics.f1_weighted(y_test, y_test_pred)
    test_precision = Metrics.precision(y_test, y_test_pred)
    test_recall = Metrics.recall(y_test, y_test_pred)
    
    per_class_acc = Metrics.per_class_accuracy(y_test, y_test_pred, NUM_CLASSES)
    
    # Confusion matrix
    cm = Metrics.confusion_matrix_calc(y_test, y_test_pred, NUM_CLASSES)
    
    # Print results
    print("\n" + "-"*60)
    print("Performance Metrics:")
    print("-"*60)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score (weighted): {test_f1:.4f}")
    print(f"Test Precision (macro): {test_precision:.4f}")
    print(f"Test Recall (macro): {test_recall:.4f}")
    
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Digit {i}: {acc:.4f}")
    
    # ===== Visualization Results =====
    print("\n[4/4] Visualizing results...")
    
    # 绘制训练曲线
    save_path_loss = os.path.join(RESULTS_DIR, "exp1_loss_curve.png")
    Visualizer.plot_training_history(
        model.train_losses, model.val_losses,
        model.train_accs, model.val_accs,
        save_path=save_path_loss
    )
    
    # 绘制混淆矩阵
    save_path_cm = os.path.join(RESULTS_DIR, "exp1_confusion_matrix.png")
    Visualizer.plot_confusion_matrix(cm, save_path=save_path_cm)
    
    # 绘制每类准确率
    save_path_per_class = os.path.join(RESULTS_DIR, "exp1_per_class_accuracy.png")
    Visualizer.plot_per_class_accuracy(per_class_acc, save_path=save_path_per_class)
    
    # ===== Save Results =====
    results = {
        'model_type': 'BP-Basic',
        'config': {
            'layer_sizes': BP_BASIC['layer_sizes'],
            'learning_rate': BP_BASIC['learning_rate'],
            'batch_size': BP_BASIC['batch_size'],
            'epochs': BP_BASIC['epochs'],
            'early_stopping_patience': BP_BASIC['early_stopping_patience'],
        },
        'metrics': {
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'per_class_acc': per_class_acc.tolist(),
        },
        'training_history': {
            'train_losses': model.train_losses,
            'val_losses': model.val_losses,
            'train_accs': model.train_accs,
            'val_accs': model.val_accs,
        },
        'confusion_matrix': cm.tolist()
    }
    
    import json
    results_file = os.path.join(RESULTS_DIR, "exp1_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*60)
    print("Experiment 1 Complete!")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
