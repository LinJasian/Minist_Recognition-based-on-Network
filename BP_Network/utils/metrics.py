"""
评估指标和可视化工具
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import os
import platform

# ===== Comprehensive Matplotlib Font Configuration =====
# 解决macOS和Linux上的字体显示问题

# 设置matplotlib后端
matplotlib.use('Agg')

# 配置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.format'] = 'png'

# 确保文字正确编码
matplotlib.rcParams['axes.formatter.use_mathtext'] = False

# macOS特定配置
if platform.system() == 'Darwin':
    try:
        # macOS系统字体
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    except:
        pass

# 全局禁用任何特殊字符处理
matplotlib.rcParams['text.usetex'] = False


class Metrics:
    """模型评估指标计算"""
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算准确率"""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def confusion_matrix_calc(y_true: np.ndarray, y_pred: np.ndarray, 
                              num_classes: int = 10) -> np.ndarray:
        """计算混淆矩阵"""
        return confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    @staticmethod
    def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算宏平均F1分数"""
        return f1_score(y_true, y_pred, average='macro')
    
    @staticmethod
    def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算加权F1分数"""
        return f1_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
        """计算精确率"""
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
        """计算召回率"""
        return recall_score(y_true, y_pred, average=average, zero_division=0)
    
    @staticmethod
    def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                           num_classes: int = 10) -> np.ndarray:
        """计算每个类别的准确率"""
        class_acc = np.zeros(num_classes)
        for i in range(num_classes):
            mask = (y_true == i)
            if mask.sum() > 0:
                class_acc[i] = (y_pred[mask] == i).mean()
        return class_acc


class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_training_history(train_losses: list, val_losses: list, 
                             train_accs: list = None, val_accs: list = None,
                             save_path: str = None):
        """
        绘制训练曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_accs: 训练准确率列表（可选）
            val_accs: 验证准确率列表（可选）
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=(14, 5))
        
        if train_accs is None:
            axes = [axes]
        
        # Loss曲线
        axes[0].plot(train_losses, label='Training Loss', linewidth=2)
        axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线（如果有）
        if train_accs is not None:
            axes[1].plot(train_accs, label='Training Accuracy', linewidth=2)
            axes[1].plot(val_accs, label='Validation Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Training and Validation Accuracy', fontsize=14)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Ensure proper font rendering when saving
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       format='png', facecolor='white', edgecolor='none')
            print(f"Training curve saved to: {save_path}")
        
        plt.show()
        plt.close('all')
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, save_path: str = None, 
                             normalize: bool = True):
        """
        绘制混淆矩阵，使用seaborn heatmap获得更好的可视化效果
        
        Args:
            cm: 混淆矩阵
            save_path: 保存路径（可选）
            normalize: 是否归一化显示
        """
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            cm_display = cm
            fmt = 'd'
        
        plt.figure(figsize=(11, 9))
        
        # Use seaborn heatmap for better visualization
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10),
                   cbar_kws={'label': 'Frequency'},
                   linewidths=0.5, linecolor='gray',
                   vmin=0, vmax=1 if normalize else None)
        
        plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
        plt.ylabel('True Label', fontsize=13, fontweight='bold')
        title = 'Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix'
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       format='png', facecolor='white', edgecolor='none')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        plt.close('all')
    
    @staticmethod
    def plot_per_class_accuracy(accuracies: np.ndarray, save_path: str = None):
        """
        绘制每个类别的准确率
        
        Args:
            accuracies: 每个类别的准确率数组 (10,)
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(10), accuracies, color='steelblue', alpha=0.8)
        
        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Digit Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=14)
        plt.xticks(range(10))
        plt.ylim([0, 1.1])
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       format='png', facecolor='white', edgecolor='none')
            print(f"Per-class accuracy saved to: {save_path}")
        
        plt.show()
        plt.close('all')
    
    @staticmethod
    def plot_sample_images(X: np.ndarray, y: np.ndarray, img_shape: tuple = (32, 32),
                          num_samples: int = 10, save_path: str = None):
        """
        绘制样本图像
        
        Args:
            X: 图像数据 (N, H*W)
            y: 标签数据 (N,)
            img_shape: 原始图像形状
            num_samples: 显示的样本数
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(X))):
            img = X[i].reshape(img_shape)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {y[i]}', fontsize=11)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample images saved to: {save_path}")
        
        plt.show()


class ModelComparator:
    """模型对比工具"""
    
    def __init__(self):
        self.results = {}
    
    def add_model(self, model_name: str, metrics_dict: dict):
        """
        添加模型结果
        
        Args:
            model_name: 模型名称
            metrics_dict: 指标字典，如 {'accuracy': 0.95, 'f1': 0.94, ...}
        """
        self.results[model_name] = metrics_dict
    
    def compare(self) -> dict:
        """比较所有模型的性能"""
        return self.results
    
    def print_comparison(self):
        """打印对比结果"""
        if not self.results:
            print("No model results to compare")
            return
        
        print("\n" + "="*60)
        print("Model Performance Comparison Table")
        print("="*60)
        
        # 获取所有指标名称
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(list(all_metrics))
        
        # 打印表头
        header = f"{'Model':<20}" + "".join(f"{m:<15}" for m in all_metrics)
        print(header)
        print("-" * len(header))
        
        # 打印每行数据
        for model_name, metrics in self.results.items():
            row = f"{model_name:<20}"
            for metric in all_metrics:
                value = metrics.get(metric, 'N/A')
                if isinstance(value, float):
                    row += f"{value:<15.4f}"
                else:
                    row += f"{str(value):<15}"
            print(row)
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # 测试代码
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 0, 2, 1, 1, 2, 0])
    
    acc = Metrics.accuracy(y_true, y_pred)
    print(f"准确率: {acc:.4f}")
    
    f1 = Metrics.f1_weighted(y_true, y_pred)
    print(f"加权F1: {f1:.4f}")
    
    cm = Metrics.confusion_matrix_calc(y_true, y_pred, num_classes=3)
    print(f"混淆矩阵:\n{cm}")
