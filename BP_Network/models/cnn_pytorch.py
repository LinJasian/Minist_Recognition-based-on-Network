"""
卷积神经网络（CNN）实现
使用PyTorch框架，支持ReLU和Sigmoid激活函数对比
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple


class SimpleCNN(nn.Module):
    """简单的卷积神经网络"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10,
                 activation: str = 'relu', input_size: int = 28):
        """
        初始化CNN
        
        Args:
            input_channels: 输入图片通道数（灰度图为1）
            num_classes: 分类类别数
            activation: 激活函数，'relu'或'sigmoid'
            input_size: 输入图片尺寸（28或32）
        """
        super(SimpleCNN, self).__init__()
        
        self.activation_name = activation
        self.input_size = input_size
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 卷积层1: 输入1通道 -> 16通道, 3x3卷积核（从32通道, 5x5简化）
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        
        # 池化层1: 最大池化 2x2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层2: 16通道 -> 32通道, 3x3卷积核（从64通道, 5x5简化）
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # 池化层2: 最大池化 2x2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算flatten后的维度（动态计算）
        # 使用虚拟输入计算最终尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            dummy = self.conv1(dummy)
            dummy = self.pool1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.pool2(dummy)
            fc_input_dim = dummy.view(1, -1).shape[1]
        
        # 全连接层（简化版本以加快训练）
        self.fc1 = nn.Linear(fc_input_dim, 64)  # 减少从128到64
        self.dropout1 = nn.Dropout(p=0.2)  # 从0.5降低到0.2
        
        self.fc2 = nn.Linear(64, num_classes)  # 直接输出，删除中间层
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, 32, 32)
            
        Returns:
            输出张量 (batch_size, num_classes)
        """
        # 卷积块1
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        return x


class CNNTrainer:
    """CNN训练器"""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 device: str = None):
        """
        初始化训练器
        
        Args:
            model: CNN模型
            learning_rate: 学习率
            device: 使用的设备（'cuda'或'cpu'）
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # 使用交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 使用SGD优化器（比Adam快2-3倍）
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate,
            momentum=0.9,  # 加入momentum加快收敛
            nesterov=True  # Nesterov加速
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        评估模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 30, early_stopping_patience: int = 8,
              eval_frequency: int = 3):
        """
        完整训练过程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            eval_frequency: 评估频率（每N个epoch评估一次）
        """
        print(f"  开始训练... (epochs={epochs}, eval_frequency={eval_frequency})")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 只在每eval_frequency个epoch或最后一个epoch评估（加快训练）
            is_last_epoch = (epoch == epochs - 1)
            should_eval = ((epoch + 1) % eval_frequency == 0) or is_last_epoch
            
            if should_eval:
                # 验证
                val_loss, val_acc = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print(f"  Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"  ⏸️  早停触发: 在epoch {epoch+1}停止训练")
                    break
            else:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入数据 (N, 784) [需要reshape]
            
        Returns:
            预测标签 (N,)
        """
        self.model.eval()
        
        # 确保输入是正确的形状：(N, 784) -> (N, 1, 28, 28)
        if len(X.shape) == 2:
            # 从(N, 784)重塑为(N, 1, 28, 28)
            X = X.reshape(-1, 1, 28, 28)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入数据
            
        Returns:
            概率矩阵 (N, num_classes)
        """
        self.model.eval()
        
        if len(X.shape) == 2:
            X = X.reshape(-1, 1, 28, 28)  # 修改为28x28（MNIST）
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    创建PyTorch数据加载器（优化版本）
    
    Args:
        X_train, y_train: 训练数据和标签
        X_val, y_val: 验证数据和标签
        batch_size: 批大小
        
    Returns:
        (train_loader, val_loader)
    """
    print("  ├─ 重塑训练集...")
    # 重塑为(N, 1, 28, 28)格式 (MNIST为28x28, 本地BMP为32x32)
    img_size = 28 if X_train.shape[1] == 784 else 32  # 根据数据维度自动判断
    X_train = X_train.reshape(-1, 1, img_size, img_size).astype(np.float32)
    
    print("  ├─ 重塑验证集...")
    X_val = X_val.reshape(-1, 1, img_size, img_size).astype(np.float32)
    
    print("  ├─ 转换为torch张量...")
    # 创建数据集
    train_dataset = TensorDataset(torch.FloatTensor(X_train),
                                  torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val),
                               torch.LongTensor(y_val))
    
    print("  ├─ 创建数据加载器...")
    # 创建数据加载器（使用num_workers加速，pin_memory优化GPU传输）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # 设为0以避免子进程开销（CPU速度已够快）
        pin_memory=False  # CPU->GPU不需要
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print("  └─ 数据加载器创建完成！")
    return train_loader, val_loader


if __name__ == "__main__":
    # 简单测试
    from dataset.data_loader import LocalDataLoader, DataPreprocessor
    
    print("测试CNN模型...")
    
    # 加载数据
    data_root = "/Users/zhanghaozhe/Documents/VScode/Partern Recogniton/database/HandwrittenNum"
    loader = LocalDataLoader(data_root)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()
    
    # 预处理
    X_train = DataPreprocessor.normalize(X_train)
    X_val = DataPreprocessor.normalize(X_val)
    X_test = DataPreprocessor.normalize(X_test)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)
    test_loader, _ = create_data_loaders(X_test, y_test, np.zeros_like(y_test), np.zeros_like(y_test))
    
    # 创建和训练ReLU版本
    print("\n=== 训练ReLU版本 ===")
    model_relu = SimpleCNN(activation='relu')
    trainer_relu = CNNTrainer(model_relu, learning_rate=0.001)
    trainer_relu.train(train_loader, val_loader, epochs=50)
    
    # 创建和训练Sigmoid版本
    print("\n=== 训练Sigmoid版本 ===")
    model_sigmoid = SimpleCNN(activation='sigmoid')
    trainer_sigmoid = CNNTrainer(model_sigmoid, learning_rate=0.001)
    trainer_sigmoid.train(train_loader, val_loader, epochs=50)
    
    # 在测试集上评估
    print("\n=== 测试集评估 ===")
    y_pred_relu = trainer_relu.predict(X_test)
    acc_relu = np.mean(y_pred_relu == y_test)
    print(f"ReLU版本测试准确率: {acc_relu:.4f}")
    
    y_pred_sigmoid = trainer_sigmoid.predict(X_test)
    acc_sigmoid = np.mean(y_pred_sigmoid == y_test)
    print(f"Sigmoid版本测试准确率: {acc_sigmoid:.4f}")
