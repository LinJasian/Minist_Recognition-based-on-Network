"""
基本BP神经网络实现
使用梯度下降和反向传播训练，激活函数为Sigmoid，损失函数为MSE
"""
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


class BasicBPNetwork:
    """基本的BP神经网络"""
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01,
                 random_seed: int = 42):
        """
        初始化BP神经网络
        
        Args:
            layer_sizes: 网络层大小列表，如[784, 128, 64, 10]
            learning_rate: 学习率
            random_seed: 随机种子，确保可复现性
        """
        np.random.seed(random_seed)
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier初始化
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * \
                np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid导数"""
        return x * (1 - x)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        前向传播
        
        Args:
            X: 输入数据 (batch_size, input_dim)
            
        Returns:
            output: 输出 (batch_size, output_dim)
            z_list: 未激活值列表
            a_list: 激活值列表
        """
        a = X
        a_list = [a]
        z_list = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            z_list.append(z)
            
            if i < self.num_layers - 2:
                # 隐层使用Sigmoid
                a = self.sigmoid(z)
            else:
                # 输出层使用Sigmoid（多分类情形）
                a = self.sigmoid(z)
            
            a_list.append(a)
        
        return a, z_list, a_list
    
    def backward(self, X: np.ndarray, y: np.ndarray, 
                 z_list: List[np.ndarray], a_list: List[np.ndarray]):
        """
        反向传播
        
        Args:
            X: 输入数据
            y: 目标标签 (one-hot编码)
            z_list: 前向传播的未激活值
            a_list: 前向传播的激活值
        """
        batch_size = X.shape[0]
        
        # 输出层梯度：δ^L = (a^L - y) * sigmoid'(z^L)
        delta = (a_list[-1] - y) * self.sigmoid_derivative(a_list[-1])
        
        # 反向计算每层梯度
        for i in range(self.num_layers - 2, 0, -1):
            # 计算权重和偏置梯度
            dW = np.dot(a_list[i].T, delta) / batch_size
            db = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # 更新权重和偏置
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # 计算前一层的梯度
            delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(a_list[i])
        
        # 更新第一层权重和偏置
        dW = np.dot(a_list[0].T, delta) / batch_size
        db = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        self.weights[0] -= self.learning_rate * dW
        self.biases[0] -= self.learning_rate * db
    
    def mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算MSE损失"""
        return np.mean((y_pred - y_true) ** 2)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              early_stopping_patience: int = 20):
        """
        训练网络
        
        Args:
            X_train: 训练数据 (N, input_dim)
            y_train: 训练标签 (N, output_dim，one-hot编码)
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            early_stopping_patience: 早停耐心值
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = 0.0
            num_batches = 0
            
            # 随机打乱训练数据
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 批量训练
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # 前向传播
                output, z_list, a_list = self.forward(X_batch)
                
                # 计算损失
                batch_loss = self.mse_loss(y_batch, output)
                train_loss += batch_loss
                num_batches += 1
                
                # 反向传播
                self.backward(X_batch, y_batch, z_list, a_list)
            
            # 计算平均训练损失
            avg_train_loss = train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # 计算训练准确率
            y_train_pred, _, _ = self.forward(X_train)
            y_train_pred_labels = np.argmax(y_train_pred, axis=1)
            y_train_labels = np.argmax(y_train, axis=1)
            train_acc = np.mean(y_train_pred_labels == y_train_labels)
            self.train_accs.append(train_acc)
            
            # 验证
            y_val_pred, _, _ = self.forward(X_val)
            val_loss = self.mse_loss(y_val, y_val_pred)
            self.val_losses.append(val_loss)
            
            # 计算验证准确率
            y_val_pred_labels = np.argmax(y_val_pred, axis=1)
            y_val_labels = np.argmax(y_val, axis=1)
            val_acc = np.mean(y_val_pred_labels == y_val_labels)
            self.val_accs.append(val_acc)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发: 在epoch {epoch+1}停止训练")
                break
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入数据 (N, input_dim)
            
        Returns:
            预测标签 (N,)
        """
        output, _, _ = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入数据 (N, input_dim)
            
        Returns:
            概率矩阵 (N, num_classes)
        """
        output, _, _ = self.forward(X)
        return output
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        评估模型准确率
        
        Args:
            X: 测试数据
            y: 测试标签（标量形式，非one-hot）
            
        Returns:
            准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_weights_norms(self) -> List[float]:
        """获取各层权重的范数"""
        norms = []
        for w in self.weights:
            norms.append(np.linalg.norm(w))
        return norms


if __name__ == "__main__":
    # 简单测试
    from dataset.data_loader import LocalDataLoader, DataPreprocessor
    
    print("测试基本BP神经网络...")
    
    # 加载数据
    data_root = "/Users/zhanghaozhe/Documents/VScode/Partern Recogniton/database/HandwrittenNum"
    loader = LocalDataLoader(data_root)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()
    
    # 预处理
    X_train = DataPreprocessor.normalize(X_train)
    X_val = DataPreprocessor.normalize(X_val)
    X_test = DataPreprocessor.normalize(X_test)
    
    y_train_onehot = DataPreprocessor.one_hot_encode(y_train)
    y_val_onehot = DataPreprocessor.one_hot_encode(y_val)
    
    # 创建和训练网络
    model = BasicBPNetwork([X_train.shape[1], 128, 64, 10], learning_rate=0.1)
    
    print("\n开始训练基本BP神经网络...")
    model.train(X_train, y_train_onehot, X_val, y_val_onehot,
                epochs=100, batch_size=32)
    
    # 评估
    test_acc = model.evaluate(X_test, y_test)
    print(f"\n测试准确率: {test_acc:.4f}")
