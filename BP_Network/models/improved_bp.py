"""
改进BP神经网络实现
在基本BP网络基础上添加：
1. 交叉熵损失函数（替代MSE）
2. L2正则化（防止过拟合）
3. 学习率衰减
4. 批标准化（可选）
"""
import numpy as np
from typing import List, Tuple
from models.basic_bp import BasicBPNetwork


class ImprovedBPNetwork(BasicBPNetwork):
    """改进的BP神经网络"""
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01,
                 lambda_reg: float = 0.001, use_batch_norm: bool = False,
                 random_seed: int = 42):
        """
        初始化改进BP神经网络
        
        Args:
            layer_sizes: 网络层大小列表
            learning_rate: 学习率
            lambda_reg: L2正则化系数
            use_batch_norm: 是否使用批标准化
            random_seed: 随机种子
        """
        super().__init__(layer_sizes, learning_rate, random_seed)
        
        self.lambda_reg = lambda_reg
        self.use_batch_norm = use_batch_norm
        
        # 批标准化参数（如果使用）
        if use_batch_norm:
            self.gamma = [np.ones((1, size)) for size in layer_sizes[1:-1]]
            self.beta = [np.zeros((1, size)) for size in layer_sizes[1:-1]]
            self.bn_mean = [np.zeros((1, size)) for size in layer_sizes[1:-1]]
            self.bn_var = [np.ones((1, size)) for size in layer_sizes[1:-1]]
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax激活函数（多分类）"""
        # 数值稳定性处理
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(np.clip(x_shifted, -500, 500))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算交叉熵损失
        
        Args:
            y_true: 真实标签（one-hot编码）(batch_size, num_classes)
            y_pred: 预测概率 (batch_size, num_classes)
            
        Returns:
            交叉熵损失值
        """
        # 数值稳定性：避免log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    def l2_regularization_loss(self) -> float:
        """计算L2正则化项"""
        l2_loss = 0.0
        for w in self.weights:
            l2_loss += np.sum(w ** 2)
        return self.lambda_reg * l2_loss / 2
    
    def total_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算总损失 = 交叉熵损失 + L2正则化"""
        ce_loss = self.cross_entropy_loss(y_true, y_pred)
        l2_loss = self.l2_regularization_loss()
        return ce_loss + l2_loss
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        改进的前向传播
        输出层使用Softmax激活（多分类）
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
                # 输出层使用Softmax（多分类）
                a = self.softmax(z)
            
            a_list.append(a)
        
        return a, z_list, a_list
    
    def backward(self, X: np.ndarray, y: np.ndarray,
                 z_list: List[np.ndarray], a_list: List[np.ndarray]):
        """
        改进的反向传播（支持交叉熵损失和正则化）
        
        Args:
            X: 输入数据
            y: 目标标签 (one-hot编码)
            z_list: 前向传播的未激活值
            a_list: 前向传播的激活值
        """
        batch_size = X.shape[0]
        
        # 对于交叉熵损失+Softmax，输出层梯度简化为：δ^L = (a^L - y)
        delta = (a_list[-1] - y)
        
        # 反向计算每层梯度
        for i in range(self.num_layers - 2, 0, -1):
            # 计算权重和偏置梯度
            dW = np.dot(a_list[i].T, delta) / batch_size
            db = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # 加入L2正则化项
            dW += self.lambda_reg * self.weights[i]
            
            # 更新权重和偏置
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # 计算前一层的梯度
            delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(a_list[i])
        
        # 更新第一层权重和偏置
        dW = np.dot(a_list[0].T, delta) / batch_size
        db = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        # 加入L2正则化项
        dW += self.lambda_reg * self.weights[0]
        
        self.weights[0] -= self.learning_rate * dW
        self.biases[0] -= self.learning_rate * db
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              early_stopping_patience: int = 20,
              lr_decay: float = 0.99):
        """
        改进的训练函数
        
        Args:
            X_train: 训练数据
            y_train: 训练标签（one-hot编码）
            X_val: 验证数据
            y_val: 验证标签（one-hot编码）
            epochs: 训练轮数
            batch_size: 批大小
            early_stopping_patience: 早停耐心值
            lr_decay: 学习率衰减因子（每个epoch）
        """
        best_val_loss = float('inf')
        patience_counter = 0
        initial_lr = self.learning_rate
        
        for epoch in range(epochs):
            # 学习率衰减
            self.learning_rate = initial_lr * (lr_decay ** epoch)
            
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
                
                # 计算损失（包含正则化）
                batch_loss = self.total_loss(y_batch, output)
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
            val_loss = self.total_loss(y_val, y_val_pred)
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
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"LR: {self.learning_rate:.6f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发: 在epoch {epoch+1}停止训练")
                break


if __name__ == "__main__":
    # 简单测试
    from dataset.data_loader import LocalDataLoader, DataPreprocessor
    
    print("测试改进BP神经网络...")
    
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
    
    # 创建和训练改进网络
    model = ImprovedBPNetwork([X_train.shape[1], 128, 64, 10],
                              learning_rate=0.1, lambda_reg=0.0001)
    
    print("\n开始训练改进BP神经网络...")
    model.train(X_train, y_train_onehot, X_val, y_val_onehot,
                epochs=100, batch_size=32, lr_decay=0.99)
    
    # 评估
    test_acc = model.evaluate(X_test, y_test)
    print(f"\n测试准确率: {test_acc:.4f}")
