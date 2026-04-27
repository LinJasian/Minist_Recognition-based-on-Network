"""
数据加载器 - 支持本地BMP图片和MNIST数据集
"""
import os
import numpy as np
from PIL import Image
from typing import Tuple
import pickle
import gzip


class LocalDataLoader:
    """本地手写数字数据集加载器（BMP格式）"""
    
    def __init__(self, data_root: str, img_size: Tuple[int, int] = (32, 32)):
        """
        初始化本地数据加载器
        
        Args:
            data_root: 数据集根目录，应包含 0-9 和 数字 子文件夹
            img_size: 图片缩放目标尺寸
        """
        self.data_root = data_root
        self.img_size = img_size
        
    def load_data(self, train_per_class: int = 88, 
                  val_per_class: int = 19,
                  test_per_class: int = 19,
                  stratified: bool = True) -> Tuple[np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
        """
        加载本地数据集，支持分层抽样
        
        Args:
            train_per_class: 每个类别的训练集数量 (70%)
            val_per_class: 每个类别的验证集数量 (15%)
            test_per_class: 每个类别的测试集数量 (15%)
            stratified: 是否使用分层抽样确保各集合中类别分布均衡
            
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
            其中X为(N, H*W)的扁平化数组，y为(N,)的标签数组
        """
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        total_per_class = train_per_class + val_per_class + test_per_class
        
        # 遍历0-9数字文件夹
        for digit in range(10):
            digit_dir = os.path.join(self.data_root, str(digit))
            if not os.path.exists(digit_dir):
                print(f"Warning: {digit_dir} does not exist")
                continue
            
            # 获取该文件夹下的所有图片
            images = sorted([f for f in os.listdir(digit_dir) if f.endswith('.bmp')])
            
            if stratified:
                # 分层抽样：均匀分配数据
                selected_indices = np.random.choice(len(images), 
                                                   min(total_per_class, len(images)), 
                                                   replace=False)
                selected_images = [images[i] for i in sorted(selected_indices)]
            else:
                # 顺序选择：1-xxx为训练集，xxx+1-yyy为验证集，yyy+1-zzz为测试集
                selected_images = images[:total_per_class]
            
            # 按比例分割
            train_split = selected_images[:train_per_class]
            val_split = selected_images[train_per_class:train_per_class + val_per_class]
            test_split = selected_images[train_per_class + val_per_class:total_per_class]
            
            # 加载训练集
            for img_file in train_split:
                img_path = os.path.join(digit_dir, img_file)
                img_array = self._load_image(img_path)
                X_train.append(img_array.flatten())
                y_train.append(digit)
            
            # 加载验证集
            for img_file in val_split:
                img_path = os.path.join(digit_dir, img_file)
                img_array = self._load_image(img_path)
                X_val.append(img_array.flatten())
                y_val.append(digit)
            
            # 加载测试集
            for img_file in test_split:
                img_path = os.path.join(digit_dir, img_file)
                img_array = self._load_image(img_path)
                X_test.append(img_array.flatten())
                y_test.append(digit)
        
        return (np.array(X_train), np.array(y_train),
                np.array(X_val), np.array(y_val),
                np.array(X_test), np.array(y_test))
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        加载单张BMP图片并转换为灰度数组
        
        Args:
            img_path: 图片文件路径
            
        Returns:
            (H, W)的灰度数组，像素值范围[0, 255]
        """
        img = Image.open(img_path).convert('L')  # 转换为灰度图
        img_resized = img.resize(self.img_size)  # 缩放
        return np.array(img_resized, dtype=np.float32)


class MNISTDataLoader:
    """MNIST官方数据集加载器（mnist.pkl.gz格式）"""
    
    @staticmethod
    def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
        """
        从pkl.gz文件加载MNIST数据集
        
        Args:
            data_path: mnist.pkl.gz文件路径
            
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
            其中X为(N, 784)的数组，y为(N,)的标签数组
        """
        with gzip.open(data_path, 'rb') as f:
            train_set, val_set, test_set = pickle.load(f, encoding='latin1')
        
        X_train, y_train = train_set
        X_val, y_val = val_set
        X_test, y_test = test_set
        
        return X_train, y_train, X_val, y_val, X_test, y_test


class DataPreprocessor:
    """数据预处理工具"""
    
    @staticmethod
    def normalize(X: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        数据标准化
        
        Args:
            X: 输入数据 (N, D)
            method: 标准化方法，'minmax'或'zscore'
            
        Returns:
            标准化后的数据
        """
        if method == 'minmax':
            # 归一化到[0, 1]
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            return (X - X_min) / (X_max - X_min + 1e-8)
        
        elif method == 'zscore':
            # z-score标准化
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True)
            return (X - mean) / (std + 1e-8)
        else:
            raise ValueError(f"未知的标准化方法: {method}")
    
    @staticmethod
    def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """
        One-hot编码
        
        Args:
            y: 标签数组 (N,)，值范围[0, num_classes-1]
            num_classes: 类别数量
            
        Returns:
            One-hot编码矩阵 (N, num_classes)
        """
        n = y.shape[0]
        one_hot = np.zeros((n, num_classes))
        one_hot[np.arange(n), y.astype(int)] = 1
        return one_hot
    
    @staticmethod
    def one_hot_decode(y_one_hot: np.ndarray) -> np.ndarray:
        """
        One-hot解码
        
        Args:
            y_one_hot: One-hot编码矩阵 (N, num_classes)
            
        Returns:
            标签数组 (N,)
        """
        return np.argmax(y_one_hot, axis=1)


if __name__ == "__main__":
    # 测试本地数据加载器
    data_root = "/Users/zhanghaozhe/Documents/VScode/Partern Recogniton/database/HandwrittenNum"
    loader = LocalDataLoader(data_root)
    
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()
    
    print("数据加载完成!")
    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"验证集: {X_val.shape}, 标签: {y_val.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"特征维度: {X_train.shape[1]}")
    print(f"像素值范围: [{X_train.min()}, {X_train.max()}]")
    
    # 测试预处理
    X_train_norm = DataPreprocessor.normalize(X_train)
    y_train_onehot = DataPreprocessor.one_hot_encode(y_train)
    
    print(f"\n标准化后像素值范围: [{X_train_norm.min()}, {X_train_norm.max()}]")
    print(f"One-hot编码形状: {y_train_onehot.shape}")
