"""
项目超参数配置
"""
import os

# ===== 数据相关配置 =====
DATA_ROOT = "/Users/zhanghaozhe/Documents/VScode/Partern Recogniton/database/HandwrittenNum"
IMG_SIZE = (32, 32)
NUM_CLASSES = 10

# 本地数据集划分 - 优化后的比例 (70% train, 15% val, 15% test)
TRAIN_PER_CLASS = 88    # 每类训练样本数 (70%)
VAL_PER_CLASS = 19      # 每类验证样本数 (15%)
TEST_PER_CLASS = 19     # 每类测试样本数 (15%) - 使用分层抽样
USE_STRATIFIED_SAMPLING = True  # 启用分层抽样

# ===== BP网络配置 =====
# 基本BP网络
BP_BASIC = {
    'layer_sizes': [1024, 128, 64, 10],  # 输入层, 隐层1, 隐层2, 输出层
    'learning_rate': 0.1,
    'batch_size': 32,
    'epochs': 200,
    'early_stopping_patience': 20,
}

# 改进BP网络
BP_IMPROVED = {
    'layer_sizes': [1024, 128, 64, 10],
    'learning_rate': 0.1,
    'lambda_reg': 0.0001,  # L2正则化系数
    'batch_size': 32,
    'epochs': 200,
    'early_stopping_patience': 20,
    'lr_decay': 0.99,  # 学习率衰减
}

# ===== CNN配置 =====
CNN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'early_stopping_patience': 10,
}

# ===== 优化器配置 =====
OPTIMIZERS = {
    'SGD': {'lr': 0.01, 'momentum': 0.9},
    'Adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
    'RMSprop': {'lr': 0.001, 'alpha': 0.99},
}

# ===== 输出路径 =====
PROJECT_ROOT = "/Users/zhanghaozhe/Documents/VScode/Partern Recogniton/BP_Network"
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 创建输出目录
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
