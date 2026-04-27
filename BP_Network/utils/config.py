"""
项目超参数配置
"""
import os

# ===== 【核心配置】数据集选择 =====
# 切换为 True 使用标准MNIST数据集（推荐！）
# 切换为 False 使用本地小数据集
USE_MNIST = True  # ⭐ 改为True以使用标准MNIST

# ===== 数据相关配置 =====
if USE_MNIST:
    # 标准MNIST配置
    # - 训练集：48,000张 (原60,000张的80%)
    # - 验证集：12,000张 (原60,000张的20%)
    # - 测试集：10,000张 (官方标准)
    MNIST_DOWNLOAD_DIR = "./data"  # MNIST下载目录
    NUM_CLASSES = 10
    IMG_SIZE = (28, 28)  # MNIST原始尺寸
    
    # MNIST数据集会直接从torchvision加载，不需要以下参数
    DATA_ROOT = None
    TRAIN_PER_CLASS = None
    VAL_PER_CLASS = None
    TEST_PER_CLASS = None
    USE_STRATIFIED_SAMPLING = False
else:
    # 本地数据集配置
    DATA_ROOT = "/Users/zhanghaozhe/Documents/VScode/Partern Recogniton/database/HandwrittenNum"
    IMG_SIZE = (32, 32)
    NUM_CLASSES = 10
    
    # 本地数据集划分 - 优化后的比例 (70% train, 15% val, 15% test)
    TRAIN_PER_CLASS = 88    # 每类训练样本数 (70%)
    VAL_PER_CLASS = 19      # 每类验证样本数 (15%)
    TEST_PER_CLASS = 19     # 每类测试样本数 (15%) - 使用分层抽样
    USE_STRATIFIED_SAMPLING = True  # 启用分层抽样

print(f"⚙️  配置: 使用{'标准MNIST数据集'if USE_MNIST else '本地BMP数据集'}")

# ===== BP网络配置 =====
# 基本BP网络
BP_BASIC = {
    'layer_sizes': [784, 128, 64, 10] if USE_MNIST else [1024, 128, 64, 10],  # 输入层, 隐层1, 隐层2, 输出层
    'learning_rate': 0.1,
    'batch_size': 128,  # 增加到128加快训练 (从32)
    'epochs': 50,  # 减少到50 (从200) - 依赖early stopping
    'early_stopping_patience': 10,  # 早停更激进 (从20)
}

# 改进BP网络
BP_IMPROVED = {
    'layer_sizes': [784, 128, 64, 10] if USE_MNIST else [1024, 128, 64, 10],
    'learning_rate': 0.1,
    'lambda_reg': 0.0001,  # L2正则化系数
    'batch_size': 128,  # 增加到128加快训练 (从32)
    'epochs': 50,  # 减少到50 (从200) - 依赖early stopping
    'early_stopping_patience': 10,  # 早停更激进 (从20)
    'lr_decay': 0.99,  # 学习率衰减
}

# ===== CNN配置 =====
CNN_CONFIG = {
    'batch_size': 256,  # 增加到256以加快batch处理 (从128)
    'learning_rate': 0.01,  # SGD学习率提高到0.01 (从Adam的0.001)
    'epochs': 20,  # 降低到20（CNN收敛快，激进一点）
    'early_stopping_patience': 5,  # 更激进的早停 (从8)
    'eval_frequency': 2,  # 每2个epoch评估一次 (从3)
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
