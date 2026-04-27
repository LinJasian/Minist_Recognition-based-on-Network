# 手写数字识别 - 完整工程项目

本项目是一个综合的机器学习实验平台，用于比较不同的神经网络模型在MNIST手写数字识别任务上的性能。

## 项目结构

```
BP_Network/
├── dataset/                          # 数据处理模块
│   ├── data_loader.py               # 数据加载器
│   └── __init__.py
│
├── models/                           # 模型实现模块
│   ├── basic_bp.py                  # 基本BP神经网络
│   ├── improved_bp.py               # 改进BP神经网络（交叉熵+正则化）
│   ├── cnn_pytorch.py               # CNN模型（PyTorch）
│   └── __init__.py
│
├── utils/                            # 工具模块
│   ├── config.py                    # 超参数配置
│   ├── metrics.py                   # 评估指标和可视化
│   └── __init__.py
│
├── experiments/                      # 实验脚本
│   ├── exp1_basic_bp.py            # 实验1：基本BP网络
│   ├── exp2_improved_bp.py         # 实验2：改进BP网络对比
│   ├── exp3_cnn.py                  # 实验3：CNN及激活函数对比
│   ├── exp4_comprehensive_comparison.py  # 实验4：综合对比
│   └── __init__.py
│
├── report/                           # 报告输出目录
└── results/                          # 实验结果存储目录
    ├── exp1_results.json
    ├── exp2_results.json
    ├── exp3_results.json
    ├── exp4_results.json
    └── *.png                         # 可视化图表
```

## 核心模块说明

### 1. 数据加载模块 (`dataset/data_loader.py`)

**LocalDataLoader**: 本地BMP格式图片加载器
- 支持从HandwrittenNum目录加载数字0-9的手写图片
- 自动按编号划分训练/验证/测试集

**DataPreprocessor**: 数据预处理工具
- `normalize()`: 数据标准化（MinMax或Z-score）
- `one_hot_encode()`: One-hot编码
- `one_hot_decode()`: One-hot解码

### 2. 模型模块 (`models/`)

#### BasicBPNetwork (`basic_bp.py`)
- **特点**：标准的多层感知机
- **激活函数**：Sigmoid
- **损失函数**：MSE (平方误差)
- **优化器**：梯度下降
- **特殊功能**：
  - Xavier权重初始化
  - 早停机制
  - 完整的前向和反向传播

#### ImprovedBPNetwork (`improved_bp.py`)
- **基类**：继承自BasicBPNetwork
- **改进1**：使用交叉熵损失函数（多分类）
- **改进2**：L2正则化防止过拟合
- **改进3**：学习率衰减调度
- **输出层激活**：Softmax（更适合多分类）

#### SimpleCNN (`cnn_pytorch.py`)
- **框架**：PyTorch
- **架构**：2层卷积 + 3层全连接
- **特征**：
  - 支持ReLU和Sigmoid激活函数对比
  - dropout层防止过拟合
  - 使用Adam优化器
  - 交叉熵损失

### 3. 评估工具模块 (`utils/metrics.py`)

**Metrics**: 性能指标计算
- `accuracy()`: 准确率
- `f1_macro/f1_weighted()`: F1分数
- `precision/recall()`: 精确率和召回率
- `per_class_accuracy()`: 每类准确率
- `confusion_matrix_calc()`: 混淆矩阵

**Visualizer**: 可视化工具
- `plot_training_history()`: 训练曲线
- `plot_confusion_matrix()`: 混淆矩阵热力图
- `plot_per_class_accuracy()`: 每类准确率柱状图
- `plot_sample_images()`: 样本图像展示

**ModelComparator**: 模型对比工具
- 统一的对比框架
- 自动生成对比表格

## 实验流程

### 实验1: 基本BP神经网络 (`exp1_basic_bp.py`)

**目标**：建立基线模型，评估基本BP网络的性能

**工作流程**：
1. 加载本地手写数字数据集
2. 数据标准化和One-hot编码
3. 创建BP网络 [784 → 128 → 64 → 10]
4. 训练和评估
5. 可视化：训练曲线、混淆矩阵、每类准确率

**输出结果**：
- 准确率、F1分数等指标
- 训练曲线图表
- 混淆矩阵

### 实验2: 改进BP网络 (`exp2_improved_bp.py`)

**目标**：对比改进方法的有效性（交叉熵+正则化）

**改进内容**：
1. 将MSE损失替换为交叉熵损失
2. 添加L2正则化项（λ=0.0001）
3. 实现学习率衰减
4. 使用Softmax输出层

**对比维度**：
- 准确率对比
- 损失函数差异
- 过拟合程度
- 收敛速度

**输出结果**：
- 基本模型 vs 改进模型的性能对比
- 提升幅度分析
- 过拟合程度对比

### 实验3: CNN及激活函数对比 (`exp3_cnn.py`)

**目标**：
1. 验证CNN在此任务上的优势
2. 比较ReLU vs Sigmoid激活函数的效果

**CNN架构**：
```
Input (1×32×32)
  ↓
Conv2d(1→32, 5×5) + Activation + MaxPool(2×2)
  ↓
Conv2d(32→64, 5×5) + Activation + MaxPool(2×2)
  ↓
Flatten → FC(1600→128) + Dropout + Activation
  ↓
FC(128→64) + Dropout + Activation
  ↓
FC(64→10)
```

**对比维度**：
- 激活函数（ReLU vs Sigmoid）对收敛速度的影响
- 激活函数对准确率的影响
- 每类准确率分布差异

**输出结果**：
- ReLU和Sigmoid版本的性能对比
- 激活函数效果分析
- 收敛曲线对比

### 实验4: 综合对比 (`exp4_comprehensive_comparison.py`)

**目标**：
对比三种模型在同一数据集上的综合性能，得出最优方案

**对比模型**：
1. BP基础网络
2. BP改进网络
3. CNN-ReLU

**对比维度**：
- 测试准确率
- 加权F1分数
- 训练-验证曲线
- 过拟合程度
- 性能排名

**输出结果**：
- 综合对比表格和图表
- 模型性能排名
- 最优模型推荐

## 运行方式

### 环境要求

```bash
# Python版本
Python >= 3.8

# 必需库
numpy
matplotlib
scikit-learn
scipy
PIL (Pillow)
torch
torchvision
```

### 运行所有实验

```bash
cd /Users/zhanghaozhe/Documents/VScode/Partern\ Recogniton/BP_Network

# 运行所有4个实验
python run_experiments.py

# 或指定运行单个实验
python run_experiments.py --exp 1  # 运行实验1
python run_experiments.py --exp 2  # 运行实验2
python run_experiments.py --exp 3  # 运行实验3
python run_experiments.py --exp 4  # 运行实验4
```

### 单独运行各模块

```bash
# 测试数据加载器
python dataset/data_loader.py

# 测试基本BP网络
python models/basic_bp.py

# 测试改进BP网络
python models/improved_bp.py

# 测试CNN模型
python models/cnn_pytorch.py

# 运行单个实验
python experiments/exp1_basic_bp.py
python experiments/exp2_improved_bp.py
python experiments/exp3_cnn.py
python experiments/exp4_comprehensive_comparison.py
```

## 超参数配置

在 `utils/config.py` 中统一管理所有超参数：

### BP基础网络
- 层大小：[1024, 128, 64, 10]
- 学习率：0.1
- 批大小：32
- 最大轮数：200

### BP改进网络
- 层大小：[1024, 128, 64, 10]
- 学习率：0.1
- L2正则化系数：0.0001
- 学习率衰减：0.99/epoch
- 批大小：32

### CNN
- 学习率：0.001
- 批大小：32
- 最大轮数：50
- 优化器：Adam

## 结果输出

所有实验结果保存在 `results/` 目录：

```
results/
├── exp1_results.json              # 实验1结果
├── exp1_loss_curve.png            # 实验1：损失曲线
├── exp1_confusion_matrix.png      # 实验1：混淆矩阵
├── exp1_per_class_accuracy.png    # 实验1：每类准确率
│
├── exp2_results.json              # 实验2结果
├── exp2_training_comparison.png   # 实验2：对比曲线
│
├── exp3_results.json              # 实验3结果
├── exp3_relu_training_curve.png
├── exp3_sigmoid_training_curve.png
├── exp3_activation_comparison.png
│
├── exp4_results.json              # 实验4结果
├── exp4_comprehensive_comparison.png  # 综合对比图
│
└── all_experiments_summary.json    # 所有实验总结
```

## 关键设计特点

### 1. 模块化设计
- 数据、模型、评估相互独立
- 易于扩展和修改

### 2. 完整的数据流
```
原始数据 → 加载 → 预处理 → 训练 → 评估 → 可视化 → 结果保存
```

### 3. 对比实验框架
- 统一的评估指标
- 自动化的结果对比
- 完整的可视化

### 4. 早停机制
- 防止过度训练
- 自动选择最优模型

### 5. 可复现性
- 固定随机种子
- 详细的超参数记录
- 完整的结果保存

## 预期结果

根据常见的MNIST基准：

| 模型 | 准确率 | 特点 |
|------|--------|------|
| BP基础 | ~94-96% | 收敛较快，可能过拟合 |
| BP改进 | ~95-97% | 正则化减少过拟合 |
| CNN | ~98-99% | 最佳性能，参数效率高 |

## 常见问题

### Q1: 运行时出现内存不足错误
**A**: 减少批大小或数据集大小，或者减少隐层神经元数量

### Q2: CNN训练很慢
**A**: 确认PyTorch是否正确安装，如果GPU不可用会使用CPU自动计算

### Q3: 如何修改超参数
**A**: 编辑 `utils/config.py` 文件中的配置字典

### Q4: 如何添加新的模型
**A**: 
1. 在 `models/` 目录创建新文件
2. 继承或参考现有模型
3. 创建对应的实验脚本

## 参考文献

- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. IJCNN
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. ICLR
- He, K., et al. (2015). Delving Deep into Rectifiers. ICML

## 作者
学生ID: [Your ID]
日期: 2026-04-26

---

**最后更新**: 2026年4月26日
