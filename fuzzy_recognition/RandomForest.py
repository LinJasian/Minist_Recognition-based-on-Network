import os
import numpy as np
from PIL import Image
from collections import Counter


class Node:
    """决策树节点"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx      # 分裂的特征索引
        self.threshold = threshold          # 分裂的阈值
        self.left = left                    # 左子树（满足条件的样本）
        self.right = right                  # 右子树（不满足条件的样本）
        self.value = value                  # 叶节点的类别值


class DecisionTree:
    """决策树分类器"""
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        """构建决策树"""
        self.n_features = X.shape[1] if self.n_features is None else self.n_features
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 停止条件
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # 寻找最佳分裂
        best_gain = -1
        best_feature_idx = None
        best_threshold = None

        # 随机选择特征子集（随机森林特性）
        feature_idxs = np.random.choice(n_features, 
                                       size=min(int(np.sqrt(n_features)), n_features),
                                       replace=False)

        for feature_idx in feature_idxs:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                # 分裂
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # 计算信息增益（使用Gini指数）
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                
                # 父节点的Gini
                parent_gini = self._gini(y)
                
                # 子节点的Gini
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                
                # 加权平均Gini
                weighted_gini = (n_left / n_samples * left_gini + 
                                n_right / n_samples * right_gini)
                
                # 信息增益
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        # 如果没有找到好的分裂，返回叶节点
        if best_feature_idx is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # 递归构建左右子树
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_idx=best_feature_idx, threshold=best_threshold,
                   left=left_subtree, right=right_subtree)

    def _gini(self, y):
        """计算Gini指数"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def predict(self, X):
        """预测"""
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """遍历树进行预测"""
        if node.value is not None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class RandomForest:
    """随机森林分类器"""
    def __init__(self, n_trees=50, max_depth=10, min_samples_split=2, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """训练随机森林"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap抽样
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # 训练决策树
            tree = DecisionTree(max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """预测"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # 多数投票
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] 
                        for i in range(X.shape[0])])


def load_image_as_vector(image_path, target_size=(16, 16), threshold=128, binarize=False):
    """加载图像为向量"""
    img = Image.open(image_path).convert("L")

    try:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    except AttributeError:
        img = img.resize(target_size, Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    if binarize:
        arr = (arr < threshold).astype(np.float32)
    else:
        arr = 1.0 - arr / 255.0

    return arr.flatten()


def load_dataset_fixed_split(root_dir, target_size=(16, 16), threshold=128, binarize=False):
    """加载数据集（固定分割）"""
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_print, y_print = [], []

    for digit in range(10):
        digit_dir = os.path.join(root_dir, str(digit))
        if not os.path.isdir(digit_dir):
            raise FileNotFoundError(f"找不到类别文件夹: {digit_dir}")

        for idx in range(1, 127):
            image_path = os.path.join(digit_dir, f"{idx}.bmp")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"找不到图像文件: {image_path}")

            feature = load_image_as_vector(
                image_path=image_path,
                target_size=target_size,
                threshold=threshold,
                binarize=binarize
            )

            if 1 <= idx <= 100:
                X_train.append(feature)
                y_train.append(digit)
            elif 101 <= idx <= 125:
                X_val.append(feature)
                y_val.append(digit)
            else:
                X_print.append(feature)
                y_print.append(digit)

    return (
        np.array(X_train, dtype=np.float32),
        np.array(y_train, dtype=np.int32),
        np.array(X_val, dtype=np.float32),
        np.array(y_val, dtype=np.int32),
        np.array(X_print, dtype=np.float32),
        np.array(y_print, dtype=np.int32),
    )


def accuracy_score(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes=10):
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def classification_report_simple(y_true, y_pred, num_classes=10):
    """生成分类报告"""
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    lines = []
    lines.append(f"{'类别':<8}{'precision':<12}{'recall':<12}{'f1-score':<12}{'support':<10}")

    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = np.sum(cm[:, cls]) - tp
        fn = np.sum(cm[cls, :]) - tp
        support = np.sum(cm[cls, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        lines.append(
            f"{cls:<8}{precision:<12.4f}{recall:<12.4f}{f1_score:<12.4f}{support:<10d}"
        )

    return "\n".join(lines)


def main():
    """主函数"""
    # 数据路径配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "database", "HandwrittenNum")

    # 参数配置
    target_size = (16, 16)
    threshold = 240
    binarize = False
    n_trees = 50
    max_depth = 15
    min_samples_split = 2
    random_state = 42

    print("="*80)
    print("随机森林手写数字识别")
    print("="*80)

    # 加载数据集
    print("\n[1] 加载数据集...")
    X_train, y_train, X_val, y_val, X_print, y_print = load_dataset_fixed_split(
        root_dir=data_dir,
        target_size=target_size,
        threshold=threshold,
        binarize=binarize
    )

    print(f"    训练集样本数: {len(y_train)}")
    print(f"    验证集样本数: {len(y_val)}")
    print(f"    印刷体测试样本数: {len(y_print)}")
    print(f"    图像尺寸: {target_size[0]} x {target_size[1]}")
    print(f"    特征维度: {X_train.shape[1]}")

    # 训练随机森林
    print("\n[2] 训练随机森林...")
    print(f"    树的数量: {n_trees}")
    print(f"    树的最大深度: {max_depth}")
    print(f"    最小分裂样本数: {min_samples_split}")

    rf_clf = RandomForest(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    rf_clf.fit(X_train, y_train)
    print("    ✓ 模型训练完成")

    # 评估验证集（手写体）
    print("\n[3] 验证集评估（手写体）:")
    y_val_pred = rf_clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"    准确率: {val_acc:.4f}")
    print("\n    详细分类报告:")
    report = classification_report_simple(y_val, y_val_pred, num_classes=10)
    for line in report.split('\n'):
        print(f"    {line}")

    print("\n    混淆矩阵:")
    cm_val = confusion_matrix(y_val, y_val_pred, num_classes=10)
    for i, row in enumerate(cm_val):
        print(f"    类别{i}: {list(row)}")

    # 评估印刷体测试集
    print("\n[4] 印刷体测试集评估:")
    y_print_pred = rf_clf.predict(X_print)
    print_acc = accuracy_score(y_print, y_print_pred)
    print(f"    准确率: {print_acc:.4f}")

    print("\n    预测详情（前20个样本）:")
    for i, (true_label, pred_label) in enumerate(zip(y_print[:20], y_print_pred[:20])):
        status = "✓" if true_label == pred_label else "✗"
        print(f"    [{status}] 样本{i:2d}: 真实类别={true_label}, 预测类别={pred_label}")

    print("\n    混淆矩阵:")
    cm_print = confusion_matrix(y_print, y_print_pred, num_classes=10)
    for i, row in enumerate(cm_print):
        print(f"    类别{i}: {list(row)}")

    # 总体统计
    print("\n[5] 总体统计:")
    print(f"    总体准确率（验证集）: {val_acc:.4%}")
    print(f"    总体准确率（印刷体）: {print_acc:.4%}")
    print(f"    模型参数总数: {n_trees} 棵决策树")
    print("="*80)


if __name__ == "__main__":
    main()
