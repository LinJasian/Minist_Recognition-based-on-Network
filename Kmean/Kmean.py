import os
import itertools
from typing import Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import ceil

# 设置matplotlib使用无中文字体避免显示问题
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 基本配置
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "database", "HandwrittenNum")

IMG_SIZE = (32, 32)
NUM_CLASSES = 10
K_CLUSTERS = 12

TRAIN_PER_CLASS = 100   # 1~100: 手写训练
VAL_PER_CLASS = 25      # 101~125: 手写验证

MAX_ITERS = 200
TOL = 1e-6
RANDOM_SEED = 42
PCA_DIM = 50


# =========================
# 数据读取与预处理
# =========================
def load_image_as_vector(img_path: str, img_size=(32, 32)) -> np.ndarray:
    """
    读取 bmp 图像 -> 灰度 -> resize -> 反色 -> 展平
    输出范围约为 [0,1]，其中笔画值更大，背景值更小
    """
    img = Image.open(img_path).convert("L")
    img = img.resize(img_size)
    arr = np.array(img, dtype=np.float64) / 255.0

    # 反色：黑色笔画 -> 高值，白背景 -> 低值
    arr = 1.0 - arr
    return arr.flatten()


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    对每个样本向量单独做 L2 归一化
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def load_dataset_fixed_split(dataset_root: str):
    """
    固定划分：
      1~100   -> train
      101~125 -> val
    126.bmp 为印刷体，这里不再单独验证，因此不载入。
    """
    X_train, y_train = [], []
    X_val, y_val = [], []

    for digit in range(NUM_CLASSES):
        digit_dir = os.path.join(dataset_root, str(digit))
        if not os.path.isdir(digit_dir):
            raise FileNotFoundError(f"缺少类别文件夹: {digit_dir}")

        # 训练集
        for idx in range(1, TRAIN_PER_CLASS + 1):
            img_path = os.path.join(digit_dir, f"{idx}.bmp")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"缺少图像文件: {img_path}")
            X_train.append(load_image_as_vector(img_path, IMG_SIZE))
            y_train.append(digit)

        # 验证集
        for idx in range(TRAIN_PER_CLASS + 1, TRAIN_PER_CLASS + VAL_PER_CLASS + 1):
            img_path = os.path.join(digit_dir, f"{idx}.bmp")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"缺少图像文件: {img_path}")
            X_val.append(load_image_as_vector(img_path, IMG_SIZE))
            y_val.append(digit)

    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int64)
    X_val = np.array(X_val, dtype=np.float64)
    y_val = np.array(y_val, dtype=np.int64)

    return X_train, y_train, X_val, y_val


# =========================
# PCA
# =========================
class MyPCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        n_components = min(self.n_components, n_samples, n_features)
        self.mean_ = X.mean(axis=0, keepdims=True)
        X_centered = X - self.mean_

        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        X_pca = np.asarray(X_pca, dtype=np.float64)
        return X_pca @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# =========================
# 距离函数
# =========================
def euclidean_distance_squared(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    X: [N, D]
    centers: [K, D]
    return: [N, K]
    """
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    c2 = np.sum(centers ** 2, axis=1)[None, :]
    xc = X @ centers.T
    dists = x2 + c2 - 2.0 * xc
    return np.maximum(dists, 0.0)


# =========================
# KMeans
# =========================
class MyKMeans:
    def __init__(self, n_clusters=10, max_iters=10, tol=1e-6, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

        self.centers = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centers_kmeans_plus_plus(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples = X.shape[0]
        centers = []

        first_idx = rng.integers(0, n_samples)
        centers.append(X[first_idx].copy())

        for _ in range(1, self.n_clusters):
            current_centers = np.array(centers, dtype=np.float64)
            dists = euclidean_distance_squared(X, current_centers)
            min_dists = np.min(dists, axis=1)
            min_dists = np.maximum(min_dists, 0.0)

            total = np.sum(min_dists)
            if total < 1e-12:
                idx = rng.integers(0, n_samples)
            else:
                probs = min_dists / total
                probs = np.maximum(probs, 0.0)
                probs_sum = probs.sum()

                if probs_sum < 1e-12:
                    idx = rng.integers(0, n_samples)
                else:
                    probs = probs / probs_sum
                    idx = rng.choice(n_samples, p=probs)

            centers.append(X[idx].copy())

        return np.array(centers, dtype=np.float64)

    def _fit_once(self, X: np.ndarray, seed: int):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        rng = np.random.default_rng(seed)

        centers = self._init_centers_kmeans_plus_plus(X, rng)

        for _ in range(self.max_iters):
            dists = euclidean_distance_squared(X, centers)
            labels = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) == 0:
                    rand_idx = rng.integers(0, n_samples)
                    new_centers[k] = X[rand_idx]
                else:
                    new_centers[k] = cluster_points.mean(axis=0)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if shift < self.tol:
                break

        final_dists = euclidean_distance_squared(X, centers)
        final_labels = np.argmin(final_dists, axis=1)
        inertia = float(np.sum(np.min(final_dists, axis=1)))
        return centers, final_labels, inertia

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)

        best_inertia = None
        best_centers = None
        best_labels = None

        for i in range(self.n_init):
            seed = self.random_state + i
            centers, labels, inertia = self._fit_once(X, seed)
            print(f"KMeans 第 {i + 1}/{self.n_init} 次初始化, inertia = {inertia:.6f}")

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.centers = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        print(f"采用最优初始化结果, 最终 inertia = {self.inertia_:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dists = euclidean_distance_squared(X, self.centers)
        return np.argmin(dists, axis=1)


# =========================
# 簇与数字的一一最优匹配
# =========================
def build_cluster_class_count_matrix(
    cluster_ids: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
    n_classes: int,
) -> np.ndarray:
    """
    count_matrix[k, c] = 第 k 个簇中，真实为 c 的样本个数
    """
    count_matrix = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for cid, label in zip(cluster_ids, true_labels):
        count_matrix[int(cid), int(label)] += 1
    return count_matrix


def find_best_one_to_one_mapping(count_matrix: np.ndarray) -> Dict[int, int]:
    """
    贪心算法：找簇到数字的最优一一匹配
    支持 K >= NUM_CLASSES 的情况
    """
    n_clusters, n_classes = count_matrix.shape
    
    if n_clusters == n_classes:
        # 当 K == NUM_CLASSES 时，穷举所有排列找最优
        best_score = -1
        best_perm = None
        for perm in itertools.permutations(range(n_classes)):
            score = sum(count_matrix[cid, perm[cid]] for cid in range(n_clusters))
            if score > best_score:
                best_score = score
                best_perm = perm
        return {cid: int(best_perm[cid]) for cid in range(n_clusters)}
    
    elif n_clusters > n_classes:
        # 当 K > NUM_CLASSES 时，使用贪心算法
        # 每个数字最多匹配一个簇
        mapping = {}
        used_classes = set()
        
        # 贪心：每次找未分配类和未分配簇中，计数最大的配对
        for _ in range(n_classes):
            best_score = -1
            best_cid = -1
            best_class = -1
            
            for cid in range(n_clusters):
                if cid in mapping:
                    continue
                for cls in range(n_classes):
                    if cls in used_classes:
                        continue
                    if count_matrix[cid, cls] > best_score:
                        best_score = count_matrix[cid, cls]
                        best_cid = cid
                        best_class = cls
            
            if best_cid >= 0:
                mapping[best_cid] = best_class
                used_classes.add(best_class)
        
        # 剩余未分配的簇映射到 0
        for cid in range(n_clusters):
            if cid not in mapping:
                mapping[cid] = 0
        
        return mapping
    
    else:
        raise ValueError(f"簇数 ({n_clusters}) 小于类别数 ({n_classes})")


def map_clusters_to_digits(cluster_ids: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(cid)] for cid in cluster_ids], dtype=np.int64)


# =========================
# 评估与输出
# =========================
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes=10) -> np.ndarray:
    """
    生成混淆矩阵
    行：真实类别 (应为 0-9)
    列：预测类别 (应为 0-9，通过 map_clusters_to_digits 已转换)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray):
    print("混淆矩阵（行是真实类别，列是预测类别）:")
    header = "     " + " ".join([f"{i:>4d}" for i in range(cm.shape[1])])
    print(header)
    for i in range(cm.shape[0]):
        row_str = " ".join([f"{cm[i, j]:>4d}" for j in range(cm.shape[1])])
        print(f"{i:>2d} : {row_str}")


def print_cluster_results(count_matrix: np.ndarray, cluster_to_digit: Dict[int, int]):
    """
    输出聚类结果：
    1. 每个簇样本数
    2. 每个簇内各类别分布
    3. 每个簇映射到哪个数字
    """
    print("\n聚类结果统计：")
    for cid in range(count_matrix.shape[0]):
        total = int(np.sum(count_matrix[cid]))
        mapped_digit = cluster_to_digit[cid]
        class_dist = ", ".join(
            [f"{digit}:{int(count_matrix[cid, digit])}" for digit in range(count_matrix.shape[1])]
        )
        print(f"簇 {cid}: 样本数 = {total}, 映射数字 = {mapped_digit}, 类别分布 = [{class_dist}]")


def show_cluster_centers(
    centers_pca: np.ndarray,
    pca: MyPCA,
    img_size=(32, 32),
    cols: int = 5,
    save_path: str = "cluster_centers.png",
):
    """
    将 PCA 空间中的聚类中心还原到原始图像空间，并显示/保存
    """
    centers_raw = pca.inverse_transform(centers_pca)

    centers_img = []
    for i in range(centers_raw.shape[0]):
        vec = centers_raw[i]
        vmin, vmax = vec.min(), vec.max()
        if vmax - vmin < 1e-12:
            img = np.zeros(img_size, dtype=np.float64)
        else:
            vec = (vec - vmin) / (vmax - vmin)
            img = vec.reshape(img_size)
        centers_img.append(img)

    n_centers = len(centers_img)
    rows = ceil(n_centers / cols)

    plt.figure(figsize=(2.2 * cols, 2.2 * rows))
    for i, img in enumerate(centers_img):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Cluster {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n聚类中心图像已保存到: {save_path}")


def show_cluster_samples(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_to_digit: dict,
    img_size=(32, 32),
    samples_per_cluster: int = 10,
    save_path: str = "cluster_samples.png",
):
    """
    为每个簇显示若干代表样本，并标注聚类标签和映射的数字
    """
    n_clusters = len(cluster_to_digit)
    cols = samples_per_cluster
    rows = n_clusters

    plt.figure(figsize=(cols * 0.6, rows * 0.6))

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_ids == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # 从每个簇中随机选择样本
        selected_indices = np.random.choice(
            cluster_indices, 
            size=min(samples_per_cluster, len(cluster_indices)), 
            replace=False
        )

        for col_idx, sample_idx in enumerate(selected_indices):
            subplot_idx = cluster_id * cols + col_idx + 1
            plt.subplot(rows, cols, subplot_idx)
            
            img = X[sample_idx].reshape(img_size)
            plt.imshow(img, cmap="gray")
            
            if col_idx == 0:
                digit = cluster_to_digit[cluster_id]
                plt.ylabel(f"C{cluster_id}→{digit}", fontsize=10)
            
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"聚类样本图像已保存到: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix.png"):
    """
    绘制混淆矩阵的热力图
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(label="Count")
    
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    
    plt.xticks(range(cm.shape[1]))
    plt.yticks(range(cm.shape[0]))
    
    # 添加数值标签
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


# =========================
# 主流程
# =========================
def main():
    print("开始按固定规则加载数据集...")
    print(f"统一图像尺寸: {IMG_SIZE[0]} x {IMG_SIZE[1]}")

    X_train, y_train, X_val, y_val = load_dataset_fixed_split(DATASET_ROOT)

    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")

    # 每个样本单独做 L2 归一化
    X_train = l2_normalize_rows(X_train)
    X_val = l2_normalize_rows(X_val)

    # PCA
    pca_dim = min(PCA_DIM, X_train.shape[0], X_train.shape[1])
    print(f"\n开始 PCA 降维: {X_train.shape[1]} -> {pca_dim}")
    pca = MyPCA(n_components=pca_dim)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    # KMeans
    print(f"\n开始训练 KMeans，k = {K_CLUSTERS} ...")
    kmeans = MyKMeans(
        n_clusters=K_CLUSTERS,
        max_iters=MAX_ITERS,
        tol=TOL,
        random_state=RANDOM_SEED,
        n_init=10,
    )
    kmeans.fit(X_train_pca)

    # 训练集簇-类别计数矩阵
    train_cluster_ids = kmeans.labels_
    count_matrix = build_cluster_class_count_matrix(
        train_cluster_ids, y_train, K_CLUSTERS, NUM_CLASSES
    )

    print("\n训练集上的簇-类别计数矩阵:")
    print(count_matrix)

    # 一一最优匹配
    cluster_to_digit = find_best_one_to_one_mapping(count_matrix)

    print("\n聚类簇 -> 数字标签（一一最优匹配）:")
    for cid in range(K_CLUSTERS):
        print(f"簇 {cid} -> 数字 {cluster_to_digit[cid]}")

    # 输出聚类结果
    print_cluster_results(count_matrix, cluster_to_digit)

    # 展示并保存聚类中心图像
    show_cluster_centers(
        centers_pca=kmeans.centers,
        pca=pca,
        img_size=IMG_SIZE,
        cols=5,
        save_path=os.path.join(PROJECT_ROOT, "cluster_centers.png"),
    )

    # 展示训练集中每个簇的代表样本
    print("\n展示每个簇的代表样本...")
    show_cluster_samples(
        X=X_train,
        cluster_ids=train_cluster_ids,
        cluster_to_digit=cluster_to_digit,
        img_size=IMG_SIZE,
        samples_per_cluster=10,
        save_path=os.path.join(PROJECT_ROOT, "cluster_samples_train.png"),
    )

    # 训练集评估
    y_train_pred = map_clusters_to_digits(train_cluster_ids, cluster_to_digit)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n训练集准确率: {train_acc:.4f}")

    # 验证集评估
    print("\n评估验证集（手写体）...")
    val_cluster_ids = kmeans.predict(X_val_pca)
    y_val_pred = map_clusters_to_digits(val_cluster_ids, cluster_to_digit)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"验证集准确率: {val_acc:.4f}")

    val_cm = confusion_matrix(y_val, y_val_pred, NUM_CLASSES)
    print_confusion_matrix(val_cm)

    # 绘制混淆矩阵热力图
    plot_confusion_matrix(val_cm, save_path=os.path.join(PROJECT_ROOT, "confusion_matrix.png"))

    # 输出性能评估总结
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    print(f"K: {K_CLUSTERS} | Dim: 1024 -> {pca_dim}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Inertia: {kmeans.inertia_:.6f}")
    print(f"Train Acc: {train_acc:.4f} ({int(train_acc * len(X_train))}/{len(X_train)})")
    print(f"Val Acc:   {val_acc:.4f} ({int(val_acc * len(X_val))}/{len(X_val)})")
    print("="*60)

    # 各数字类别准确率
    print("\nPer-digit accuracy (Validation):")
    print("-" * 40)
    for digit in range(NUM_CLASSES):
        digit_mask = y_val == digit
        if np.sum(digit_mask) > 0:
            digit_acc = accuracy_score(y_val[digit_mask], y_val_pred[digit_mask])
            correct = np.sum((y_val == digit) & (y_val_pred == digit))
            total = np.sum(digit_mask)
            print(f"Digit {digit}: {digit_acc:.4f} ({correct}/{total})")
    print("-" * 40)


if __name__ == "__main__":
    main()