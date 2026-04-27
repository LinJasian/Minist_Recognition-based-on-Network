import os
from typing import Dict

import numpy as np
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "database", "HandwrittenNum")

IMG_SIZE = (32, 32)
NUM_CLASSES = 10
N_CLUSTERS = 10

TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 25

PCA_DIM = 50
MAX_KMEANS_ITERS = 60
MAX_FCM_ITERS = 40
TOL = 1e-5
FUZZINESS = 2.0
RANDOM_SEED = 42
N_INIT = 3
INIT_CONFIDENCE = 0.90
CROP_THRESHOLD = 245


def load_image_as_vector(img_path: str, img_size=(32, 32)) -> np.ndarray:
    img = Image.open(img_path).convert("L")
    img_array = np.array(img, dtype=np.uint8)
    img_array = crop_foreground(img_array, threshold=CROP_THRESHOLD)
    img_array = pad_to_square(img_array, pad_value=255)

    img = Image.fromarray(img_array)
    try:
        img = img.resize(img_size, Image.Resampling.LANCZOS)
    except AttributeError:
        img = img.resize(img_size, Image.LANCZOS)

    arr = np.array(img, dtype=np.float64) / 255.0
    arr = 1.0 - arr
    return arr.flatten()


def crop_foreground(img_array: np.ndarray, threshold=245) -> np.ndarray:
    mask = img_array < threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img_array

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return img_array[y_min:y_max + 1, x_min:x_max + 1]


def pad_to_square(img_array: np.ndarray, pad_value=255) -> np.ndarray:
    height, width = img_array.shape
    size = max(height, width)
    square = np.full((size, size), pad_value, dtype=img_array.dtype)

    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    square[y_offset:y_offset + height, x_offset:x_offset + width] = img_array
    return square


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def standardize_features(X_train: np.ndarray, X_val: np.ndarray, eps: float = 1e-12):
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (X_train - mean) / std, (X_val - mean) / std


def load_dataset_fixed_split(dataset_root: str):
    X_train, y_train = [], []
    X_val, y_val = [], []

    for digit in range(NUM_CLASSES):
        digit_dir = os.path.join(dataset_root, str(digit))
        if not os.path.isdir(digit_dir):
            raise FileNotFoundError(f"缺少类别文件夹: {digit_dir}")

        for idx in range(1, TRAIN_PER_CLASS + 1):
            img_path = os.path.join(digit_dir, f"{idx}.bmp")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"缺少图像文件: {img_path}")
            X_train.append(load_image_as_vector(img_path, IMG_SIZE))
            y_train.append(digit)

        for idx in range(TRAIN_PER_CLASS + 1, TRAIN_PER_CLASS + VAL_PER_CLASS + 1):
            img_path = os.path.join(digit_dir, f"{idx}.bmp")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"缺少图像文件: {img_path}")
            X_val.append(load_image_as_vector(img_path, IMG_SIZE))
            y_val.append(digit)

    return (
        np.array(X_train, dtype=np.float64),
        np.array(y_train, dtype=np.int64),
        np.array(X_val, dtype=np.float64),
        np.array(y_val, dtype=np.int64),
    )


class MyPCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        n_components = min(self.n_components, n_samples, n_features)

        self.mean_ = np.mean(X, axis=0, keepdims=True)
        X_centered = X - self.mean_
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = vt[:n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def euclidean_distance_squared(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    c2 = np.sum(centers ** 2, axis=1)[None, :]
    xc = X @ centers.T
    distances = x2 + c2 - 2.0 * xc
    return np.maximum(distances, 1e-12)


class MyKMeans:
    def __init__(self, n_clusters=10, max_iters=60, tol=1e-5, random_state=42, n_init=3):
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
            distances = euclidean_distance_squared(X, current_centers)
            min_distances = np.min(distances, axis=1)
            total = np.sum(min_distances)

            if total <= 1e-12:
                next_idx = rng.integers(0, n_samples)
            else:
                probs = min_distances / total
                next_idx = rng.choice(n_samples, p=probs)

            centers.append(X[next_idx].copy())

        return np.array(centers, dtype=np.float64)

    def _fit_once(self, X: np.ndarray, seed: int):
        n_samples = X.shape[0]
        rng = np.random.default_rng(seed)
        centers = self._init_centers_kmeans_plus_plus(X, rng)

        for _ in range(self.max_iters):
            distances = euclidean_distance_squared(X, centers)
            labels = np.argmin(distances, axis=1)

            new_centers = np.zeros_like(centers)
            for cluster_id in range(self.n_clusters):
                mask = (labels == cluster_id)
                if np.any(mask):
                    new_centers[cluster_id] = np.mean(X[mask], axis=0)
                else:
                    new_centers[cluster_id] = X[rng.integers(0, n_samples)]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        distances = euclidean_distance_squared(X, centers)
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum(np.min(distances, axis=1)))
        return centers, labels, inertia

    def fit(self, X: np.ndarray):
        best_inertia = None
        best_centers = None
        best_labels = None

        for init_idx in range(self.n_init):
            centers, labels, inertia = self._fit_once(X, self.random_state + init_idx)
            print(f"KMeans 预热第 {init_idx + 1}/{self.n_init} 次, inertia = {inertia:.6f}")

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.centers = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self


class FuzzyCMeans:
    def __init__(
        self,
        n_clusters=10,
        fuzziness=2.0,
        max_iters=40,
        tol=1e-5,
        random_state=42,
        n_init=3,
        init_confidence=0.90
    ):
        if fuzziness <= 1.0:
            raise ValueError("模糊系数 m 必须大于 1。")

        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.init_confidence = init_confidence

        self.centers = None
        self.membership_ = None
        self.labels_ = None
        self.objective_ = None
        self.kmeans_centers_ = None
        self.kmeans_labels_ = None

    def _membership_from_labels(self, labels: np.ndarray) -> np.ndarray:
        n_samples = labels.shape[0]
        base = (1.0 - self.init_confidence) / max(1, self.n_clusters - 1)
        membership = np.full((n_samples, self.n_clusters), base, dtype=np.float64)
        membership[np.arange(n_samples), labels] = self.init_confidence
        return membership

    def _update_centers(self, X: np.ndarray, membership: np.ndarray) -> np.ndarray:
        um = membership ** self.fuzziness
        denominator = np.sum(um, axis=0, keepdims=True).T
        denominator = np.maximum(denominator, 1e-12)
        return (um.T @ X) / denominator

    def _update_membership(self, distances: np.ndarray) -> np.ndarray:
        inv_power = -1.0 / (self.fuzziness - 1.0)
        weights = distances ** inv_power
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        return weights / np.maximum(weight_sum, 1e-12)

    def _objective_function(self, membership: np.ndarray, distances: np.ndarray) -> float:
        return float(np.sum((membership ** self.fuzziness) * distances))

    def _fit_once(self, X: np.ndarray, seed: int):
        kmeans = MyKMeans(
            n_clusters=self.n_clusters,
            max_iters=MAX_KMEANS_ITERS,
            tol=self.tol,
            random_state=seed,
            n_init=1
        )
        kmeans.fit(X)

        membership = self._membership_from_labels(kmeans.labels_)
        best_membership = membership.copy()
        best_centers = kmeans.centers.copy()
        best_objective = self._objective_function(
            membership,
            euclidean_distance_squared(X, best_centers)
        )

        for iteration in range(self.max_iters):
            centers = self._update_centers(X, membership)
            distances = euclidean_distance_squared(X, centers)
            new_membership = self._update_membership(distances)
            objective = self._objective_function(new_membership, distances)

            membership_shift = np.linalg.norm(new_membership - membership)
            center_spread = np.mean(euclidean_distance_squared(centers, centers))

            if objective < best_objective and center_spread > 1e-3:
                best_objective = objective
                best_membership = new_membership.copy()
                best_centers = centers.copy()

            membership = new_membership

            if membership_shift < self.tol:
                break

        labels = np.argmax(best_membership, axis=1)
        return (
            best_centers,
            best_membership,
            labels,
            best_objective,
            iteration + 1,
            kmeans.centers.copy(),
            kmeans.labels_.copy(),
        )

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)

        best_objective = None
        best_centers = None
        best_membership = None
        best_labels = None
        best_kmeans_centers = None
        best_kmeans_labels = None

        for init_idx in range(self.n_init):
            seed = self.random_state + init_idx
            (
                centers,
                membership,
                labels,
                objective,
                n_iters,
                kmeans_centers,
                kmeans_labels,
            ) = self._fit_once(X, seed)
            print(
                f"FCM 第 {init_idx + 1}/{self.n_init} 次初始化, "
                f"迭代轮数 = {n_iters}, objective = {objective:.6f}"
            )

            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_centers = centers
                best_membership = membership
                best_labels = labels
                best_kmeans_centers = kmeans_centers
                best_kmeans_labels = kmeans_labels

        self.centers = best_centers
        self.membership_ = best_membership
        self.labels_ = best_labels
        self.objective_ = best_objective
        self.kmeans_centers_ = best_kmeans_centers
        self.kmeans_labels_ = best_kmeans_labels
        return self

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        distances = euclidean_distance_squared(np.asarray(X, dtype=np.float64), self.centers)
        return self._update_membership(distances)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_membership(X), axis=1)


def build_cluster_class_probability_matrix(
    membership: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
    n_classes: int,
    fuzziness: float
) -> np.ndarray:
    weighted_membership = membership ** fuzziness
    class_matrix = np.zeros((n_clusters, n_classes), dtype=np.float64)

    for cls in range(n_classes):
        mask = (true_labels == cls)
        if np.any(mask):
            class_matrix[:, cls] = np.sum(weighted_membership[mask], axis=0)

    row_sums = np.sum(class_matrix, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    return class_matrix / row_sums


def build_cluster_class_count_matrix(
    cluster_ids: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
    n_classes: int,
) -> np.ndarray:
    count_matrix = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for cid, label in zip(cluster_ids, true_labels):
        count_matrix[int(cid), int(label)] += 1
    return count_matrix


def find_best_one_to_one_mapping(count_matrix: np.ndarray) -> Dict[int, int]:
    n_clusters, n_classes = count_matrix.shape
    if n_clusters != n_classes:
        raise ValueError("当前实现要求聚类数与类别数相同。")

    total_masks = 1 << n_classes
    dp = np.full(total_masks, -1, dtype=np.int64)
    parent_mask = np.full(total_masks, -1, dtype=np.int64)
    parent_choice = np.full(total_masks, -1, dtype=np.int64)
    dp[0] = 0

    for mask in range(total_masks):
        cluster_id = int(mask.bit_count())
        if cluster_id >= n_clusters or dp[mask] < 0:
            continue

        for cls in range(n_classes):
            if mask & (1 << cls):
                continue

            next_mask = mask | (1 << cls)
            score = dp[mask] + int(count_matrix[cluster_id, cls])
            if score > dp[next_mask]:
                dp[next_mask] = score
                parent_mask[next_mask] = mask
                parent_choice[next_mask] = cls

    mapping = {}
    mask = total_masks - 1
    for cluster_id in range(n_clusters - 1, -1, -1):
        mapping[cluster_id] = int(parent_choice[mask])
        mask = int(parent_mask[mask])

    return mapping


def predict_with_cluster_class_probs(
    membership: np.ndarray,
    cluster_class_probs: np.ndarray
) -> np.ndarray:
    class_scores = membership @ cluster_class_probs
    return np.argmax(class_scores, axis=1).astype(np.int64)


def predict_from_cluster_ids(
    cluster_ids: np.ndarray,
    cluster_to_digit: Dict[int, int]
) -> np.ndarray:
    return np.array([cluster_to_digit[int(cluster_id)] for cluster_id in cluster_ids], dtype=np.int64)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes=10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray):
    print("混淆矩阵（行是真实类别，列是预测类别）:")
    print("     " + " ".join([f"{i:>4d}" for i in range(cm.shape[1])]))
    for i in range(cm.shape[0]):
        row = " ".join([f"{cm[i, j]:>4d}" for j in range(cm.shape[1])])
        print(f"{i:>2d} : {row}")


def print_membership_examples(membership: np.ndarray, y_true: np.ndarray, n_show: int = 10):
    print("\n部分样本的模糊隶属度（前 10 个训练样本）:")
    for idx in range(min(n_show, membership.shape[0])):
        values = ", ".join([f"{v:.3f}" for v in membership[idx]])
        print(f"样本 {idx:>2d}, 真实标签 = {int(y_true[idx])}, 隶属度 = [{values}]")


def main():
    print("开始按固定规则加载数据集...")
    print(f"统一图像尺寸: {IMG_SIZE[0]} x {IMG_SIZE[1]}")

    X_train, y_train, X_val, y_val = load_dataset_fixed_split(DATASET_ROOT)
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")

    X_train = l2_normalize_rows(X_train)
    X_val = l2_normalize_rows(X_val)

    pca_dim = min(PCA_DIM, X_train.shape[0], X_train.shape[1])
    print(f"\n开始 PCA 降维: {X_train.shape[1]} -> {pca_dim}")
    pca = MyPCA(n_components=pca_dim)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_train_pca, X_val_pca = standardize_features(X_train_pca, X_val_pca)

    print(f"\n开始训练模糊 K 均值聚类器，聚类数 = {N_CLUSTERS} ...")
    fcm = FuzzyCMeans(
        n_clusters=N_CLUSTERS,
        fuzziness=FUZZINESS,
        max_iters=MAX_FCM_ITERS,
        tol=TOL,
        random_state=RANDOM_SEED,
        n_init=N_INIT,
        init_confidence=INIT_CONFIDENCE
    )
    fcm.fit(X_train_pca)

    train_cluster_ids = fcm.labels_
    count_matrix = build_cluster_class_count_matrix(
        train_cluster_ids, y_train, N_CLUSTERS, NUM_CLASSES
    )
    cluster_class_probs = build_cluster_class_probability_matrix(
        fcm.membership_, y_train, N_CLUSTERS, NUM_CLASSES, FUZZINESS
    )
    cluster_to_digit_fcm = find_best_one_to_one_mapping(count_matrix)

    kmeans_count_matrix = build_cluster_class_count_matrix(
        fcm.kmeans_labels_, y_train, N_CLUSTERS, NUM_CLASSES
    )
    cluster_to_digit_kmeans = find_best_one_to_one_mapping(kmeans_count_matrix)

    print("\n训练集上的簇-类别计数矩阵:")
    print(count_matrix)
    print("\n训练集上的簇-类别概率矩阵:")
    print(cluster_class_probs)
    print("\n聚类簇 -> 数字标签（一一最优匹配）:")
    for cluster_id in range(N_CLUSTERS):
        print(f"簇 {cluster_id} -> 数字 {cluster_to_digit_fcm[cluster_id]}")

    print_membership_examples(fcm.membership_, y_train, n_show=10)

    y_train_pred_fcm = predict_from_cluster_ids(fcm.labels_, cluster_to_digit_fcm)
    train_acc_fcm = accuracy_score(y_train, y_train_pred_fcm)
    y_train_pred_fcm_soft = predict_with_cluster_class_probs(fcm.membership_, cluster_class_probs)
    train_acc_fcm_soft = accuracy_score(y_train, y_train_pred_fcm_soft)
    y_train_pred_kmeans = predict_from_cluster_ids(fcm.kmeans_labels_, cluster_to_digit_kmeans)
    train_acc_kmeans = accuracy_score(y_train, y_train_pred_kmeans)

    candidates = [
        ("FCM-hard", train_acc_fcm),
        ("FCM-soft", train_acc_fcm_soft),
        ("KMeans", train_acc_kmeans),
    ]
    best_mode, final_train_acc = max(candidates, key=lambda item: item[1])

    print(f"\nFCM 训练集准确率: {train_acc_fcm:.4f}")
    print(f"FCM 软判决训练集准确率: {train_acc_fcm_soft:.4f}")
    print(f"KMeans 预热训练集准确率: {train_acc_kmeans:.4f}")
    print(f"最终采用: {best_mode}")
    print(f"训练集准确率: {final_train_acc:.4f}")

    print("\n评估验证集（手写体）...")
    if best_mode == "FCM-hard":
        val_membership = fcm.predict_membership(X_val_pca)
        y_val_pred = predict_from_cluster_ids(np.argmax(val_membership, axis=1), cluster_to_digit_fcm)
    elif best_mode == "FCM-soft":
        val_membership = fcm.predict_membership(X_val_pca)
        y_val_pred = predict_with_cluster_class_probs(val_membership, cluster_class_probs)
    else:
        val_distances = euclidean_distance_squared(X_val_pca, fcm.kmeans_centers_)
        val_cluster_ids = np.argmin(val_distances, axis=1)
        y_val_pred = predict_from_cluster_ids(val_cluster_ids, cluster_to_digit_kmeans)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"验证集准确率: {val_acc:.4f}")

    val_cm = confusion_matrix(y_val, y_val_pred, NUM_CLASSES)
    print_confusion_matrix(val_cm)


if __name__ == "__main__":
    main()
