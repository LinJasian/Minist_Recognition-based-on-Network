"""
测试 Improvement_Guide 中的方案一和方案二
方案一：增加 PCA 维度 (50 -> 75 -> 100 -> 128)
方案二：增加 K 值 (10 -> 12 -> 15)
"""

import os
import sys
import time
import itertools
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix

# 导入主程序的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Kmean.Kmean import (
    load_dataset_fixed_split,
    load_image_as_vector,
    l2_normalize_rows,
    MyPCA,
    MyKMeans,
    build_cluster_class_count_matrix,
    map_clusters_to_digits,
    NUM_CLASSES,
    IMG_SIZE,
    PROJECT_ROOT,
    DATASET_ROOT,
    TRAIN_PER_CLASS,
    VAL_PER_CLASS,
    MAX_ITERS,
    TOL,
    RANDOM_SEED,
)


def find_best_one_to_one_mapping_flexible(count_matrix: np.ndarray) -> Dict[int, int]:
    """
    改进版本：支持 n_clusters > n_classes 或 n_clusters < n_classes
    
    使用匈牙利算法思想（贪心简化版）：
    - 如果 n_clusters == n_classes: 穷举所有排列
    - 如果 n_clusters > n_classes: 每个类最多匹配一个簇，找最优匹配
    - 如果 n_clusters < n_classes: 每个簇可能代表多个类（不建议）
    """
    n_clusters, n_classes = count_matrix.shape
    
    if n_clusters == n_classes:
        # 原始方法：穷举所有排列
        best_score = -1
        best_perm = None
        
        for perm in itertools.permutations(range(n_classes)):
            score = 0
            for cid in range(n_clusters):
                score += count_matrix[cid, perm[cid]]
            if score > best_score:
                best_score = score
                best_perm = perm
        
        return {cid: int(best_perm[cid]) for cid in range(n_clusters)}
    
    elif n_clusters > n_classes:
        # K > 类别数：每个类最多匹配一个簇，其他簇不匹配
        # 使用贪心算法找最优匹配
        mapping = {}
        assigned_classes = set()
        
        for _ in range(n_classes):
            # 找未分配类中，与任何未分配簇计数最大的 (簇, 类) 对
            best_score = -1
            best_cid = -1
            best_clas = -1
            
            for cid in range(n_clusters):
                if cid in mapping:
                    continue
                for clas in range(n_classes):
                    if clas in assigned_classes:
                        continue
                    if count_matrix[cid, clas] > best_score:
                        best_score = count_matrix[cid, clas]
                        best_cid = cid
                        best_clas = clas
            
            if best_cid >= 0:
                mapping[best_cid] = best_clas
                assigned_classes.add(best_clas)
        
        # 补充未分配的簇（分配到任意类，因为不会用）
        for cid in range(n_clusters):
            if cid not in mapping:
                mapping[cid] = 0
        
        return mapping
    
    else:
        # n_clusters < n_classes: 不推荐
        raise ValueError(f"簇数 ({n_clusters}) < 类别数 ({n_classes})，不支持此配置")


def test_configuration(pca_dim: int, k_clusters: int, n_init: int = 10) -> Dict:
    """
    测试单个配置
    返回：{'pca_dim': ..., 'k_clusters': ..., 'train_acc': ..., 'val_acc': ..., 'time': ...}
    """
    print(f"\n{'='*70}")
    print(f"测试配置：PCA_DIM={pca_dim}, K={k_clusters}, n_init={n_init}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # 加载数据
    print("加载数据...")
    X_train, y_train, X_val, y_val = load_dataset_fixed_split(DATASET_ROOT)
    
    # L2 归一化
    X_train = l2_normalize_rows(X_train)
    X_val = l2_normalize_rows(X_val)
    print(f"数据已加载：训练集 {X_train.shape}, 验证集 {X_val.shape}")
    
    # PCA 降维
    print(f"PCA 降维到 {pca_dim} 维...")
    pca = MyPCA(n_components=pca_dim)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    print(f"PCA 完成：{X_train_pca.shape}")
    
    # KMeans 聚类
    print(f"KMeans 聚类 (K={k_clusters})...")
    kmeans = MyKMeans(
        n_clusters=k_clusters,
        max_iters=MAX_ITERS,
        tol=TOL,
        random_state=RANDOM_SEED,
        n_init=n_init
    )
    
    # 隐藏 KMeans 的详细输出
    import io
    from contextlib import redirect_stdout
    
    with redirect_stdout(io.StringIO()):
        kmeans.fit(X_train_pca)
    
    train_cluster_ids = kmeans.labels_
    print(f"KMeans 完成，inertia={kmeans.inertia_:.6f}")
    
    # 寻找最优簇-数字匹配
    print("寻找最优簇-数字匹配...")
    count_matrix = build_cluster_class_count_matrix(
        train_cluster_ids, y_train, k_clusters, NUM_CLASSES
    )
    cluster_to_digit = find_best_one_to_one_mapping_flexible(count_matrix)
    
    # 训练集准确率
    y_train_pred = map_clusters_to_digits(train_cluster_ids, cluster_to_digit)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"训练集准确率：{train_acc:.4f} ({int(train_acc * len(X_train))}/{len(X_train)})")
    
    # 验证集准确率
    val_cluster_ids = kmeans.predict(X_val_pca)
    y_val_pred = map_clusters_to_digits(val_cluster_ids, cluster_to_digit)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"验证集准确率：{val_acc:.4f} ({int(val_acc * len(X_val))}/{len(X_val)})")
    
    elapsed_time = time.time() - start_time
    print(f"耗时：{elapsed_time:.2f} 秒")
    
    return {
        'pca_dim': pca_dim,
        'k_clusters': k_clusters,
        'n_init': n_init,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'inertia': kmeans.inertia_,
        'time': elapsed_time,
    }


def test_plan_1_pca_dimensions():
    """方案一：测试不同的 PCA 维度"""
    print("\n" + "="*70)
    print("方案一：增加 PCA 维度")
    print("="*70)
    
    pca_dims = [50, 75, 100, 128]
    results_plan1 = []
    
    for pca_dim in pca_dims:
        result = test_configuration(pca_dim=pca_dim, k_clusters=10, n_init=10)
        results_plan1.append(result)
    
    # 打印对比表格
    print("\n" + "="*70)
    print("方案一结果对比")
    print("="*70)
    print(f"{'PCA_DIM':<10} {'Train_Acc':<12} {'Val_Acc':<12} {'Inertia':<12} {'Time(s)':<10}")
    print("-" * 70)
    
    for res in results_plan1:
        print(f"{res['pca_dim']:<10} {res['train_acc']:<12.4f} {res['val_acc']:<12.4f} "
              f"{res['inertia']:<12.6f} {res['time']:<10.2f}")
    
    # 计算改进百分比
    baseline_val_acc = results_plan1[0]['val_acc']
    print("\n改进百分比（相对于基线 50维）：")
    print("-" * 70)
    for res in results_plan1:
        improvement = (res['val_acc'] - baseline_val_acc) * 100
        print(f"PCA_DIM={res['pca_dim']}: {improvement:+.2f}% "
              f"(从 {baseline_val_acc:.4f} 到 {res['val_acc']:.4f})")
    
    return results_plan1


def test_plan_2_k_values():
    """方案二：测试不同的 K 值"""
    print("\n" + "="*70)
    print("方案二：增加 K 值")
    print("="*70)
    
    k_values = [10, 12, 15]
    results_plan2 = []
    
    for k in k_values:
        result = test_configuration(pca_dim=50, k_clusters=k, n_init=10)
        results_plan2.append(result)
    
    # 打印对比表格
    print("\n" + "="*70)
    print("方案二结果对比")
    print("="*70)
    print(f"{'K':<10} {'Train_Acc':<12} {'Val_Acc':<12} {'Inertia':<12} {'Time(s)':<10}")
    print("-" * 70)
    
    for res in results_plan2:
        print(f"{res['k_clusters']:<10} {res['train_acc']:<12.4f} {res['val_acc']:<12.4f} "
              f"{res['inertia']:<12.6f} {res['time']:<10.2f}")
    
    # 计算改进百分比
    baseline_val_acc = results_plan2[0]['val_acc']
    print("\n改进百分比（相对于基线 K=10）：")
    print("-" * 70)
    for res in results_plan2:
        improvement = (res['val_acc'] - baseline_val_acc) * 100
        print(f"K={res['k_clusters']}: {improvement:+.2f}% "
              f"(从 {baseline_val_acc:.4f} 到 {res['val_acc']:.4f})")
    
    return results_plan2


def test_combined_config():
    """额外测试：结合方案一和方案二"""
    print("\n" + "="*70)
    print("结合方案：PCA_DIM=100 + K=12")
    print("="*70)
    
    result = test_configuration(pca_dim=100, k_clusters=12, n_init=10)
    
    print("\n" + "="*70)
    print("结合结果")
    print("="*70)
    print(f"PCA_DIM=100, K=12:")
    print(f"  Train_Acc: {result['train_acc']:.4f}")
    print(f"  Val_Acc:   {result['val_acc']:.4f}")
    print(f"  Inertia:   {result['inertia']:.6f}")
    print(f"  Time:      {result['time']:.2f}s")
    
    return result


def print_final_summary(results_plan1, results_plan2, combined_result):
    """打印最终总结"""
    print("\n\n" + "="*70)
    print("最终总结")
    print("="*70)
    
    baseline = results_plan1[0]['val_acc']
    
    print(f"\n基线配置 (K=10, PCA=50):")
    print(f"  验证集准确率: {baseline:.4f}")
    
    best_plan1 = max(results_plan1, key=lambda x: x['val_acc'])
    print(f"\n方案一最优 (PCA_DIM={best_plan1['pca_dim']}):")
    print(f"  验证集准确率: {best_plan1['val_acc']:.4f}")
    print(f"  改进: {(best_plan1['val_acc'] - baseline) * 100:+.2f}%")
    
    best_plan2 = max(results_plan2, key=lambda x: x['val_acc'])
    print(f"\n方案二最优 (K={best_plan2['k_clusters']}):")
    print(f"  验证集准确率: {best_plan2['val_acc']:.4f}")
    print(f"  改进: {(best_plan2['val_acc'] - baseline) * 100:+.2f}%")
    
    print(f"\n结合方案 (PCA=100, K=12):")
    print(f"  验证集准确率: {combined_result['val_acc']:.4f}")
    print(f"  改进: {(combined_result['val_acc'] - baseline) * 100:+.2f}%")
    
    print("\n" + "="*70)


def main():
    """主测试流程"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Improvement Guide 方案验证" + " "*26 + "║")
    print("║" + " "*10 + "测试方案一：增加 PCA 维度 (50->75->100->128)" + " "*14 + "║")
    print("║" + " "*10 + "测试方案二：增加 K 值 (10->12->15)" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    # 执行方案一
    results_plan1 = test_plan_1_pca_dimensions()
    
    # 执行方案二
    results_plan2 = test_plan_2_k_values()
    
    # 执行结合方案
    combined_result = test_combined_config()
    
    # 打印最终总结
    print_final_summary(results_plan1, results_plan2, combined_result)
    
    print("\n✅ 所有测试完成！\n")


if __name__ == "__main__":
    main()
