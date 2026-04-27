import os

from dataclasses import dataclass



import numpy as np

from PIL import Image





@dataclass

class DecisionStump:

    feature_index: int = 0

    threshold: float = 0.0

    polarity: int = 1

    alpha: float = 0.0



    def predict(self, X):

        feature_values = X[:, self.feature_index]

        if self.polarity == 1:

            return np.where(feature_values >= self.threshold, 1.0, -1.0)

        return np.where(feature_values < self.threshold, 1.0, -1.0)





class AdaBoostBinaryClassifier:

    def __init__(self, n_estimators=30, max_thresholds=10):

        self.n_estimators = n_estimators

        self.max_thresholds = max_thresholds

        self.stumps = []



    def _build_thresholds(self, feature_values):

        unique_values = np.unique(feature_values)

        if unique_values.size <= 1:

            return unique_values



        if unique_values.size <= self.max_thresholds:

            return unique_values



        quantiles = np.linspace(0.1, 0.9, self.max_thresholds)

        thresholds = np.quantile(feature_values, quantiles)

        thresholds = np.unique(thresholds)

        return thresholds



    def _find_best_stump(self, X, y, weights):

        n_samples, n_features = X.shape

        best_stump = DecisionStump()

        best_error = np.inf



        for feature_idx in range(n_features):

            feature_values = X[:, feature_idx]

            thresholds = self._build_thresholds(feature_values)

            if thresholds.size == 0:

                continue



            for threshold in thresholds:

                pred_pos = np.where(feature_values >= threshold, 1.0, -1.0)

                error_pos = np.sum(weights[pred_pos != y])



                if error_pos < best_error:

                    best_error = error_pos

                    best_stump.feature_index = feature_idx

                    best_stump.threshold = float(threshold)

                    best_stump.polarity = 1



                pred_neg = np.where(feature_values < threshold, 1.0, -1.0)

                error_neg = np.sum(weights[pred_neg != y])



                if error_neg < best_error:

                    best_error = error_neg

                    best_stump.feature_index = feature_idx

                    best_stump.threshold = float(threshold)

                    best_stump.polarity = -1



        return best_stump, best_error



    def fit(self, X, y):

        n_samples = X.shape[0]

        weights = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)

        self.stumps = []



        for _ in range(self.n_estimators):

            stump, error = self._find_best_stump(X, y, weights)

            error = np.clip(error, 1e-10, 1.0 - 1e-10)



            if error >= 0.5:

                break



            stump.alpha = 0.5 * np.log((1.0 - error) / error)

            predictions = stump.predict(X)



            weights *= np.exp(-stump.alpha * y * predictions)

            weights_sum = np.sum(weights)

            if weights_sum <= 0:

                break

            weights /= weights_sum



            self.stumps.append(stump)



        return self



    def decision_function(self, X):

        if not self.stumps:

            return np.zeros(X.shape[0], dtype=np.float64)



        scores = np.zeros(X.shape[0], dtype=np.float64)

        for stump in self.stumps:

            scores += stump.alpha * stump.predict(X)

        return scores



    def predict(self, X):

        return np.where(self.decision_function(X) >= 0.0, 1, -1)





class AdaBoostMultiClassifier:

    def __init__(self, n_estimators=30, max_thresholds=10):

        self.n_estimators = n_estimators

        self.max_thresholds = max_thresholds

        self.classes_ = None

        self.classifiers = {}



    def fit(self, X, y):

        self.classes_ = np.unique(y)

        self.classifiers = {}



        for cls in self.classes_:

            y_binary = np.where(y == cls, 1.0, -1.0)

            clf = AdaBoostBinaryClassifier(

                n_estimators=self.n_estimators,

                max_thresholds=self.max_thresholds

            )

            clf.fit(X, y_binary)

            self.classifiers[int(cls)] = clf



        return self



    def decision_function(self, X):

        score_matrix = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)



        for idx, cls in enumerate(self.classes_):

            score_matrix[:, idx] = self.classifiers[int(cls)].decision_function(X)



        return score_matrix



    def predict(self, X):

        score_matrix = self.decision_function(X)

        best_indices = np.argmax(score_matrix, axis=1)

        return self.classes_[best_indices]





def load_image_as_vector(image_path, target_size=(16, 16), threshold=128, binarize=False):

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

    return np.mean(y_true == y_pred)





def confusion_matrix(y_true, y_pred, num_classes=10):

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for true_label, pred_label in zip(y_true, y_pred):

        cm[true_label, pred_label] += 1

    return cm





def classification_report_simple(y_true, y_pred, num_classes=10):

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

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_dir = os.path.join(project_root, "database", "HandwrittenNum")



    target_size = (16, 16)

    threshold = 240

    binarize = False

    n_estimators = 30

    max_thresholds = 10



    print("开始按固定规则加载数据集...")

    X_train, y_train, X_val, y_val, X_print, y_print = load_dataset_fixed_split(

        root_dir=data_dir,

        target_size=target_size,

        threshold=threshold,

        binarize=binarize

    )



    print(f"训练集样本数: {len(y_train)}")

    print(f"验证集样本数: {len(y_val)}")

    print(f"印刷体测试样本数: {len(y_print)}")

    print(f"单张图像尺寸: {target_size[0]} x {target_size[1]}")

    print(f"特征维度: {X_train.shape[1]}")

    print(f"AdaBoost 迭代轮数: {n_estimators}")



    print("\n开始训练 AdaBoost 多分类器...")

    clf = AdaBoostMultiClassifier(

        n_estimators=n_estimators,

        max_thresholds=max_thresholds

    )

    clf.fit(X_train, y_train)



    print("\n评估验证集（手写体）...")

    y_val_pred = clf.predict(X_val)

    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"验证集准确率: {val_acc:.4f}")

    print("验证集分类报告:")

    print(classification_report_simple(y_val, y_val_pred, num_classes=10))

    print("\n验证集混淆矩阵:")

    print(confusion_matrix(y_val, y_val_pred, num_classes=10))



    print("\n评估印刷体测试集...")

    y_print_pred = clf.predict(X_print)

    print_acc = accuracy_score(y_print, y_print_pred)

    print(f"印刷体测试集准确率: {print_acc:.4f}")

    print("印刷体预测结果:")

    for true_label, pred_label in zip(y_print, y_print_pred):

        print(f"真实类别: {true_label}, 预测类别: {pred_label}")



    print("\n印刷体测试集混淆矩阵:")

    print(confusion_matrix(y_print, y_print_pred, num_classes=10))





if __name__ == "__main__":

    main()