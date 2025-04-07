import numpy as np
from confusion_matrix import confusion_matrix
from metrics import calculate_metrics
# from sklearn.base import clone # 如果需要克隆模型

def k_fold_cross_validation(model, X, y, k=5, random_state=None):
    """
    执行 K 折交叉验证。

    Args:
        model: 具有 fit(X, y) 和 predict(X) 方法的模型实例。
               注意：如果模型是有状态的，建议在每次循环开始时克隆模型。
        X: 特征数据 (np.array or list of lists)。
        y: 标签数据 (np.array or list)。
        k: 折数 (int)。
        random_state: 随机种子，用于复现打乱过程 (int or None)。

    Returns:
        list: 包含 k 次验证的评估指标字典的列表。
    """
    X = np.array(X)
    y = np.array(y)
    n_samples = len(y)
    indices = np.arange(n_samples)

    if k <= 1 or k > n_samples:
        raise ValueError(f"折数 k={k} 必须大于 1 且小于等于样本数 {n_samples}")

    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices) # 打乱索引

    # 计算每折的大小，处理不能整除的情况
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    all_metrics = []

    print(f"开始 {k}-折交叉验证...")
    for i in range(k):
        start, stop = current, current + fold_sizes[i]
        val_indices = indices[start:stop]
        # 使用 boolean masking 或者 set difference 来获取训练索引更鲁棒
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_indices] = False
        train_indices = indices[train_mask]
        # 或者 train_indices = np.setdiff1d(indices, val_indices, assume_unique=True)

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # print(f"  Fold {i+1}/{k}: Train size={len(y_train)}, Val size={len(y_val)}")

        # --- 模型训练和预测 ---
        # 如果模型是有状态的，在这里克隆: current_model = clone(model)
        current_model = model # 假设模型无状态或 fit 会重置状态
        current_model.fit(X_train, y_train)
        y_pred_val = current_model.predict(X_val)
        # ------

        # --- 计算指标 ---
        cm_val = confusion_matrix(y_val, y_pred_val)
        metrics = calculate_metrics(cm_val)
        all_metrics.append(metrics)
        # print(f"    Metrics: {metrics}")
        # ------

        current = stop
    print(f"{k}-折交叉验证完成.")
    return all_metrics

def calculate_average_metrics(all_metrics):
    """
    计算交叉验证结果的平均指标和标准差。

    Args:
        all_metrics: k_fold_cross_validation 返回的指标列表。

    Returns:
        dict: 包含每个指标平均值 ('_avg') 和标准差 ('_std') 的字典。
    """
    avg_metrics = {}
    if not all_metrics:
        return avg_metrics

    # 假设所有字典有相同的键
    keys = all_metrics[0].keys()
    num_folds = len(all_metrics)

    for key in keys:
        metric_values = [m[key] for m in all_metrics]
        avg_metrics[key + '_avg'] = np.mean(metric_values)
        avg_metrics[key + '_std'] = np.std(metric_values)

    return avg_metrics

# --- 测试代码 ---
if __name__ == '__main__':
    print("\n--- 测试 cross_validation ---")
    # 定义一个简单的 DummyClassifier 供测试
    class DummyClassifierForTest:
        def fit(self, X, y):
            pass
        def predict(self, X):
            # 预测结果依赖于输入特征的第一个维度是否大于等于均值
            # （只是一个任意规则，确保预测不是恒定的）
            if X.ndim == 1: X = X.reshape(1, -1) # 处理单个样本预测
            means = np.mean(X, axis=0)
            # print(f"Dummy predict: means={means}")
            return (X[:, 0] >= 5).astype(int) # 简单规则

    # 准备测试数据
    X_test = np.array([[i, i*2] for i in range(10)]) # 0..9
    y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 前5个 0，后5个 1

    model_test = DummyClassifierForTest()
    k_test = 5
    seed_test = 42

    print(f"\n执行 {k_test}-折交叉验证 (模型: DummyClassifierForTest, random_state={seed_test})...")
    cv_results_test = k_fold_cross_validation(
        model_test, X_test, y_test, k=k_test, random_state=seed_test
    )

    print(f"\n{k_test}-折交叉验证结果 (每折):")
    assert len(cv_results_test) == k_test
    for i, metrics in enumerate(cv_results_test):
        print(f"Fold {i+1}: {metrics}")

    avg_metrics_test = calculate_average_metrics(cv_results_test)
    print("\n平均交叉验证指标:")
    # 打印每个平均值和标准差
    for key, value in avg_metrics_test.items():
        print(f"  {key}: {value:.4f}")

    # 可以添加更具体的 assert 来验证指标的预期范围
    # 例如，基于 Dummy 模型的预测规则和数据分布，估计一下 Accuracy 等
    # Dummy 预测: X[0] >= 5 时为 1，否则为 0
    # y_test: [0,0,0,0,0, 1,1,1,1,1]
    # X_test[0]: [0,1,2,3,4, 5,6,7,8,9]
    # y_pred_test (如果对整个 X_test 预测): [0,0,0,0,0, 1,1,1,1,1] -> 完美预测！
    # 但交叉验证时，训练/验证集不同，结果不会完美
    assert avg_metrics_test['Accuracy_avg'] > 0.5 # 应该比随机好
    assert avg_metrics_test['F1_avg'] > 0.5 # F1 也应该不错

    print("\n交叉验证模块测试基本完成。") 