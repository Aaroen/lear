import numpy as np

def calculate_metrics(cm):
    """
    根据混淆矩阵计算评估指标。

    Args:
        cm: 2x2 的混淆矩阵 [[TN, FP], [FN, TP]] (NumPy array)。

    Returns:
        dict: 包含 Accuracy, Precision, Recall, F1 的字典。
              如果分母为零，对应指标返回 0.0。
    """
    if not isinstance(cm, np.ndarray) or cm.shape != (2, 2):
        raise ValueError("输入必须是 2x2 的 NumPy 数组")

    TN, FP, FN, TP = cm.ravel() # 展平 [TN, FP, FN, TP]
    Total = TN + FP + FN + TP

    # 准确率
    accuracy = (TP + TN) / Total if Total > 0 else 0.0

    # 查准率 (Precision)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # 查全率 (Recall)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 值
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

# --- 可选的测试代码 ---
if __name__ == '__main__':
    print("--- 测试 calculate_metrics ---")
    # 使用 confusion_matrix.py 中的测试用例结果
    cm_test = np.array([[2, 1], [1, 2]]) # TN=2, FP=1, FN=1, TP=2
    metrics = calculate_metrics(cm_test)
    print(f"输入混淆矩阵:\n{cm_test}")
    print(f"计算指标: {metrics}")

    # 预期值
    expected_acc = (2 + 2) / 6
    expected_prec = 2 / (2 + 1)
    expected_rec = 2 / (2 + 1)
    expected_f1 = 2 * expected_prec * expected_rec / (expected_prec + expected_rec)

    assert np.isclose(metrics['Accuracy'], expected_acc), "Accuracy 计算错误!"
    assert np.isclose(metrics['Precision'], expected_prec), "Precision 计算错误!"
    assert np.isclose(metrics['Recall'], expected_rec), "Recall 计算错误!"
    assert np.isclose(metrics['F1'], expected_f1), "F1 计算错误!"
    print("基本测试通过!")

    print("\n--- 测试分母为零情况 ---")
    # Precision 和 Recall 分母都为 0
    cm_zero_pr = np.array([[5, 0], [0, 0]]) # TP=0, FP=0, FN=0
    metrics_zero_pr = calculate_metrics(cm_zero_pr)
    print(f"输入 (TP=0, FP=0, FN=0):\n{cm_zero_pr}\n指标: {metrics_zero_pr}")
    assert metrics_zero_pr['Precision'] == 0.0 and metrics_zero_pr['Recall'] == 0.0 and metrics_zero_pr['F1'] == 0.0

    # Recall 分母为 0
    cm_zero_rec = np.array([[5, 2], [0, 0]]) # TP=0, FN=0
    metrics_zero_r = calculate_metrics(cm_zero_rec)
    print(f"输入 (TP=0, FN=0):\n{cm_zero_rec}\n指标: {metrics_zero_r}")
    assert metrics_zero_r['Recall'] == 0.0 and metrics_zero_r['F1'] == 0.0

    # Precision 分母为 0
    cm_zero_prec = np.array([[5, 0], [3, 0]]) # TP=0, FP=0
    metrics_zero_p = calculate_metrics(cm_zero_prec)
    print(f"输入 (TP=0, FP=0):\n{cm_zero_prec}\n指标: {metrics_zero_p}")
    assert metrics_zero_p['Precision'] == 0.0 and metrics_zero_p['F1'] == 0.0
    print("分母为零测试通过!") 