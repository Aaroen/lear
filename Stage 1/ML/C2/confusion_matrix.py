import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵。

    Args:
        y_true: 真实标签列表 (list or np.array), 假设标签为 0 或 1。
        y_pred: 预测标签列表 (list or np.array), 假设标签为 0 或 1。

    Returns:
        np.array: 2x2 的混淆矩阵 [[TN, FP], [FN, TP]]。
                  行表示真实类别 (0, 1)，列表示预测类别 (0, 1)。
    """
    if len(y_true) != len(y_pred):
        raise ValueError("输入列表长度必须相同")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 验证标签是否有效 (可选，但推荐)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    if not np.all(np.isin(labels, [0, 1])):
         print(f"警告: 标签包含非 0/1 值: {labels}. 仅计算 0 和 1 的部分。")
         # 或者可以抛出错误: raise ValueError("标签必须是 0 或 1")

    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))

    cm = np.array([[TN, FP], [FN, TP]], dtype=int)

    return cm

# --- 可选的测试代码 ---
if __name__ == '__main__':
    # 当直接运行此文件时执行
    print("--- 测试 confusion_matrix ---")
    y_t = [0, 1, 0, 1, 1, 0]
    y_p = [0, 0, 0, 1, 1, 1] # TN=2, FP=1, FN=1, TP=2
    expected_cm = np.array([[2, 1], [1, 2]])
    calculated_cm = confusion_matrix(y_t, y_p)

    print(f"输入 y_true: {y_t}")
    print(f"输入 y_pred: {y_p}")
    print(f"预期混淆矩阵:\n{expected_cm}")
    print(f"计算混淆矩阵:\n{calculated_cm}")
    assert np.array_equal(calculated_cm, expected_cm), "混淆矩阵计算错误!"
    print("测试通过!")

    # 测试包含非 0/1 值的情况
    y_t_extra = [0, 1, 2, 0]
    y_p_extra = [0, 1, 1, 3]
    print("\n--- 测试包含非 0/1 标签 ---")
    cm_extra = confusion_matrix(y_t_extra, y_p_extra)
    print(f"输入 y_true: {y_t_extra}")
    print(f"输入 y_pred: {y_p_extra}")
    # 预期: TN=1 (0,0), FP=0 (0,?), FN=0 (1,?), TP=1 (1,1)
    expected_cm_extra = np.array([[1, 0], [0, 1]])
    print(f"预期混淆矩阵 (仅0/1):\n{expected_cm_extra}")
    print(f"计算混淆矩阵:\n{cm_extra}")
    assert np.array_equal(cm_extra, expected_cm_extra), "非 0/1 标签混淆矩阵计算错误!"
    print("测试通过!") 