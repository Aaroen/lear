1. 评估指标与交叉验证 (第 2 章)

核心目标: 准确评估模型的泛化性能，避免过拟合或欠拟合的误判，并选择最优模型。

关键概念:

混淆矩阵 (Confusion Matrix): TP, TN, FP, FN

评估指标: 准确率 (Accuracy), 错误率 (Error Rate), 查准率 (Precision), 查全率 (Recall), F1 值, ROC 曲线, AUC 值。

验证策略: K 折交叉验证 (k-fold Cross Validation)。

实现思路与步骤:

混淆矩阵:

输入: 真实标签列表 y_true, 预测标签列表 y_pred。

输出: 2x2 的 NumPy 数组表示混淆矩阵。

方法: 遍历 y_true 和 y_pred，根据 (true, pred) 对累加到矩阵的对应位置 (如 (1,1) 是 TP, (0,0) 是 TN, (0,1) 是 FP, (1,0) 是 FN，假设 1 是正类, 0 是负类)。

基础评估指标 (基于混淆矩阵):

输入: 混淆矩阵。

输出: Accuracy, Precision, Recall, F1。

方法: 根据各自的公式 (Accuracy = (TP+TN)/Total, Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2 * Precision * Recall / (Precision + Recall)) 计算。注意处理分母为零的情况（返回 0 或 NaN，或加小 epsilon）。

K 折交叉验证:

输入: 完整数据集 X, y, 折数 k。

输出: k 次验证的评估指标列表（或平均值）。

方法:

将数据集索引随机打乱。

将打乱后的索引平均分成 k 份。

进行 k 次循环 (i 从 0 到 k-1):

取第 i 份作为验证集 (validation set)。

其余 k-1 份合并作为训练集 (training set)。

在训练集上训练模型。

在验证集上评估模型，记录指标。

计算 k 次指标的平均值和标准差。

注意事项:

确保指标计算与类别定义一致（哪个是正类）。

交叉验证前务必打乱数据，否则可能因数据顺序导致偏差。

分层 K 折交叉验证 (Stratified K-Fold) 在类别不平衡时更优，可以考虑实现（确保每折中类别比例与原始数据相似）。

验证方法: 使用简单已知结果的数据集手动计算验证；使用 sklearn.metrics 中的 confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 和 sklearn.model_selection.KFold 或 StratifiedKFold 对比结果。