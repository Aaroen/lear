# AI 技术栈学习路线 (AI Tech Stack Learning Roadmap)

学习自用，旨在梳理一套从基础到前沿的 AI 实用技术栈。
---

**1. 编程与数据科学基础 (Programming & Data Science Foundations)**

*   **1.1 Python 核心 (Core Python)**
    *   **必备知识 (Prerequisites):**
        *   核心语法: 变量、数据类型、运算符、控制流 (if/else, for/while)。
        *   数据结构: 列表 (List), 字典 (Dictionary), 元组 (Tuple), 集合 (Set) - 创建、访问、常用方法。
        *   函数: 定义、参数、返回值、作用域。
        *   类与对象: 面向对象编程 (OOP) 思想, 类的定义、继承、封装、多态。
        *   模块与包: 导入 (`import`), 标准库常用模块 (如 `os`, `sys`, `json`, `datetime`)。
        *   **补充:** 文件操作 (读写文本/二进制文件)。
    *   **环境管理 (Environment Management):**
        *   虚拟环境: `venv` (标准库), `conda` (Anaconda) - 创建、激活、管理依赖。
        *   包管理: `pip` - 安装、卸载、列出包, `requirements.txt`。

*   **1.2 数据科学核心库 (Core Data Science Libraries)**
    *   **NumPy (Numerical Python):**
        *   核心概念: 多维数组 (`ndarray`) - 高效数值计算的基础。
        *   创建数组: `np.array()`, `np.arange()`, `np.linspace()`, `np.zeros()`, `np.ones()`, `np.random.*`。
        *   数组属性: `.shape` (形状), `.dtype` (数据类型), `.ndim` (维度数), `.size` (元素总数)。
        *   索引与切片:
            *   基础索引: 整数索引, 切片 (`start:stop:step`)。
            *   花式索引: 使用整数数组或列表进行索引。
            *   布尔索引: 使用布尔数组进行条件过滤。
        *   运算:
            *   元素级运算: `+`, `-`, `*`, `/`, `**` 等。
            *   广播机制 (Broadcasting): 不同形状数组间运算的规则。
        *   常用函数:
            *   聚合函数: `np.sum`, `np.mean`, `np.std`, `np.min`, `np.max`, `np.argmin`, `np.argmax` (沿指定轴计算)。
            *   通用函数 (ufuncs): `np.sqrt`, `np.exp`, `np.log`, `np.sin`, `np.cos` 等 (逐元素操作)。
            *   线性代数: `np.dot` (点积), `np.matmul` (矩阵乘法), `np.linalg.inv` (逆矩阵), `np.linalg.solve` (解线性方程组)。
        *   数组操作: `reshape` (改变形状), `flatten` (展平), `concatenate` (连接), `vstack` (垂直堆叠), `hstack` (水平堆叠), `split` (分割)。
        *   **核心思想:** 向量化 (Vectorization) - 利用 NumPy 内置 C 实现的函数替代 Python 循环, 大幅提升性能。
    *   **Pandas (Python Data Analysis Library):**
        *   核心数据结构:
            *   `Series`: 一维带标签数组 (创建, 索引, 常用属性/方法)。
            *   `DataFrame`: 二维带标签表格型数据 (创建, 行/列索引, 多级索引基础)。
        *   数据导入/导出: `pd.read_csv`, `pd.read_excel`, `pd.read_json`, `pd.read_sql` 等; `.to_csv`, `.to_excel`, `.to_json` 等。
        *   数据查看与探索: `.head()`, `.tail()`, `.info()` (概览), `.describe()` (统计摘要), `.shape`, `.columns`, `.index`, `.dtypes`, `.value_counts()` (频率统计), `.unique()`, `.nunique()`。
        *   数据索引与选择:
            *   列选择: `df['col_name']`, `df[['col1', 'col2']]`。
            *   基于标签 (`.loc[]`): `df.loc[row_label, col_label]` (支持切片, 列表, 布尔)。
            *   基于位置 (`.iloc[]`): `df.iloc[row_index, col_index]` (支持切片, 列表, 布尔)。
            *   条件选择 (布尔索引): `df[df['col'] > value]`。
        *   数据清洗 (Data Cleaning): **(补充: 数据分析流程中的关键步骤)**
            *   缺失值处理: `.isnull().sum()` (检查), `.dropna()` (删除), `.fillna()` (填充 - 均值/中位数/众数/指定值)。
            *   重复值处理: `.duplicated().sum()` (检查), `.drop_duplicates()` (删除)。
        *   数据转换 (Data Transformation):
            *   类型转换: `.astype()`。
            *   应用函数: `.apply()` (行/列), `.map()` (Series 元素), `.applymap()` (DataFrame 元素)。
            *   重命名: `.rename()` (列名/索引名)。
            *   替换值: `.replace()`。
        *   合并与连接 (Merging & Joining):
            *   `.merge()`: 类 SQL 连接 (参数 `how`, `on`, `left_on`, `right_on`)。
            *   `.concat()`: 沿轴堆叠 (参数 `axis`)。
            *   `.join()`: 基于索引的连接。
        *   分组与聚合 (Grouping & Aggregation):
            *   `.groupby()`: 创建 GroupBy 对象 (按单/多列分组)。
            *   聚合操作: `.agg()` (应用多种聚合函数), `.sum()`, `.mean()`, `.size()`, `.count()`, `.std()`, `.var()`, `.min()`, `.max()` 等。
            *   应用自定义函数: `.apply()`.
        *   透视表与交叉表: `pd.pivot_table()` (数据透视), `pd.crosstab()` (频率交叉表)。
        *   时间序列 (Time Series):
            *   `datetime` 对象: `pd.to_datetime()`.
            *   `DatetimeIndex`: 创建时间索引。
            *   重采样 (`.resample()`): 时间频率转换 (降采样/升采样)。
            *   滑动窗口 (`.rolling()`): 计算移动统计量。
        *   **性能与新特性:** 理解向量化操作的重要性, 了解 Pandas 2.x Copy-on-Write 机制以避免不必要的复制。
    *   **数据可视化 (Data Visualization):**
        *   **Matplotlib:** Python 基础绘图库。
            *   核心概念: `Figure` (画布), `Axes` (绘图区域/子图)。
            *   常用图表: `plt.plot()` (线图), `plt.scatter()` (散点图), `plt.bar()` (柱状图), `plt.hist()` (直方图)。
            *   定制化: 设置标题 (`plt.title`), 坐标轴标签 (`plt.xlabel`, `plt.ylabel`), 图例 (`plt.legend`), 刻度, 颜色, 线型等。
        *   **Seaborn:** 基于 Matplotlib 的高级统计可视化库。
            *   优点: 简化统计图形绘制, 提供更美观的默认样式和调色板。
            *   常用图表: `sns.heatmap` (热力图), `sns.distplot`/`histplot`/`kdeplot` (分布图), `sns.pairplot` (变量关系对图), `sns.boxplot` (箱线图), `sns.violinplot` (小提琴图), `sns.lmplot` (回归图)。

---

**2. 机器学习基础 (Machine Learning Fundamentals)**

*   **2.1 Scikit-learn 核心理念与工具 (Scikit-learn Core Concepts & Tools)**
    *   **核心 API & 理念:**
        *   Estimator (估计器) 接口: 统一的 `fit()` (训练), `predict()` (预测), `transform()` (数据转换), `fit_predict()`, `fit_transform()` 方法。
        *   数据表示: 通常接受 NumPy arrays, SciPy sparse matrices, Pandas DataFrames。
    *   **数据预处理 (`sklearn.preprocessing`):** **(补充: 模型性能的关键)**
        *   特征缩放 (Feature Scaling):
            *   `StandardScaler`: 标准化 (零均值, 单位方差)。
            *   `MinMaxScaler`: 归一化到 [0, 1] 或指定范围。
            *   `RobustScaler`: 使用中位数和四分位数范围, 对异常值不敏感。
            *   **适用场景:** 基于距离的算法 (如 KNN, SVM), 梯度下降优化的算法 (如线性回归, 神经网络)。
        *   特征编码 (Feature Encoding):
            *   `LabelEncoder`: 用于目标变量 (y) 的编码。
            *   `OneHotEncoder`: 用于名义类别特征 (无序), 避免引入虚假顺序。
            *   `OrdinalEncoder`: 用于有序类别特征。
        *   缺失值填充 (Missing Value Imputation):
            *   `SimpleImputer`: 使用均值 (`mean`), 中位数 (`median`), 众数 (`most_frequent`), 或常量 (`constant`) 填充。
    *   **模型评估 (`sklearn.metrics`):** **(补充: 选择合适的指标至关重要)**
        *   分类 (Classification):
            *   `accuracy_score`: 准确率 (整体预测正确比例)。
            *   `precision_score`: 精确率 (预测为正的样本中, 实际为正的比例)。
            *   `recall_score`: 召回率 (实际为正的样本中, 被预测为正的比例)。
            *   `f1_score`: F1 分数 (精确率和召回率的调和平均)。
            *   `roc_auc_score`: ROC 曲线下面积 (衡量模型区分正负样本的能力)。
            *   `confusion_matrix`: 混淆矩阵 (详细展示 TP, FP, FN, TN)。
            *   `classification_report`: 汇总主要分类指标。
            *   **权衡:** 理解不同指标的侧重点, 尤其在类别不平衡时。
        *   回归 (Regression):
            *   `mean_squared_error` (MSE): 均方误差。
            *   `mean_absolute_error` (MAE): 平均绝对误差。
            *   `r2_score`: R-squared (决定系数, 解释模型拟合优度)。
        *   聚类 (Clustering):
            *   `silhouette_score`: 轮廓系数 (衡量簇内紧密度和簇间分离度)。
    *   **模型选择与调优 (`sklearn.model_selection`):**
        *   数据划分: `train_test_split` (划分训练集和测试集), `stratify` 参数用于保持类别比例。
        *   交叉验证 (Cross-Validation): **(补充: 更可靠的模型性能评估)**
            *   `KFold`: K 折交叉验证。
            *   `StratifiedKFold`: 分层 K 折交叉验证 (保持类别比例)。
            *   `cross_val_score`: 方便计算交叉验证得分。
            *   **必要性:** 避免在单一测试集上过拟合, 得到更鲁棒的性能估计。
        *   超参数搜索 (Hyperparameter Tuning):
            *   `GridSearchCV`: 网格搜索 (尝试所有参数组合)。
            *   `RandomizedSearchCV`: 随机搜索 (在参数空间随机采样)。
            *   `HalvingGridSearchCV`: 逐步减半网格搜索 (更高效)。
            *   **原理:** 自动化寻找最佳超参数组合。
    *   **管道 (`sklearn.pipeline`):**
        *   `Pipeline`: 将多个步骤 (如预处理, 模型训练) 串联起来。
        *   **优点:** 简化代码流程, 防止在交叉验证中发生数据泄露 (将预处理步骤包含在内), 便于与 GridSearchCV 结合进行整体调优。

*   **2.2 监督学习算法 (Supervised Learning Algorithms - Scikit-learn)**
    *   **线性模型 (`sklearn.linear_model`):**
        *   `LinearRegression`: 普通最小二乘法 (OLS)。
        *   `Ridge`: 岭回归 (L2 正则化), `alpha` 控制正则化强度, 缓解多重共线性。
        *   `Lasso`: 套索回归 (L1 正则化), `alpha` 控制正则化强度, 可进行特征选择 (使部分系数为零)。
        *   `LogisticRegression`: 逻辑回归 (用于分类)。
            *   原理: Sigmoid 函数映射输出到 (0, 1), 交叉熵损失函数。
            *   参数: `C` (正则化强度的倒数), `penalty` ('l1', 'l2'), `solver` (优化算法), `multi_class` (多分类策略 'ovr', 'multinomial')。
    *   **支持向量机 (SVM) (`sklearn.svm`):**
        *   `SVC` (分类), `SVR` (回归):
            *   原理: 寻找最大间隔超平面, 依赖支持向量。
            *   核函数 (`kernel`):
                *   `'linear'`: 线性核。
                *   `'poly'`: 多项式核。
                *   `'rbf'`: 径向基函数核 (高斯核) - 常用。
                *   `'sigmoid'`: Sigmoid 核。
            *   关键超参数: `C` (惩罚系数, 控制对误分类的容忍度), `gamma` (RBF 核宽度, 控制影响范围)。
    *   **决策树 (`sklearn.tree`):**
        *   `DecisionTreeClassifier`, `DecisionTreeRegressor`:
            *   原理: 树状结构, 通过特征进行递归分裂。
            *   分裂标准: `criterion` ('gini', 'entropy' for classification; 'mse', 'mae' for regression)。
            *   剪枝参数 (防止过拟合): `max_depth` (最大深度), `min_samples_split` (节点分裂所需最小样本数), `min_samples_leaf` (叶节点最小样本数)。
    *   **集成学习 (`sklearn.ensemble`):** **(补充: 通常能获得更好的性能)**
        *   **Bagging (Bootstrap Aggregating):**
            *   `RandomForestClassifier`, `RandomForestRegressor`:
                *   原理: 构建多个决策树, 每棵树基于自助采样 (Bootstrap) 的样本和随机选择的特征子集进行训练, 最后聚合结果 (投票/平均)。
                *   参数: `n_estimators` (树的数量), `max_features` (每棵树考虑的特征数量)。
        *   **Boosting:**
            *   `AdaBoostClassifier`:
                *   原理: 串行学习, 逐步提升被前一轮弱学习器错误分类的样本权重。
            *   `GradientBoostingClassifier`/`Regressor`:
                *   原理: 串行学习, 每棵树拟合前面所有树预测结果的残差 (梯度)。
            *   `HistGradientBoostingClassifier`/`Regressor`:
                *   优化: 基于直方图的快速梯度提升实现, 对缺失值有原生支持。

*   **2.3 无监督学习算法 (Unsupervised Learning Algorithms - Scikit-learn)**
    *   **聚类 (`sklearn.cluster`):**
        *   `KMeans`:
            *   算法流程: 初始化 K 个中心点 -> 分配样本到最近中心点 -> 更新中心点 -> 迭代直至收敛。
            *   关键参数: `n_clusters` (簇的数量)。
            *   `n_clusters` 选择方法: 肘部法则 (Elbow Method), 轮廓系数 (Silhouette Score)。
            *   注意: 对初始点敏感, 可能陷入局部最优。
    *   **降维 (`sklearn.decomposition`):**
        *   `PCA` (Principal Component Analysis):
            *   原理: 寻找数据方差最大的方向 (主成分), 将数据投影到低维空间。
            *   应用: 数据压缩, 可视化, 去除噪声。
            *   参数: `n_components` (保留的主成分数量)。
            *   选择 `n_components`: 基于累计解释方差比率 (explained variance ratio)。

*   **2.4 高级梯度提升 (Advanced Gradient Boosting):** **(补充: 竞赛和工业界常用)**
    *   **核心原理:** GBDT (梯度提升决策树) 的加法模型和前向分步思想。
    *   **XGBoost:**
        *   优化点:
            *   正则化项: 损失函数中加入 L1/L2 正则化, 防止过拟合。
            *   二阶泰勒展开: 对损失函数进行二阶展开, 更精确地拟合。
            *   内置缺失值处理: 自动学习缺失值分裂方向。
            *   并行化: 特征粒度的并行 (块结构)。
            *   近似分位数算法: 处理大规模数据。
        *   API: 原生 API (`xgb.train`, `xgb.DMatrix`), Scikit-learn API (`XGBClassifier`, `XGBRegressor`)。
        *   关键参数: `eta` (学习率), `max_depth`, `subsample` (行采样), `colsample_bytree` (列采样), `lambda` (L2), `alpha` (L1), `gamma` (分裂最小损失下降), `objective` (目标函数), `eval_metric` (评估指标)。
    *   **LightGBM:**
        *   优化点:
            *   GOSS (Gradient-based One-Side Sampling): 保留梯度大的样本, 随机采样梯度小的样本。
            *   EFB (Exclusive Feature Bundling): 将互斥特征捆绑, 减少特征维度。
            *   基于直方图的算法: 加速分裂点查找。
            *   Leaf-wise 生长策略: 按叶子节点分裂, 而非按层 (可能导致过拟合, 但通常更高效)。
        *   API: 原生 API (`lgb.train`, `lgb.Dataset`), Scikit-learn API (`LGBMClassifier`, `LGBMRegressor`)。
        *   关键参数: `learning_rate`, `num_leaves` (叶子节点数), `max_depth`, `bagging_fraction` (行采样), `feature_fraction` (列采样), `reg_lambda` (L2), `reg_alpha` (L1), `objective`, `metric`。
    *   **CatBoost:**
        *   优化点:
            *   类别特征处理: 自动高效处理类别特征 (Ordered Boosting, Target-based statistics), 无需手动编码。
            *   对称树 (Oblivious Trees): 简化模型, 加速预测。
        *   API: Scikit-learn API (`CatBoostClassifier`, `CatBoostRegressor`)。
        *   关键参数: 类似 LightGBM, 关注 `cat_features` 参数指定类别特征。
    *   **通用实践:**
        *   特征重要性: `.feature_importances_` 属性获取特征贡献度。
        *   早停法 (Early Stopping): `early_stopping_rounds` 参数, 在验证集性能不再提升时停止训练, 防止过拟合。

---

**3. 深度学习 (Deep Learning)**

*   **3.1 深度学习框架基础 (Deep Learning Framework Basics)**
    *   **PyTorch:** **(补充: 动态图, Pythonic, 学术界常用)**
        *   Tensor 核心:
            *   创建与类型: `torch.tensor`, `torch.randn`, `torch.zeros`, `torch.ones`, `dtype` 参数。
            *   数学运算 & 形状操作: 类似 NumPy (`+`, `-`, `*`, `/`, `matmul`, `reshape`, `squeeze`, `unsqueeze` 等)。
            *   与 NumPy 转换: `.numpy()` (共享内存), `torch.from_numpy()`。
        *   自动求导 (`torch.autograd`):
            *   梯度追踪: `requires_grad=True` 标记需要计算梯度的 Tensor。
            *   梯度计算: `loss.backward()` 计算损失相对于参数的梯度。
            *   梯度访问: `.grad` 属性存储梯度。
            *   关闭梯度计算: `with torch.no_grad():` 或 `model.eval()` (评估模式)。
            *   动态计算图: 计算图在运行时构建。
        *   神经网络模块 (`torch.nn`):
            *   基类: `nn.Module` - 所有神经网络层的父类。
            *   子类化: `__init__` 中定义层, `forward` 方法中定义前向传播逻辑。
        *   优化器 (`torch.optim`):
            *   常用优化器: `SGD` (随机梯度下降), `Adam`, `AdamW` (带权重衰减的 Adam)。
            *   优化流程: `optimizer.zero_grad()` (清空梯度) -> `loss.backward()` (计算梯度) -> `optimizer.step()` (更新参数)。
            *   学习率调度 (`lr_scheduler`): 动态调整学习率 (如 `StepLR`, `ReduceLROnPlateau`)。
        *   数据加载 (`torch.utils.data`):
            *   `Dataset`: 自定义数据集类, 实现 `__len__` 和 `__getitem__`。
            *   `DataLoader`: 批量加载数据, `batch_size`, `shuffle` (打乱), `num_workers` (多进程加载), `collate_fn` (自定义批处理方式)。
        *   训练与评估:
            *   标准训练循环: 迭代 DataLoader -> 数据送入设备 -> 前向传播 -> 计算损失 -> 反向传播 -> 更新参数。
            *   评估模式: `model.eval()` - 关闭 Dropout 和 BatchNorm 的更新。
        *   GPU 使用:
            *   设备指定: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
            *   数据/模型迁移: `.to(device)`。
        *   模型保存与加载:
            *   `state_dict`: 推荐方式, 只保存模型参数。
            *   保存: `torch.save(model.state_dict(), PATH)`。
            *   加载: `model.load_state_dict(torch.load(PATH))`, `model.eval()`。
    *   **(可选) TensorFlow/Keras:** **(补充: 静态图/函数图, 工业界部署广泛)**
        *   Keras API: 高层 API, 易于快速构建模型 (`Sequential`, Functional API)。
        *   TensorFlow 底层: 提供更灵活的控制。
        *   生态: TensorFlow Extended (TFX) 用于生产部署。
    *   **(可选) JAX:** **(补充: 函数式, 高性能, TPU 优化)**
        *   函数式范式: 强调纯函数 (Pure functions)。
        *   核心变换: `grad` (求导), `jit` (即时编译), `vmap` (自动向量化), `pmap` (并行化)。
        *   NumPy-like API: 易于 NumPy 用户上手。
        *   设备无关性: CPU/GPU/TPU 透明切换。
        *   生态: Flax, Haiku (基于 JAX 的神经网络库)。

*   **3.2 神经网络构建块 (Neural Network Building Blocks - PyTorch `torch.nn`)**
    *   常用层:
        *   `nn.Linear`: 全连接层 (`in_features`, `out_features`)。
        *   `nn.Conv1d/2d/3d`: 卷积层 (`in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`)。
        *   `nn.MaxPool1d/2d/3d`, `nn.AvgPool1d/2d/3d`: 池化层 (`kernel_size`, `stride`)。
        *   循环层 (`nn.RNN`, `nn.LSTM`, `nn.GRU`):
            *   参数: `input_size`, `hidden_size`, `num_layers`, `batch_first=True` (输入/输出形状)。
            *   理解: 输入输出形状, 隐藏状态 (hidden state), 细胞状态 (cell state - LSTM)。
        *   `nn.Embedding`: 嵌入层 (将离散 ID 映射为向量) (`num_embeddings`, `embedding_dim`)。
        *   归一化层:
            *   `nn.BatchNorm1d/2d/3d`: 批归一化 (加速收敛, 缓解梯度消失) (`momentum`, `eps`)。
            *   `nn.LayerNorm`: 层归一化 (常用于 Transformer)。
        *   `nn.Dropout`: Dropout 层 (随机失活神经元, 防止过拟合) (`p` 参数)。
        *   Transformer 相关: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`, `nn.Transformer`, `nn.MultiheadAttention`。
    *   激活函数 (Activation Functions): **(补充: 引入非线性)**
        *   `nn.ReLU`, `nn.LeakyReLU`, `nn.PReLU`: 修正线性单元及其变体。
        *   `nn.Sigmoid`: Sigmoid 函数 (输出 0~1, 用于二分类或门控)。
        *   `nn.Tanh`: 双曲正切 (输出 -1~1)。
        *   `nn.Softmax`: Softmax 函数 (输出概率分布, 用于多分类)。
        *   `nn.GELU`: 高斯误差线性单元 (Transformer 常用)。
        *   函数式接口: `torch.nn.functional` (如 `F.relu`, `F.softmax`)。
    *   损失函数 (Loss Functions): **(补充: 衡量预测与真实值的差距)**
        *   `nn.MSELoss`: 均方误差损失 (回归)。
        *   `nn.L1Loss`: 平均绝对误差损失 (回归)。
        *   `nn.CrossEntropyLoss`: 交叉熵损失 (多分类, 内部包含 LogSoftmax)。
        *   `nn.BCELoss`: 二元交叉熵损失 (二分类, 需要输入 Sigmoid 输出)。
        *   `nn.BCEWithLogitsLoss`: 带 Sigmoid 的二元交叉熵损失 (更稳定)。
    *   容器 (Containers):
        *   `nn.Sequential`: 按顺序执行的模块容器。
        *   `nn.ModuleList`: 模块列表 (类似 Python list, 但正确注册模块)。
        *   `nn.ModuleDict`: 模块字典 (类似 Python dict, 但正确注册模块)。

*   **3.3 经典深度学习模型 (Classic Deep Learning Models)**
    *   **卷积神经网络 (CNN - Convolutional Neural Networks):** **(补充: 擅长处理网格结构数据, 如图像)**
        *   关键组件:
            *   卷积层 (`Conv2d`): 局部连接 (Local Connectivity), 参数共享 (Parameter Sharing), 提取局部特征。
            *   池化层 (`MaxPool2d`, `AvgPool2d`): 降低特征图维度 (降维), 提高计算效率, 增强平移不变性。
        *   经典架构演进:
            *   LeNet: 早期 CNN 雏形。
            *   AlexNet: 引入 ReLU 激活函数, Dropout, GPU 加速。
            *   VGGNet: 使用小的卷积核 (3x3) 堆叠, 构建更深的网络。
            *   GoogLeNet (Inception): Inception 模块并行使用不同大小卷积核, 增加网络宽度和适应性。
            *   ResNet (Residual Network): 引入残差连接 (Residual Connection), 解决深度网络训练中的梯度消失和退化问题, 使训练极深网络成为可能。
    *   **循环神经网络 (RNN - Recurrent Neural Networks):** **(补充: 擅长处理序列数据, 如文本, 时间序列)**
        *   基本单元: 隐藏状态在时间步之间循环传递, 捕捉序列信息。
        *   问题: 长期依赖问题 (Long-term Dependency Problem) - 梯度消失或梯度爆炸。
        *   改进变体:
            *   LSTM (Long Short-Term Memory): 引入门控机制 (遗忘门, 输入门, 输出门) 和细胞状态 (`cell state`), 有效缓解长期依赖问题。
            *   GRU (Gated Recurrent Unit): 简化的门控机制 (更新门, 重置门), 参数更少, 效果与 LSTM 相当。
        *   双向 RNN (BiRNN): 同时考虑过去和未来的上下文信息。

*   **3.4 Transformer 架构 (Transformer Architecture):** **(补充: 当前 NLP 和许多其他领域的主流架构)**
    *   **核心组件:**
        *   自注意力机制 (Self-Attention):
            *   核心思想: 计算序列内各元素之间的相互依赖关系。
            *   Q, K, V (Query, Key, Value): 输入向量通过线性变换得到。
            *   Scaled Dot-Product Attention: `softmax(QK^T / sqrt(d_k))V` (计算注意力权重并加权求和 V)。
            *   Masking: Padding Mask (忽略填充位), Look-ahead Mask (Decoder 中防止看到未来信息)。
        *   多头自注意力 (Multi-Head Attention):
            *   原理: 将 Q, K, V 拆分为多组 (头), 并行计算自注意力, 最后拼接并线性变换输出。
            *   目的: 允许模型在不同表示子空间中学习不同的依赖关系。
        *   位置编码 (Positional Encoding):
            *   必要性: Transformer 本身不包含序列顺序信息。
            *   方法: Sinusoidal (三角函数), Learned (可学习参数), RoPE (旋转位置编码), AliBi (注意力偏置) 等。
        *   Add & Norm (残差连接与层归一化):
            *   Add (Residual Connection): 将子层输入直接加到子层输出, 缓解梯度消失, 加速训练。
            *   Norm (Layer Normalization): 对每个样本的特征进行归一化, 稳定训练。
            *   应用位置: 通常在每个子层 (Multi-Head Attention, FFN) 之后。
        *   前馈网络 (Feed-Forward Network - FFN):
            *   结构: 通常是两个线性层加一个激活函数 (如 ReLU 或 GELU)。
            *   作用: 对每个位置的表示进行非线性变换。
    *   **整体架构:**
        *   Encoder: 由 N 个相同的 Encoder Layer 堆叠而成 (每个 Layer 包含 Multi-Head Self-Attention 和 FFN)。
        *   Decoder: 由 N 个相同的 Decoder Layer 堆叠而成 (每个 Layer 包含 Masked Multi-Head Self-Attention, Multi-Head Cross-Attention (关注 Encoder 输出), 和 FFN)。
        *   架构类型:
            *   Encoder-Only: 如 BERT, RoBERTa (适用于 NLU 任务, 如分类, 抽取)。
            *   Decoder-Only: 如 GPT 系列, Llama, Mistral (适用于 NLG 任务, 如文本生成)。
            *   Encoder-Decoder: 如 T5, BART (适用于 Seq2Seq 任务, 如翻译, 摘要)。
    *   **注意力变体与优化 (概念):** FlashAttention (IO 优化), 滑动窗口注意力 (处理长序列), 稀疏注意力 (减少计算量)。

*   **3.5 PyTorch 进阶 (Advanced PyTorch)**
    *   分布式训练 (`torch.distributed`): **(补充: 训练大模型或加速训练)**
        *   基本概念: `world_size` (总进程数), `rank` (当前进程编号), `backend` (通信后端, 如 `nccl` for GPU)。
        *   常用策略: `DistributedDataParallel` (DDP) - 数据并行, 每个 GPU 复制模型, 处理不同数据批次, 同步梯度。
        *   启动方式: `torchrun` (推荐) 或 `python -m torch.distributed.launch`。
    *   PyTorch 2.x 编译优化:
        *   `torch.compile()`: 通过 TorchDynamo (捕获 Python 字节码), TorchInductor (生成优化 C++/Triton 代码) 等后端加速模型训练和推理。

---

**4. 大语言模型 / 基础模型 (LLMs / Foundation Models)**

*   **4.1 LLM 基础与生态 (LLM Basics & Ecosystem)**
    *   **Hugging Face 生态:** **(补充: De facto 标准)**
        *   `transformers`:
            *   `pipeline`: 高度封装的接口, 用于快速原型验证常见任务 (文本生成, 分类, 问答等)。
            *   `AutoClass`: `AutoModel`, `AutoTokenizer`, `AutoConfig` - 根据模型名称自动加载对应的类。
            *   模型加载与 Hub 交互: `from_pretrained()` (从 Hub 或本地加载), `push_to_hub()` (上传模型/tokenizer)。
            *   推理 (Inference):
                *   `model(**inputs)`: 获取模型原始输出 (如 hidden states, logits)。
                *   `model.generate()`: 用于文本生成任务, 提供多种解码策略 (Greedy, Beam Search, Sampling) 和参数 (`max_length`, `num_beams`, `do_sample`, `temperature`, `top_k`, `top_p`)。
        *   `datasets`:
            *   加载: `load_dataset` (从 Hub 或本地加载)。
            *   核心结构: `Dataset`, `DatasetDict`。
            *   处理: `.map` (应用函数), `.filter`, `.shuffle`, `.select`。
            *   底层: 基于 Apache Arrow 格式, 支持内存映射和流式处理 (处理大数据集)。
        *   `tokenizers`:
            *   原理: 子词切分算法 (BPE, WordPiece, SentencePiece) - 解决 OOV 问题, 压缩词表。
            *   使用: `tokenizer(text)` (文本 -> token IDs, attention mask 等), `tokenizer.encode`, `tokenizer.batch_encode_plus`。
            *   特殊 Tokens: `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`, `bos_token`, `eos_token`。
            *   解码: `tokenizer.decode()`, `tokenizer.batch_decode()` (token IDs -> 文本)。
    *   **Prompt Engineering 基础:** **(补充: 与 LLM 高效交互的关键)**
        *   基本原则: 清晰 (Clear), 具体 (Specific), 提供上下文 (Contextual), 设定角色 (Role)。
        *   常用技巧:
            *   Zero-shot Prompting: 直接提问, 不给示例。
            *   Few-shot Prompting: 提供少量示例 (demonstrations)。
            *   Chain-of-Thought (CoT): 引导模型逐步思考, 输出推理过程。
            *   Role Prompting: 指定模型扮演的角色 (如 "你是一个资深程序员...")。
        *   迭代与优化: 不断尝试和改进 Prompt 以获得更好结果。

*   **4.2 主流 LLM 交互 (Interacting with Mainstream LLMs)**
    *   **商业 LLM APIs (OpenAI, Anthropic, Google, etc.):**
        *   通用考量:
            *   API Key 管理: 安全存储和使用。
            *   模型选择: 根据任务需求、性能、成本权衡 (如 GPT-4o vs GPT-3.5-Turbo, Claude 3 Opus vs Sonnet vs Haiku)。
            *   速率限制 (Rate Limits): 理解并处理 API 调用频率限制。
        *   核心请求参数:
            *   `model`: 指定模型 ID。
            *   `messages`: 对话历史列表, 包含 `role` (`system`, `user`, `assistant`) 和 `content`。
            *   `prompt`: (某些 API 可能使用) 单轮输入的提示。
            *   `max_tokens`: 控制生成内容的最大长度。
            *   采样参数: `temperature` (控制随机性), `top_p` (核采样)。
            *   `stream=True`: 流式输出, 逐步返回结果。
        *   响应处理: 解析返回的 JSON 数据, 处理流式输出的事件。
        *   多轮对话: 维护 `messages` 列表, 将历史对话传入。
        *   Function Calling / Tool Use (概念与流程):
            *   定义: 向 API 提供工具 (函数) 的 Schema (名称, 描述, 参数)。
            *   请求: LLM 可能决定调用哪个工具并返回所需参数。
            *   执行: 应用端执行工具。
            *   返回: 将工具执行结果返回给 LLM 继续生成。
        *   关注点: 最新模型能力 (如 GPT-4o, Claude 3 系列, Gemini 1.5 Pro), 多模态支持 (图像/音频输入), 长上下文窗口, 安全与对齐特性。
    *   **开源模型 (Llama 3, Mistral, Qwen, etc.):**
        *   了解模型:
            *   模型家族: 不同规模 (如 7B, 13B, 70B), 版本 (如 Llama 2 vs Llama 3)。
            *   训练数据: 了解大致来源和特点。
            *   **许可证 (License):** **极其重要!** 确认使用限制 (商业/研究)。
        *   本地部署基础:
            *   硬件评估: **VRAM (显存)** 是主要瓶颈。
            *   环境搭建: 安装所需库 (`transformers`, `accelerate`, `bitsandbytes` 等)。
            *   模型下载: 从 Hugging Face Hub (`huggingface-cli download` 或 `git lfs`)。
            *   加载与运行: 使用 `transformers` 库加载模型和 Tokenizer, 或使用专门的推理引擎 (见优化章节)。
        *   关注点: Llama 3 的性能和开放性, Mixtral 的 MoE 架构效率, Qwen 的多语言/多模态/长上下文能力, 各模型的微调生态和社区支持。
        *   **(补充) 选择考量:** API vs 开源 (易用性/成本 vs 灵活性/隐私)。

*   **4.3 高级 LLM 架构与概念 (Advanced LLM Architectures & Concepts)**
    *   **混合专家模型 (MoE - Mixture of Experts):**
        *   原理:
            *   稀疏激活 (Sparse Activation): 每个输入 token 只激活少数几个专家。
            *   路由器 (Router / Gating Network): 学习将 token 分配给最合适的 Top-k 个专家。
            *   专家网络 (Experts): 通常是 FFN (前馈网络)。
        *   优/缺点:
            *   优点: 大幅增加模型参数量, 但计算量 (FLOPs) 仅随 k 线性增长。
            *   缺点: 训练不稳定, 推理显存占用大 (需要加载所有专家), 通信开销。
        *   范例: Mixtral 8x7B, Grok-1。
    *   **多模态模型 (Multimodal Models):**
        *   输入处理:
            *   图像编码器: 通常使用 ViT (Vision Transformer) 或类似结构提取图像特征。
            *   音频编码器: 如 Whisper Encoder。
            *   融合: 将不同模态的 Embedding 投影到共同空间, 与文本 Embedding 结合。
        *   架构: LLaVA, CogVLM, Fuyu-8B, GPT-4V/Omni, Gemini。
        *   应用: 图像描述生成, 视觉问答 (VQA), 图文生成, 语音交互。

*   **4.4 LLM 微调与适配 (LLM Fine-tuning & Adaptation)**
    *   **全量微调 (Full Fine-tuning):** 更新模型所有参数, 效果好但成本高。
    *   **Hugging Face `Trainer`:** **(补充: 简化训练流程)**
        *   `TrainingArguments`: 配置训练超参数 (学习率, batch size, epoch 数, 输出目录, 日志记录等)。
        *   数据处理: 准备 `Dataset` 对象, 定义 `data_collator` (批处理), `compute_metrics` 函数 (评估指标计算)。
        *   训练流程: 初始化 `Trainer` -> 调用 `.train()` 启动训练, `.evaluate()` 进行评估, `.predict()` 进行预测。
    *   **参数高效微调 (PEFT - Parameter-Efficient Fine-Tuning):** **(补充: 降低微调成本的关键)**
        *   动机: 减少全量微调所需的计算资源 (GPU, 时间) 和存储空间。
        *   基本思想: 冻结预训练模型的大部分参数, 只训练少量额外添加或修改的参数。
        *   Hugging Face `peft` 库: 提供常见 PEFT 方法的实现。
        *   常见方法概览: LoRA, QLoRA, Adapter Tuning, Prefix Tuning, Prompt Tuning (细节见优化章节)。
    *   **Hugging Face `accelerate`:** **(补充: 无缝支持单机/多卡/混合精度训练)**
        *   核心: `Accelerator` 对象 - 自动处理设备分配 (CPU/GPU/TPU), 梯度同步 (分布式训练), 混合精度训练 (`fp16`/`bf16`)。
        *   核心用法: `accelerator.prepare(model, optimizer, dataloader)` 包装对象, `accelerator.backward(loss)` 替代 `loss.backward()`。
        *   启动: `accelerate config` (配置环境), `accelerate launch your_script.py` (启动脚本)。

---

**5. 模型效率与优化 (Efficiency & Optimization)**

*   **5.1 参数高效微调 (Parameter-Efficient Fine-Tuning - PEFT):**
    *   **LoRA (Low-Rank Adaptation):**
        *   原理: 在原模型权重矩阵 `W` 旁边增加一个低秩更新矩阵 `ΔW ≈ BA`, 其中 `A` 和 `B` 是低秩矩阵 (`r << d`), 只训练 `A` 和 `B`。
        *   实现 (`peft` 库):
            *   `LoraConfig`: 配置 LoRA 参数。
            *   `target_modules`: 指定要应用 LoRA 的层 (通常是 Attention 层的 `query`, `key`, `value`, 有时也包括 `dense` 层)。
            *   `get_peft_model(model, config)`: 将 LoRA 应用到模型。
        *   关键参数: `r` (秩, 控制新增参数量), `lora_alpha` (缩放因子, 类似学习率), `lora_dropout`, `task_type` (任务类型)。
        *   适配器管理: `save_pretrained` (只保存适配器参数), `from_pretrained` (加载适配器)。
    *   **QLoRA (Quantized LoRA):**
        *   原理: 将预训练模型量化到低精度 (如 4-bit NormalFloat - NF4), 然后在其上应用 LoRA 进行微调。
        *   关键技术:
            *   `bitsandbytes` 库: 提供 4-bit 量化 (NF4, FP4) 和反量化。
            *   Paged Optimizers: 优化器状态分页管理, 减少显存峰值。
            *   Double Quantization: 对量化常数本身再进行量化。
        *   效果: 显著降低微调所需的 VRAM, 使在消费级 GPU 上微调大模型成为可能。
    *   **其他方法 (概念):**
        *   Adapter Tuning: 在 Transformer 层之间插入小型 "Adapter" 模块进行训练。
        *   Prefix Tuning / Prompt Tuning: 冻结模型, 只训练添加到输入端的软提示 (soft prompt) embedding。
        *   IA3 (Infused Adapter by Inhibiting and Amplifying Activations): 通过学习到的向量缩放内部激活。
    *   **最新研究:** DoRA (Weight-Decomposed Low-Rank Adaptation) 等持续改进。

*   **5.2 模型量化 (Quantization):**
    *   目标: 减小模型文件大小, 加速推理速度, 降低内存/显存占用和能耗。
    *   精度类型: FP32 (32位浮点) -> FP16/BF16 (16位浮点) -> INT8 (8位整数) -> 更低位 (如 4-bit)。
    *   方法:
        *   PTQ (Post-Training Quantization): 训练后量化, 使用少量校准数据确定量化参数, 简单快速但可能有精度损失。
        *   QAT (Quantization-Aware Training): 量化感知训练, 在训练过程中模拟量化操作, 通常精度损失更小但需要重新训练。
    *   工具/格式:
        *   `bitsandbytes`: 提供 NF4/FP4 量化 (主要用于 QLoRA)。
        *   **GGUF (GPT-Generated Unified Format):** `llama.cpp` 使用的格式, 专为 CPU/Mac/边缘设备优化, 支持多种量化策略 (如 Q4_0, Q5_K_M, IQ2_XS), 使得在普通硬件上运行 LLM 成为可能。
        *   `AutoGPTQ`: 流行的 PTQ 库, 支持 GPTQ 算法。
        *   `AWQ` (Activation-aware Weight Quantization): 另一种 PTQ 方法, 保护显著权重。
        *   ONNX Runtime: 支持多种硬件后端和量化模型的部署。
        *   **(补充) HQQ (Half-Quadratic Quantization):** 新的量化方法, 速度快且精度损失小。
    *   **权衡:** 精度损失 vs. 效率提升。需要根据具体应用场景选择合适的量化方法和精度。

*   **5.3 高效注意力 (Efficient Attention):**
    *   **FlashAttention (v1/v2):**
        *   原理:
            *   IO 感知 (IO-Aware): 减少 GPU 高带宽内存 (HBM) 和片上 SRAM 之间的读写次数。
            *   Tiling (分块计算): 将注意力计算分解为小块在 SRAM 中进行。
            *   Recomputation (重计算): 重新计算部分中间结果而非存储它们, 节省显存。
        *   效果: 大幅加速 Attention 计算并减少显存占用, 对训练和推理都有效。
        *   集成: 已被主流库 (PyTorch 2.x, `transformers`, `xformers`, `vLLM` 等) 集成, 用户通常无需手动配置即可受益。

*   **5.4 推理优化引擎 (Inference Optimization Engines):**
    *   目标: 提高 LLM 推理的吞吐量 (Throughput), 降低延迟 (Latency), 节省显存 (Memory)。
    *   **核心技术:**
        *   **PagedAttention:** (vLLM 提出) 通过虚拟内存分页管理 KV Cache (存储 Attention 的 Key 和 Value), 解决内存碎片问题, 实现近乎零浪费的 KV Cache 共享。
        *   **Continuous Batching:** 请求动态地加入和离开处理批次, 无需等待整个批次完成, 大幅提高 GPU 利用率。
        *   算子融合 (Operator Fusion): 将多个计算操作合并为单个 GPU Kernel, 减少 Kernel 启动开销和内存读写。
        *   张量并行 (Tensor Parallelism): 将模型权重和计算分割到多个 GPU 上。
        *   **(补充) KV Cache 优化:** 量化 KV Cache, 使用 MQA/GQA (Multi-Query/Grouped-Query Attention) 减少 Key/Value 头的数量。
    *   **主流引擎:**
        *   `vLLM`: 易用性好, 提供 Python API (`LLM` 类) 和 OpenAI 兼容的 API 服务器。广泛应用于学术界和工业界。
        *   `TensorRT-LLM`: NVIDIA 官方库, 针对 NVIDIA GPU 进行深度优化, 需要模型编译步骤, 性能潜力巨大。
        *   `DeepSpeed Inference`: 微软推出, 结合 ZeRO-Inference 和模型并行技术。
        *   **(补充) TGI (Text Generation Inference):** Hugging Face 开发的推理服务器。
        *   **(补充) llama.cpp:** 专注于 CPU/Mac/边缘设备的高效推理。
    *   关注点: 对 MoE 模型、长上下文、量化模型、特定硬件的优化支持。

*   **5.5 投机解码 (Speculative Decoding):**
    *   原理: 使用一个小型、快速的 "草稿" 模型 (Draft Model) 快速生成多个候选 token, 然后使用原始的大型 "验证" 模型 (Target Model) 并行地一次性验证这些 token。如果验证通过, 则接受多个 token, 否则回退。
    *   目标: 在基本不损失生成质量的前提下, 显著降低生成延迟 (Wall Clock Time), 特别适用于交互式应用。
    *   实现: 需要大小两个模型配合, 已在一些推理框架 (如 vLLM, TGI, Transformers) 中得到支持。

---

**6. AI Agent 与应用 (Agents & Applications)**

*   **6.1 核心框架 (Core Frameworks):** **(补充: 用于构建基于 LLM 的应用程序)**
    *   **LangChain:** **(补充: 功能全面, 生态庞大, 快速迭代)**
        *   LCEL (LangChain Expression Language): 使用 `|` 操作符链式组合组件 (Prompt, Model, Parser 等), 提升代码可读性和可组合性。
        *   核心组件:
            *   `PromptTemplate`/`ChatPromptTemplate`: 管理和格式化 Prompt。
            *   `LLM`/`ChatModel`: 封装 LLM API 或本地模型。
            *   `OutputParser`: 解析 LLM 输出 (如 `StrOutputParser`, `JsonOutputParser`)。
            *   `RunnablePassthrough`: 在链中传递输入。
        *   Chains: 封装特定逻辑流程 (如 `SequentialChain`, `RetrievalQA`)。
        *   Agents:
            *   核心逻辑: ReAct (Reason + Act), LLM 决定使用哪个工具以及如何使用。
            *   组件: `Tool` (定义工具), `AgentExecutor` (执行 Agent 循环), Agent 构造器 (如 `create_openai_tools_agent`, `create_react_agent`)。
        *   Memory: 管理对话历史 (如 `ConversationBufferMemory`, `ConversationSummaryMemory`)。
    *   **LlamaIndex:** **(补充: 侧重于将私有数据连接到 LLM - RAG)**
        *   核心数据管道:
            *   `Reader`: 加载数据 (如 `SimpleDirectoryReader`)。
            *   `NodeParser`: 将文档分割成块 (Node) (如 `SentenceSplitter`)。
            *   `Index`: 构建数据索引 (如 `VectorStoreIndex`, `SummaryIndex`, `KeywordTableIndex`)。
            *   `Retriever`: 从索引中检索相关 Node。
            *   `QueryEngine`/`ChatEngine`: 结合检索到的上下文进行查询或对话。
        *   检索策略: `similarity_top_k`, `node_postprocessors` (如 Re-ranking, Filtering)。
        *   查询/聊天引擎: 可定制 Prompt, 支持流式输出。
    *   **AutoGen:** **(补充: 侧重于多 Agent 对话与协作)**
        *   Agent 定义:
            *   `ConversableAgent`: 可对话 Agent 的基类。
            *   `AssistantAgent`: 由 LLM 驱动的 Agent。
            *   `UserProxyAgent`: 代表人类用户, 可以执行代码或接收人类输入。
        *   交互模式: `initiate_chat` 发起对话, 多个 Agent 可以根据预设规则自动进行多轮对话和协作。
    *   **CrewAI:** **(补充: 侧重于结构化的 Agent 协作流程)**
        *   核心概念:
            *   `Agent`: 定义角色 (role), 目标 (goal), 背景故事 (backstory), 可用工具 (tools)。
            *   `Task`: 定义任务描述 (description), 分配给哪个 Agent, 预期输出 (expected_output)。
            *   `Crew`: 包含一组 Agents 和 Tasks, 定义协作流程 (`process` - sequential/hierarchical)。
            *   `Process`: 控制任务执行顺序。

*   **6.2 向量数据库 (Vector Databases):** **(补充: RAG 和语义搜索的核心)**
    *   **核心概念:**
        *   Embedding: 将文本、图像等数据转换为高维向量表示, 捕捉语义信息。
        *   ANN (Approximate Nearest Neighbor) 搜索: 在高维空间中快速查找与查询向量最相似的向量, 而非精确但慢速的 KNN。
    *   **关键技术/算法 (概念):**
        *   索引算法: HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), PQ (Product Quantization), Scalar Quantization - 用于加速 ANN 搜索。
        *   距离度量: Cosine Similarity (余弦相似度 - 常用), Euclidean Distance (L2 - 欧氏距离), Dot Product (IP - 内积)。
    *   **数据库操作 (以 ChromaDB/Milvus/Pinecone/Qdrant/Weaviate 为例):**
        *   客户端初始化/连接。
        *   创建 Collection/Index: 指定名称, embedding function 或向量维度, 距离度量。
        *   添加/更新数据 (`upsert`): 插入向量 (`vectors`), 关联的元数据 (`metadatas`), 唯一 ID (`ids`)。
        *   查询 (`query`/`search`): 输入查询向量 (`query_embeddings`), 返回数量 (`top_k`/`n_results`), 可选的元数据过滤 (`where`/`filter`), 指定返回内容 (`include`)。
        *   删除数据: 根据 ID 删除。
    *   **选择考量:**
        *   开源 vs. 商业/云服务: 部署运维 vs. 便捷性/成本。
        *   性能: 索引构建速度, 查询延迟, QPS。
        *   扩展性: 水平扩展能力。
        *   功能: 元数据过滤能力, 混合搜索 (Hybrid Search), 备份恢复, 多租户。
        *   社区与生态: 文档, 社区支持, 与 LangChain/LlamaIndex 等框架的集成度。

*   **6.3 检索增强生成 (Retrieval-Augmented Generation - RAG):** **(补充: 让 LLM 基于外部知识回答问题)**
    *   **基础 RAG 流程:**
        *   **Indexing (离线索引):**
            1.  文档加载 (Load): 从各种来源加载文档。
            2.  分块 (Chunking/Splitting): 将长文档切分为较小的、语义完整的块。
            3.  向量化 (Embedding): 使用 Embedding 模型将每个块转换为向量。
            4.  存储 (Store): 将向量及其元数据存入向量数据库。
        *   **Retrieval & Generation (在线检索与生成):**
            1.  用户查询向量化: 使用相同的 Embedding 模型转换用户查询。
            2.  检索 (Retrieve): 在向量数据库中搜索与查询向量最相似的 Top-K 个块。
            3.  生成 (Generate): 将用户原始查询和检索到的上下文块组合成一个 Prompt, 输入给 LLM 生成最终答案。
    *   **高级 RAG 技术 (提升效果的关键):**
        *   **查询转换 (Query Transformation):**
            *   Rewriting: 改写用户查询以提高检索相关性。
            *   Sub-Query: 将复杂查询分解为多个子查询分别检索。
            *   HyDE (Hypothetical Document Embeddings): 先让 LLM 生成一个假设性答案, 再用该答案的 Embedding 进行检索。
        *   **分块与索引 (Chunking & Indexing):**
            *   策略: Chunk Size (块大小) 与 Overlap (重叠大小) 的选择。
            *   Sentence Window Retrieval: 检索单个句子, 但返回包含该句子的更大窗口作为上下文。
            *   Hierarchical Indexing / Parent Document Retriever: 索引摘要信息, 检索时先匹配摘要, 再返回对应的原始大块。
        *   **检索优化 (Retrieval Optimization):**
            *   Hybrid Search: 结合稀疏检索 (如 BM25, 关注关键词匹配) 和稠密向量检索 (关注语义相似度)。
            *   Re-ranking: 使用更强大的 (通常是 Cross-Encoder) 模型对初步检索到的 Top-N 结果进行重新排序, 提高最终 Top-K 结果的相关性。
            *   Fusion: 融合来自不同检索策略 (如向量搜索, 关键词搜索) 的结果 (如 RRF - Reciprocal Rank Fusion)。
        *   **生成优化 (Generation Optimization):**
            *   Prompt Engineering: 精心设计 Prompt, 指导 LLM 如何利用检索到的上下文。
            *   处理长上下文: 上下文压缩, 选择性使用最相关的上下文片段。

*   **6.4 Tool Use / Function Calling:** **(补充: 赋予 LLM 执行外部操作的能力)**
    *   **LLM API 侧 (如 OpenAI):**
        *   定义 `tools`: 提供工具列表, 每个工具包含名称 (`name`), 描述 (`description`), 参数 (`parameters` - JSON Schema 格式)。
        *   设置 `tool_choice`: 控制 LLM 是否/必须/选择性地使用工具 ("auto", "required", {"type": "function", "function": {"name": "my_function"}})。
    *   **应用侧 (调用方):**
        1.  解析模型响应: 检查是否有 `tool_calls`。
        2.  识别工具与参数: 获取 `tool_call_id`, 函数名 (`function.name`), 参数 (`function.arguments` - JSON 字符串)。
        3.  执行工具: 查找本地对应的函数或调用外部 API, 传入解析后的参数。
        4.  返回结果: 构造 `role: tool` 的消息, 包含对应的 `tool_call_id` 和工具执行结果 (`content`), 发送回 LLM 继续生成。
    *   **可靠性:**
        *   错误处理: 处理模型未正确调用工具、参数格式错误、函数执行失败等情况。
        *   重试机制: 在函数执行失败时考虑重试。
        *   用户确认: 对于有风险的操作, 可能需要用户确认。
    *   **框架支持:** LangChain, LlamaIndex 等框架提供了对 Tool Use/Function Calling 的抽象和封装, 简化开发。

*   **6.5 多 Agent 系统 (Multi-Agent Systems):**
    *   概念: 多个具有不同角色、能力、目标的 AI Agent 通过通信和协作共同完成单个 Agent 难以完成的复杂任务。
    *   常见模式:
        *   分层协作: 管理者 Agent 分配任务给执行者 Agent。
        *   对话式协作: Agent 之间通过模拟辩论、评审、头脑风暴等方式进行交互。
        *   并行执行: 多个 Agent 并行处理任务的不同部分。
    *   框架使用:
        *   `AutoGen`: 提供了灵活的对话模式配置, 支持多种 Agent 类型和交互流程。
        *   `CrewAI`: 强调结构化的协作流程定义 (Agents, Tasks, Crew, Process)。
    *   挑战:
        *   规划与协调: 如何有效地规划任务分解和 Agent 间的协作。
        *   通信效率: Agent 间信息传递的效率和准确性。
        *   状态同步: 维护和同步共享状态。
        *   错误处理与容错: 单个 Agent 失败对整体任务的影响。
        *   成本控制: 多个 Agent 可能导致更高的 LLM 调用成本。
        *   Agent 间冲突解决。

---

**7. 支撑技术与平台 (Supporting Technologies & Platforms)**

*   **7.1 MLOps / LLMOps:** **(补充: 实现 AI 应用开发、部署和运维的规范化与自动化)**
    *   **基础 MLOps:**
        *   版本控制 (`Git`): 代码、配置、文档的版本管理, 分支策略 (如 Gitflow), Pull Requests (代码审查), Merge。
        *   实验跟踪 (`W&B` - Weights & Biases, `MLflow`):
            *   功能: 记录实验配置 (`log_parameters`), 指标 (`log_metrics`), 数据/模型产出物 (`log_artifacts`, `log_model`), 环境信息。
            *   目的: 保证实验可复现性, 便于比较不同实验结果, 协作共享。
        *   容器化 (`Docker`):
            *   `Dockerfile`: 定义镜像构建步骤。
            *   优点: 环境隔离 (解决 "在我机器上能跑" 的问题), 打包应用及其所有依赖, 保证开发/测试/生产环境一致性。
            *   命令: `docker build`, `docker run`, `docker push/pull`。
        *   CI/CD (持续集成/持续交付/持续部署) 概念:
            *   CI: 频繁集成代码变更到主干, 自动运行测试。
            *   CD: 自动化构建、测试、部署流程。
    *   **LLMOps 特有挑战与实践:**
        *   **Prompt 管理与工程化:** Prompt 版本控制, Prompt 模板库, Prompt A/B 测试, Prompt 性能监控与回溯。
        *   **LLM 评估:**
            *   自动化指标: BLEU, ROUGE (用于文本生成, 但有局限性)。
            *   基于模型的评估: 使用强 LLM (如 GPT-4) 对目标 LLM 的输出进行打分。
            *   人工评估: **黄金标准**, 但成本高。
            *   RAG 评估: Context Relevance (上下文相关性), Faithfulness (答案忠实于上下文), Answer Relevance (答案切题性) - 工具如 Ragas, TruLens, ARES。
            *   安全性/偏见评估。
        *   **Fine-tuning 工作流:** 数据集管理与版本化 (如 DVC), 超参数搜索记录, PEFT 适配器管理与部署。
        *   **RAG 数据管道:** 索引更新策略 (实时/批量), 检索质量监控, Embedding 模型版本管理。
        *   **成本与延迟监控:** Token 使用量跟踪, API 调用延迟分析, 推理成本优化策略。
        *   **安全与对齐:**
            *   Guardrails: 输入/输出内容过滤, 敏感信息检测与脱敏。
            *   幻觉检测与缓解。
            *   内容审核 API 集成。

*   **7.2 云平台 (Cloud Platforms - AWS/GCP/Azure):** **(补充: 提供弹性计算、存储和托管 AI 服务)**
    *   **计算 (Compute):**
        *   GPU 实例: 选择合适的实例类型 (关注 **VRAM 容量**, 型号如 NVIDIA A100/H100/L4/T4, 成本)。
        *   实例管理: 启动、连接 (SSH, Cloud Shell), 停止/终止。
    *   **存储 (Storage):**
        *   对象存储: AWS S3, Google Cloud Storage (GCS), Azure Blob Storage - 用于存储大规模非结构化数据 (数据集, 模型文件, 日志)。
    *   **AI/ML 平台 (AWS SageMaker / Google Vertex AI / Azure Machine Learning):**
        *   Notebook 环境: 提供托管的 JupyterLab/Jupyter Notebook 实例。
        *   **托管基础模型 API:**
            *   AWS Bedrock: 提供对多种基础模型 (Anthropic Claude, Meta Llama, Mistral AI, Cohere 等) 的 API 访问。
            *   Vertex AI Model Garden: 提供 Google 自研 (Gemini) 和第三方模型的访问与部署。
            *   Azure OpenAI Service: 提供对 OpenAI 模型 (GPT-4, GPT-3.5) 的托管访问。
        *   **托管训练/微调:** 提供界面或 SDK 配置数据集、计算资源、超参数, 提交和监控训练/微调任务。
        *   **托管 RAG/Agent 服务:** 提供构建 RAG 应用或 Agent 的集成化服务 (如 Vertex AI Search/Conversation, Azure AI Search)。
        *   模型部署: 创建 API Endpoint (实时推理), Serverless 推理, 批处理推理。
    *   **身份与访问管理 (IAM - Identity and Access Management):** 管理用户、角色、权限, 控制对云资源的访问, 保证安全。
    *   **成本管理:** 理解各服务定价模型 (按需, 预留实例, Spot 实例), 使用成本监控、分析和预算工具, 利用 Spot 实例等降低计算成本。

*   **7.3 硬件 (Hardware):** **(补充: AI 计算的基石)**
    *   **NVIDIA GPU:** **(补充: 当前 AI 训练和推理的主流选择)**
        *   关键指标:
            *   **VRAM 容量 (GB):** **决定能运行/微调多大模型的最关键因素。**
            *   计算能力 (TFLOPS/TOPS): 不同精度 (FP32, TF32, FP16, BF16, FP8, INT8) 下的理论峰值性能。
            *   内存带宽 (GB/s): GPU 内存与计算单元之间的数据传输速度, 对访存密集型任务 (如 Attention) 很重要。
            *   NVLink/NVSwitch: 多 GPU 间的高速互联技术, 对大规模分布式训练至关重要。
        *   主流型号 (数据中心):
            *   Ampere: A100 (40GB/80GB VRAM)。
            *   Hopper: H100 (80GB VRAM), H200 (141GB VRAM) - 带宽和容量提升。
        *   **最新架构: Blackwell:** B100, B200, GB200 超级芯片 - 新特性包括第二代 Transformer Engine (支持 FP4/FP6 精度), 第五代 NVLink, 推理能力大幅提升。
        *   消费级显卡: RTX 3090 (24GB), RTX 4090 (24GB) - VRAM 较大, 适合本地实验和小型模型微调。
        *   CUDA 生态系统: NVIDIA 的并行计算平台和 API, 包括 cuDNN (深度神经网络库), NCCL (多卡通信库), TensorRT (推理优化库) 等。
    *   **其他加速器:**
        *   Google TPU (Tensor Processing Unit): v4, v5e, v5p - 专为 TensorFlow/JAX 优化, 在 Google Cloud 上可用。
        *   AMD GPU: Instinct MI300X - 对标 H100, ROCm (AMD 的类 CUDA 软件栈) 生态在逐步完善。
        *   Intel Gaudi: AI 加速器。
    *   **选择因素:** **VRAM 容量** > 性能 (特定精度) > 成本 > 生态系统支持 (CUDA 最成熟) > 功耗与散热。

*   **7.4 数据处理 (大规模) (Large-Scale Data Processing):**
    *   **Apache Spark:** 成熟的分布式数据处理框架。
        *   核心 API: RDD (弹性分布式数据集 - 较底层), DataFrame/Dataset API (类 Pandas, 更常用), Spark SQL。
        *   MLlib: Spark 的机器学习库 (包含基础算法)。
        *   适用场景: 大规模 ETL (抽取、转换、加载), 数据清洗, 特征工程。
    *   **Ray:** 新一代分布式计算框架, 专注于 AI 工作负载。
        *   核心概念: Actor 模型 (有状态服务), Task 并行 (无状态函数)。
        *   Ray AIR (AI Runtime): 统一的库 (Ray Data, Train, Tune, Serve) 用于端到端 AI 应用。
        *   Ray Data: 分布式数据加载与处理。
        *   适用场景: 大规模分布式训练 (如 Deepspeed on Ray), 超参数搜索 (Ray Tune), 模型服务 (Ray Serve), 强化学习 (RLlib)。
    *   **Dask:** 提供并行化的 NumPy, Pandas, Scikit-learn 接口。
        *   优点: 易于将现有 Python 代码扩展到多核或集群。
        *   适用场景: 中等规模的数据处理和并行计算。

*   **7.5 负责任 AI (Responsible AI):** **(补充: 确保 AI 系统的开发和使用符合道德、法律和社会规范)**
    *   **公平性 (Fairness):**
        *   目标: 避免 AI 系统对不同群体产生不公平的偏见或歧视。
        *   偏见来源: 数据偏见 (训练数据不均衡或包含歧视), 算法偏见, 人类反馈偏见。
        *   度量指标: 如 Demographic Parity (不同群体预测结果分布一致), Equal Opportunity (不同群体正样本预测准确率一致)。
        *   缓解技术: 数据预处理 (重采样, 重加权), 算法层面调整 (In-processing), 输出后处理。
    *   **可解释性/透明度 (Interpretability/Explainability/Transparency):**
        *   目标: 理解模型如何做出决策。
        *   方法/工具:
            *   SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanations): 解释模型对单个预测的贡献度 (对复杂 LLM 适用性有限)。
            *   Attention 可视化: 有限地展示模型关注点。
        *   文档化:
            *   模型卡 (Model Cards): 记录模型预期用途、性能、局限性、伦理考量。
            *   数据表 (Datasheets for Datasets): 记录数据集来源、构成、收集过程、潜在偏见。
            *   系统卡 (System Cards): 描述整个 AI 系统的构成和行为。
    *   **鲁棒性/安全性 (Robustness/Safety):**
        *   目标: 确保模型在各种情况下 (包括恶意攻击) 都能稳定、安全地运行。
        *   对抗攻击 (Adversarial Attacks): 对输入进行微小、人难以察觉的扰动, 导致模型输出错误。
        *   内容安全:
            *   Prompt Injection 防范: 防止恶意用户通过 Prompt 操纵 LLM 行为。
            *   Guardrails: 设置规则过滤不当输入、审核敏感输出、检测越狱尝试。
        *   幻觉 (Hallucination) 检测与缓解: 识别并减少模型捏造事实的情况 (如结合 RAG, 让模型承认“不知道”)。
    *   **隐私 (Privacy):**
        *   目标: 保护训练数据和用户数据中的敏感信息。
        *   数据处理: 数据匿名化/假名化技术。
        *   隐私增强技术 (PETs):
            *   差分隐私 (Differential Privacy): 在数据分析结果中加入噪声, 提供数学保证的隐私保护 (ε, δ 参数)。
            *   联邦学习 (Federated Learning): 在本地设备上训练模型, 只上传模型更新而非原始数据。
            *   同态加密 (Homomorphic Encryption): 允许在加密数据上进行计算。
    *   **问责制 (Accountability):**
        *   目标: 明确 AI 系统开发、部署、使用中的责任主体。
        *   实践: 建立审计追踪机制, 进行风险评估, 制定治理框架。
    *   **法规与治理:**
        *   了解相关法规: 如欧盟《人工智能法案》(EU AI Act), 中国《生成式人工智能服务管理暂行办法》等。
        *   建立内部治理机制: 确保 AI 开发和应用符合法规和伦理要求。
