学习自用，旨在梳理一套从基础到前沿的 AI 实用技术栈。

---

**1. 基础 (Foundations)**

*   **1.1 Python 基础 (Prerequisites)**
    *   核心语法、数据结构（列表、字典、元组、集合）、函数、类、模块。
    *   常用标准库。
    *   面向对象编程 (OOP) 思想。
    *   虚拟环境管理 (`venv`, `conda`)。

*   **1.2 数据科学基础库 (Data Science Fundamentals)**
    *   **NumPy (`ndarray`)**
        *   核心概念: 多维数组 (`ndarray`)。
        *   创建: `np.array()`, `np.arange()`, `np.linspace()`, `np.zeros()`, `np.ones()`, `np.random.*`。
        *   属性: `.shape`, `.dtype`, `.ndim`, `.size`。
        *   索引与切片: 基础索引 (整数, 切片), 花式索引 (数组/列表索引), 布尔索引 (条件过滤)。
        *   运算: 元素级运算 (+, -, *, /, etc.), 广播机制 (Broadcasting Rules)。
        *   函数:
            *   聚合函数 (`np.sum`, `np.mean`, `np.std`, `np.min`, `np.max`, `np.argmin`, `np.argmax`)。
            *   通用函数 (ufuncs: `np.sqrt`, `np.exp`, `np.log`, `np.sin`, etc.)。
            *   线性代数 (`np.dot`, `np.matmul`, `np.linalg.inv`, `np.linalg.solve`)。
        *   操作: `reshape`, `flatten`, `concatenate`, `vstack`, `hstack`, `split`。
        *   向量化思想: 利用 NumPy 内置函数替代 Python 循环。
    *   **Pandas (`Series`, `DataFrame`)**
        *   数据结构: `Series` (创建, 索引), `DataFrame` (创建, 行/列索引, 多级索引基础)。
        *   数据导入/导出: `pd.read_csv`, `pd.read_excel`, `pd.read_json`, `pd.read_sql`; `.to_csv`, `.to_excel` etc.
        *   数据查看/检查: `.head()`, `.tail()`, `.info()`, `.describe()`, `.value_counts()`, `.isnull().sum()`, `.duplicated().sum()`。
        *   索引与选择: 列选择 (`df['col_name']`), 基于标签 (`.loc[]`), 基于位置 (`.iloc[]`), 条件选择 (布尔索引)。
        *   数据清洗: 缺失值处理 (`.dropna()`, `.fillna()`), 重复值处理 (`.duplicated()`, `.drop_duplicates()`)。
        *   数据转换: 类型转换 (`.astype()`), 应用函数 (`.apply()`, `.map()`, `.applymap()`), 重命名 (`.rename()`), 替换值 (`.replace()`)。
        *   合并与连接: `.merge()` (how, on), `.concat()` (axis), `.join()`。
        *   分组与聚合: `.groupby()` 创建对象, 聚合操作 (`.agg()`, `.sum()`, `.mean`, etc.), 应用自定义函数, 多重聚合。
        *   透视表与交叉表: `pd.pivot_table()`, `pd.crosstab()`。
        *   时间序列: `datetime` 对象, `DatetimeIndex`, 重采样 (`.resample()`), 滑动窗口 (`.rolling()`)。
        *   性能与新特性: 理解向量化操作的重要性, 了解 Pandas 2.x Copy-on-Write。
    *   **数据可视化 (Data Visualization)**
        *   **Matplotlib:** 核心绘图库, 理解 `Figure`, `Axes` 对象, 绘制常用图表 (线图, 散点图, 柱状图, 直方图), 定制化 (标签, 标题, 图例)。
        *   **Seaborn:** 基于 Matplotlib 的高级接口, 简化统计图形绘制 (热力图, 分布图, 关系图), 美观的默认样式。

*   **1.3 机器学习核心 (Machine Learning Core - Scikit-learn)**
    *   **核心 API & 理念**
        *   Estimator 接口: `fit()`, `predict()`, `transform()`, `fit_predict()`, `fit_transform()` 的统一性。
        *   数据表示: NumPy arrays, SciPy sparse matrices, Pandas DataFrames。
    *   **数据预处理 (`sklearn.preprocessing`)**
        *   特征缩放: `StandardScaler`, `MinMaxScaler`, `RobustScaler` (原理与适用场景)。
        *   特征编码: `LabelEncoder` (目标变量), `OneHotEncoder` (类别特征), `OrdinalEncoder` (有序类别)。
        *   缺失值填充: `SimpleImputer` (策略: mean, median, most_frequent, constant)。
    *   **模型评估 (`sklearn.metrics`)**
        *   分类: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report` (理解各指标含义及权衡)。
        *   回归: `mean_squared_error`, `mean_absolute_error`, `r2_score`。
        *   聚类: `silhouette_score`。
    *   **模型选择与调优 (`sklearn.model_selection`)**
        *   数据划分: `train_test_split` (`stratify` 参数)。
        *   交叉验证: `KFold`, `StratifiedKFold`, `cross_val_score` (理解其必要性)。
        *   超参数搜索: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV` (原理, 参数空间定义)。
    *   **管道 (`sklearn.pipeline`)**
        *   `Pipeline`: 创建步骤 (`steps` 列表), 优点 (简化流程, 防止数据泄露), 与 GridSearchCV 结合。

*   **1.4 监督学习算法 (Supervised Learning Algorithms - Scikit-learn)**
    *   线性模型 (`sklearn.linear_model`):
        *   `LinearRegression`: 最小二乘法。
        *   `Ridge`, `Lasso`: L2/L1 正则化原理, alpha 参数作用。
        *   `LogisticRegression`: Sigmoid, 交叉熵损失, `C` 参数 (正则化强度), `solver`, `multi_class` 策略。
    *   支持向量机 (SVM) (`sklearn.svm`):
        *   `SVC`, `SVR`: 最大间隔原理, 支持向量。
        *   核函数 (`kernel`): 'linear', 'poly', 'rbf', 'sigmoid'。
        *   超参数: `C` (惩罚系数), `gamma` (RBF 核宽度)。
    *   决策树 (`sklearn.tree`):
        *   `DecisionTreeClassifier`, `DecisionTreeRegressor`: 分裂标准 (gini, entropy, mse), 树生长, 剪枝参数 (`max_depth`, `min_samples_split`, `min_samples_leaf`)。
    *   集成学习 (`sklearn.ensemble`):
        *   `RandomForestClassifier`, `RandomForestRegressor`: Bagging 原理 (Bootstrap Aggregating), 特征随机性 (`max_features`)。
        *   `AdaBoostClassifier`, `GradientBoostingClassifier`/`Regressor`: Boosting 原理 (串行学习, 关注错误样本/残差)。
        *   `HistGradientBoostingClassifier`/`Regressor`: 高效的梯度提升实现。

*   **1.5 无监督学习算法 (Unsupervised Learning Algorithms - Scikit-learn)**
    *   聚类 (`sklearn.cluster`): `KMeans` (算法流程, `n_clusters` 选择 - 肘部法则, 轮廓系数, 初始点问题)。
    *   降维 (`sklearn.decomposition`): `PCA` (主成分原理, 方差解释率, `n_components` 选择)。

*   **1.6 高级梯度提升 (Advanced Gradient Boosting)**
    *   **核心原理:** GBDT (梯度提升决策树) 的加法模型和前向分步思想。
    *   **XGBoost:**
        *   优化点: 正则化项 (L1/L2), 损失函数的二阶泰勒展开, 内置缺失值处理, 块结构并行化, 近似分位数算法。
        *   API: 原生 API (`xgb.train`, `xgb.DMatrix`), Scikit-learn API (`XGBClassifier`, `XGBRegressor`)。
        *   关键参数: `eta`, `max_depth`, `subsample`, `colsample_bytree`, `lambda`, `alpha`, `gamma`, `objective`, `eval_metric`。
    *   **LightGBM:**
        *   优化点: GOSS (梯度单边采样), EFB (互斥特征捆绑), 基于直方图的算法, Leaf-wise 生长策略。
        *   API: 原生 API (`lgb.train`, `lgb.Dataset`), Scikit-learn API (`LGBMClassifier`, `LGBMRegressor`)。
        *   关键参数: `learning_rate`, `num_leaves`, `max_depth`, `bagging_fraction`, `feature_fraction`, `reg_lambda`, `reg_alpha`, `objective`, `metric`。
    *   **CatBoost:**
        *   优化点: 对类别特征的自动高效处理 (Ordered Boosting, Target-based statistics), 对称树。
        *   API: Scikit-learn API (`CatBoostClassifier`, `CatBoostRegressor`)。
        *   关键参数: 类似 LightGBM, 关注 `cat_features` 参数。
    *   **通用实践:** 特征重要性获取 (`.feature_importances_`), 早停法 (`early_stopping_rounds`)。

---

**2. 深度学习 (Deep Learning)**

*   **2.1 深度学习框架基础 (Deep Learning Framework Basics)**
    *   **PyTorch**
        *   Tensor 核心: 创建与类型 (`torch.tensor`, `dtype`), 数学运算 & 形状操作, 与 NumPy 转换 (`.numpy()`, `torch.from_numpy()`)。
        *   梯度追踪: `requires_grad=True`, `.grad` 属性, `.detach()`。
        *   自动求导 (`torch.autograd`): 动态计算图, 梯度计算 (`loss.backward()`), 关闭梯度计算 (`with torch.no_grad():`, `@torch.no_grad()`, `model.eval()`)。
        *   神经网络模块基类: `nn.Module` (子类化, `__init__` 定义层, `forward` 定义逻辑)。
        *   优化器 (`torch.optim`): 常用 (`SGD`, `Adam`, `AdamW`), 优化流程 (`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`), 学习率调度 (`lr_scheduler`)。
        *   数据加载 (`torch.utils.data`): `Dataset` (自定义 `__len__`, `__getitem__`), `DataLoader` (`batch_size`, `shuffle`, `num_workers`, `collate_fn`)。
        *   训练与评估: 标准训练循环逻辑, 评估模式 (`model.eval()`)。
        *   GPU 使用: `torch.device`, `.to(device)`, `torch.cuda.is_available()`。
        *   保存与加载: `state_dict` (`model.state_dict()`, `torch.save()`, `model.load_state_dict(torch.load())`)。
    *   **(可选) JAX**
        *   函数式范式: Pure functions。
        *   核心变换: `grad`, `jit`, `vmap`, `pmap` 的原理和使用。
        *   NumPy-like API: 易于上手。
        *   设备无关性: CPU/GPU/TPU。
        *   生态: Flax, Haiku (神经网络库)。

*   **2.2 神经网络构建块 (Neural Network Building Blocks - PyTorch `torch.nn`)**
    *   常用层:
        *   `nn.Linear`: `in_features`, `out_features`.
        *   `nn.Conv2d`: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`.
        *   `nn.MaxPool2d`, `nn.AvgPool2d`: `kernel_size`, `stride`.
        *   `nn.RNN`, `nn.LSTM`, `nn.GRU`: `input_size`, `hidden_size`, `num_layers`, `batch_first=True`, 理解输入输出形状和隐藏状态。
        *   `nn.Embedding`: `num_embeddings`, `embedding_dim`.
        *   `nn.BatchNorm1d/2d/3d`: 工作原理, `momentum`, `eps`.
        *   `nn.Dropout`: `p` 参数.
        *   `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`, `nn.Transformer`.
    *   激活函数: `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softmax`, `nn.GELU` (函数式接口 `torch.nn.functional` F)。
    *   损失函数: `nn.CrossEntropyLoss` (包含 LogSoftmax), `nn.MSELoss`, `nn.L1Loss`, `nn.BCELoss`, `nn.BCEWithLogitsLoss` (包含 Sigmoid)。
    *   容器: `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`。

*   **2.3 经典深度学习模型 (Classic Deep Learning Models)**
    *   **CNN:**
        *   卷积层 (`Conv2d`): 参数共享, 局部连接。
        *   池化层 (`MaxPool2d`, `AvgPool2d`): 作用 (降维, 平移不变性)。
        *   经典架构思想: LeNet, AlexNet (ReLU, Dropout), VGG (堆叠小卷积核), GoogLeNet (Inception 模块), ResNet (残差连接)。
    *   **RNN:**
        *   基本单元: 隐藏状态的循环传递。
        *   问题: 长期依赖问题 (梯度消失/爆炸)。
        *   LSTM: 门控机制 (遗忘门, 输入门, 输出门), 细胞状态 (`cell state`)。
        *   GRU: 简化门控 (更新门, 重置门)。
        *   双向 RNN (BiRNN): 结合过去和未来信息。

*   **2.4 Transformer 架构 (Transformer Architecture)**
    *   **核心组件:**
        *   自注意力 (Self-Attention): Q, K, V, Scaled Dot-Product Attention (`softmax(QK^T / sqrt(d_k))V`), Masking (Padding, Look-ahead)。
        *   多头自注意力 (Multi-Head Attention): 并行计算, 拼接输出, 目的。
        *   位置编码 (Positional Encoding): 必要性, 方法 (Sinusoidal, Learned, RoPE, AliBi)。
        *   Add & Norm: 残差连接, 层归一化, 应用位置。
        *   Feed-Forward Network (FFN): 结构, 作用。
    *   **整体架构:**
        *   Encoder: 堆叠 Encoder Layer。
        *   Decoder: 堆叠 Decoder Layer (含 Masked Self-Attention, Cross-Attention)。
        *   架构类型: Encoder-Only (BERT), Decoder-Only (GPT, Llama), Encoder-Decoder (T5, BART)。
    *   **注意力变体与优化 (概念):** FlashAttention, 滑动窗口注意力, 稀疏注意力。

*   **2.5 PyTorch 进阶 (Advanced PyTorch)**
    *   分布式训练 (`torch.distributed`): 基本概念 (`world_size`, `rank`, `backend`), 常用策略 (`DistributedDataParallel` - DDP), 启动方式 (`torchrun`)。
    *   PyTorch 2.x 编译优化: `torch.compile()` (通过 TorchDynamo/TorchInductor 加速)。

---

**3. 大语言模型 / 基础模型 (LLMs / Foundation Models)**

*   **3.1 LLM 基础与生态 (LLM Basics & Ecosystem)**
    *   **Hugging Face 生态:**
        *   `transformers`:
            *   `pipeline`: 快速原型验证。
            *   `AutoClass`: `AutoModel`, `AutoTokenizer`, `AutoConfig`.
            *   模型加载与 Hub 交互: `from_pretrained()`, `push_to_hub()`.
            *   推理: `model(**inputs)` (获取 hidden states/logits), `model.generate()` (文本生成, 参数)。
        *   `datasets`: 加载 (`load_dataset`), 核心结构 (`Dataset`, `DatasetDict`), 处理 (`.map`, `.filter`, `.shuffle`, `.select`), Arrow 格式与流式处理。
        *   `tokenizers`: 原理 (BPE, WordPiece, SentencePiece), 使用 (`tokenizer()`), 特殊 Tokens, 解码 (`.decode()`).
    *   **(新增) Prompt Engineering 基础**
        *   基本原则: 清晰、具体、提供上下文。
        *   常用技巧: Zero-shot, Few-shot learning, Chain-of-Thought (CoT), Role prompting。
        *   迭代与优化。

*   **3.2 主流 LLM 交互 (Interacting with Mainstream LLMs)**
    *   **LLM APIs (OpenAI, Anthropic, Google etc.)**
        *   通用: API Key 管理, 模型选择 (能力/成本权衡), 速率限制。
        *   请求参数: `model`, `messages` (角色 `system`, `user`, `assistant`), `prompt`, `max_tokens`, `temperature`, `top_p`, `stream=True`.
        *   响应处理: 解析 JSON, 处理流式输出。
        *   多轮对话: 维护 `messages` 历史。
        *   Function Calling / Tool Use (概念与流程): 定义 Schema, 发送请求, 解析响应, 执行工具, 返回结果。
        *   关注点: 最新模型 (GPT-4o, Claude 3, Gemini 1.5 Pro), 多模态能力, 长上下文支持, 安全特性。
    *   **开源模型 (Llama 3, Mistral, Qwen etc.)**
        *   了解: 模型家族 (规模, 版本), 训练数据, **许可证 (License)**。
        *   本地部署基础: 硬件评估 (VRAM), 环境搭建, 模型下载 (Hub, `git lfs`), 加载与运行 (`transformers` 或推理引擎)。
        *   关注点: Llama 3 性能/开放性, Mixtral MoE 架构, Qwen 多模态/长上下文, 微调社区。

*   **3.3 高级 LLM 架构与概念 (Advanced LLM Architectures & Concepts)**
    *   **混合专家模型 (MoE)**
        *   原理: 稀疏激活, 路由器 (Gating Network) 选择 Top-k 专家, 专家网络 (FFN)。
        *   优/缺点: 参数扩展 vs 计算量控制; 训练/推理挑战。
        *   范例: Mixtral 8x7B。
    *   **多模态模型**
        *   输入处理: 图像编码器 (ViT), 音频编码器, 与文本 Embedding 融合。
        *   架构: LLaVA, CogVLM, GPT-4V/Omni, Gemini。
        *   应用: 图像描述, VQA, 图文生成, 语音交互。

*   **3.4 LLM 微调与适配 (LLM Fine-tuning & Adaptation)**
    *   **Hugging Face `Trainer`:**
        *   `TrainingArguments`: 配置训练参数。
        *   数据处理: 结合 `datasets`, `compute_metrics` 函数。
        *   训练流程: `.train()`, `.evaluate()`, `.predict()`.
    *   **Hugging Face `peft` (参数高效微调概念):**
        *   动机: 减少全量微调的计算和存储成本。
        *   基本思想: 冻结大部分模型参数, 仅训练少量额外参数或修改部分参数。
        *   常见方法概览: LoRA, Adapter Tuning, Prefix Tuning (细节见优化章节)。
    *   **Hugging Face `accelerate`:**
        *   `Accelerator` 对象: 自动设备管理, 梯度同步, 简化分布式训练和混合精度训练。
        *   核心用法: `accelerator.prepare()`, `accelerator.backward()`.
        *   启动: `accelerate config`, `accelerate launch`.

---

**4. 模型效率与优化 (Efficiency & Optimization)**

*   **4.1 参数高效微调 (Parameter-Efficient Fine-Tuning - PEFT)**
    *   **LoRA (Low-Rank Adaptation):**
        *   原理: 低秩分解 `ΔW ≈ BA`, 只训练 `A`, `B`.
        *   实现: `peft` 库, `LoraConfig`, `target_modules` (通常是 `query`, `key`, `value` 层)。
        *   参数: `r` (秩), `lora_alpha` (缩放因子), `lora_dropout`, `task_type`.
        *   保存/加载适配器: `save_pretrained`, `from_pretrained`.
    *   **QLoRA (Quantized LoRA):**
        *   原理: 4-bit NormalFloat (NF4) 量化基础模型 + LoRA 微调。
        *   关键技术: `bitsandbytes` (量化), Paged Optimizers, Double Quantization.
        *   效果: 显著降低微调显存需求。
    *   其他方法: Adapter Tuning, Prefix Tuning, IA3 (基本概念及适用场景)。
    *   变体: DoRA (权重分解 LoRA) 等最新研究。

*   **4.2 模型量化 (Quantization)**
    *   目标: 减小模型尺寸, 加速推理, 降低能耗。
    *   精度类型: FP32 -> FP16/BF16 -> INT8 -> FP4/NF4.
    *   方法: PTQ (训练后量化), QAT (量化感知训练)。
    *   工具/格式:
        *   `bitsandbytes`: NF4/FP4 量化 (用于 QLoRA)。
        *   **GGUF (llama.cpp):** CPU/Mac/边缘设备运行 LLM 的关键格式, 支持多种量化策略 (Q4_0, Q5_K_M等)。
        *   `AutoGPTQ`, `AWQ`: 常用的 PTQ 方法库。
        *   ONNX Runtime: 支持多种量化模型部署。
    *   权衡: 精度损失 vs 效率提升。

*   **4.3 高效注意力 (Efficient Attention)**
    *   **FlashAttention (v1/v2):**
        *   原理: IO 感知, Tiling (分块计算), Recomputation (重计算而非存储中间结果), 减少 HBM 读写。
        *   集成: 已被主流库 (PyTorch, `transformers`, `vLLM` 等) 集成, 用户通常透明受益。

*   **4.4 推理优化引擎 (Inference Optimization Engines)**
    *   目标: 高吞吐, 低延迟, 省显存。
    *   **核心技术:**
        *   **PagedAttention:** 通过分页管理 KV Cache, 解决内存碎片, 实现近乎零浪费。
        *   **Continuous Batching:** 请求动态加入批次, 提高 GPU 利用率。
        *   算子融合 (Operator Fusion): 减少 Kernel Launch 开销和内存读写。
        *   张量并行 (Tensor Parallelism): 分割模型权重到多 GPU。
    *   **引擎:**
        *   `vLLM`: Python API (`LLM` 类), OpenAI 兼容服务器。易用性好。
        *   `TensorRT-LLM`: NVIDIA 官方库, 深度优化 NVIDIA GPU, 需要模型编译步骤。性能潜力大。
        *   `DeepSpeed Inference`: 结合 ZeRO-Inference 和模型并行。
    *   关注点: 对 MoE, 长上下文, 量化的支持。

*   **4.5 投机解码 (Speculative Decoding)**
    *   原理: 用一个快速的小模型生成多个候选 token (草稿), 然后用原始大模型一次性并行验证这些 token。
    *   目标: 在保持生成质量的同时, 显著降低生成延迟 (尤其对于交互式应用)。
    *   实现: 需要大小两个模型配合, 已在一些推理框架中得到支持。

---

**5. AI Agent 与应用 (Agents & Applications)**

*   **5.1 核心框架 (Core Frameworks)**
    *   **LangChain:**
        *   LCEL (LangChain Expression Language): `|` 操作符连接组件, 提升可组合性。
        *   组件: `ChatPromptTemplate`, `ChatModel`/`LLM`, `OutputParser` (如 `StrOutputParser`), `RunnablePassthrough`.
        *   Chains: `SequentialChain`, 理解链式调用的概念。
        *   Agents: `ReAct` 逻辑 (Reason + Act), `AgentExecutor`, `Tools` 定义, `create_openai_tools_agent` 等 Agent 构造器。
        *   Memory: `ConversationBufferMemory` 等记忆组件。
    *   **LlamaIndex:** (侧重数据与 LLM 的连接)
        *   数据管道: `Reader` (如 `SimpleDirectoryReader`) -> `NodeParser` (如 `SentenceSplitter`) -> `Index` (如 `VectorStoreIndex`) -> `Retriever` -> `QueryEngine` / `ChatEngine`.
        *   索引类型: `VectorStoreIndex`, `SummaryIndex`, `KeywordTableIndex`.
        *   检索策略: `similarity_top_k`, `node_postprocessors` (如 Re-ranking)。
        *   查询/聊天引擎: 定制 Prompt, 流式输出。
    *   **AutoGen:** (侧重多 Agent 对话)
        *   Agent 定义: `ConversableAgent`, `AssistantAgent` (LLM驱动), `UserProxyAgent` (可执行代码/人类输入)。
        *   交互模式: `initiate_chat`, 多 Agent 自动对话与协作。
    *   **CrewAI:** (侧重结构化 Agent 协作流程)
        *   核心概念: `Agent` (role, goal, backstory, tools), `Task` (description, agent, expected_output), `Crew` (agents, tasks, process - sequential/hierarchical), `Process`.

*   **5.2 向量数据库 (Vector Databases)**
    *   **核心概念:** Embedding (向量表示), ANN (近似最近邻) 搜索。
    *   **关键技术/算法:** HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), PQ (Product Quantization) 等索引算法原理。
    *   **数据库操作 (以 Chroma/Milvus/Pinecone 为例):**
        *   连接/客户端初始化。
        *   创建 Collection/Index (指定名称, embedding function/维度, 距离度量 `cosine`/`l2`/`ip`).
        *   添加/更新数据: `.add()`, `.upsert()` (包含 vectors, metadatas, ids)。
        *   查询: `.query()` (输入 `query_embeddings`, `n_results`, `where`/`filter` 元数据过滤, `include` 返回内容)。
        *   删除数据: `.delete()`.
    *   **选择考量:** 开源 vs 商业, 本地部署 vs 云服务, 性能, 扩展性, 元数据过滤能力, 混合存储, 社区支持。

*   **5.3 检索增强生成 (Retrieval-Augmented Generation - RAG)**
    *   **基础 RAG 流程:**
        *   Indexing (索引): 文档加载 -> 分块 (Chunking) -> 向量化 (Embedding) -> 存入向量数据库。
        *   Retrieval (检索): 用户查询向量化 -> 在向量库中搜索相似块 -> 获取 Top-K 相关块。
        *   Generation (生成): 将用户查询和检索到的上下文块组合成 Prompt -> 输入 LLM 生成答案。
    *   **高级 RAG 技术:**
        *   **查询转换 (Query Transformation):** Rewriting (改写), Sub-Query (子查询分解), HyDE (生成假设性文档)。
        *   **分块与索引 (Chunking & Indexing):** Chunk Size & Overlap 策略, Sentence Window Retrieval (检索句子返回窗口), Hierarchical Indexing (摘要+原始块)。
        *   **检索优化 (Retrieval Optimization):** Hybrid Search (稀疏 BM25 + 稠密向量), Re-ranking (Cross-Encoder 重排序), Fusion (RRF 结果融合)。
        *   **生成优化 (Generation Optimization):** 设计 Prompt 有效利用上下文, 处理长上下文 (压缩, 选择性使用)。

*   **5.4 Tool Use / Function Calling**
    *   **LLM API 侧:** 定义 `tools` 列表 (JSON Schema: name, description, parameters), 设置 `tool_choice` ("auto", "required", specific tool)。
    *   **应用侧:**
        *   解析模型返回的 `tool_calls` (ID, function name, arguments)。
        *   查找并执行对应的本地函数或外部 API。
        *   构造 `role: tool` 的消息返回给 LLM (包含 `tool_call_id` 和执行结果 `content`)。
    *   **可靠性:** 处理模型未正确调用、参数错误、函数执行失败等情况, 添加重试或错误处理逻辑。
    *   **框架支持:** LangChain, LlamaIndex 等框架对 Tool Use 的封装。

*   **5.5 多 Agent 系统 (Multi-Agent Systems)**
    *   概念: 多个具有不同角色、能力、目标的 AI Agent 通过协作完成复杂任务。
    *   模式: 分层 (管理者-执行者), 对话式 (辩论, 评审, 模拟), 并行执行。
    *   框架使用: `AutoGen` (灵活的对话模式), `CrewAI` (结构化协作流程)。
    *   挑战: 规划协调, 通信效率, 状态同步, 错误处理, 成本控制, Agent 间的冲突解决。

---

**6. 支撑技术与平台 (Supporting Technologies & Platforms)**

*   **6.1 MLOps / LLMOps**
    *   **基础 MLOps:**
        *   版本控制 (`Git`): 分支管理, 提交规范, Pull Requests, Merge。代码和配置管理。
        *   实验跟踪 (`W&B`/`MLflow`): `init`, `log` (metrics, params, artifacts, models), Web UI 查看与比较, 可复现性。
        *   容器化 (`Docker`): `Dockerfile` 编写, 环境隔离与打包, `docker build`, `docker run`.
        *   CI/CD 概念: 持续集成, 持续交付/部署 (自动化测试、构建、部署流程)。
    *   **LLMOps 特有挑战与实践:**
        *   **Prompt 管理:** Prompt 版本控制, Prompt 模板库, A/B 测试, 性能监控与回溯。
        *   **LLM 评估:** 自动化指标 (BLEU, ROUGE - 局限性), 基于模型评估 (GPT-4 Score), 人工评估 (重要), RAG 评估 (Context Relevance, Faithfulness, Answer Relevance - Ragas, TruLens, ARES), 安全性/偏见评估。
        *   **Fine-tuning 工作流:** 数据集管理与版本化, 超参数搜索记录, PEFT 适配器管理与部署。
        *   **RAG 数据管道:** 索引更新策略 (实时/批量), 检索质量监控, Embedding 模型管理。
        *   **成本与延迟监控:** Token 使用量跟踪, API 调用延迟分析, 推理成本优化。
        *   **安全与对齐:** Guardrails (输入/输出过滤), 幻觉检测, 内容审核 API 集成。

*   **6.2 云平台 (Cloud Platforms - AWS/GCP/Azure)**
    *   **计算:** GPU 实例类型选择 (VRAM, 型号, 成本), 启动/连接/管理 (SSH, Cloud Shell)。
    *   **存储:** 对象存储 (S3/GCS/Blob) - 数据集、模型、日志存储。
    *   **AI/ML 平台 (SageMaker/Vertex AI/Azure ML):**
        *   Notebook 环境 (托管 Jupyter)。
        *   **托管基础模型 API:** AWS Bedrock, Vertex AI Model Garden, Azure OpenAI Service (提供对多种基础模型的访问)。
        *   **托管训练/微调:** 配置数据集, 计算资源, 超参数, 提交与监控训练任务。
        *   **托管 RAG/Agent 服务:** 提供构建 RAG 应用或 Agent 的集成服务。
        *   模型部署: 创建 Endpoint, Serverless 推理, 批处理推理。
    *   **IAM (Identity and Access Management):** 用户, 角色, 权限管理基础, 保证资源安全访问。
    *   **成本管理:** 了解各服务定价模型, 使用成本监控和预算工具, 利用 Spot 实例降低成本。

*   **6.3 硬件 (Hardware)**
    *   **NVIDIA GPU:**
        *   关键指标: **VRAM 容量** (决定能跑多大模型), 计算能力 (TFLOPS - FP16/BF16/FP8/INT8, 不同精度下的性能), 内存带宽 (影响数据传输速度), NVLink (多卡互联带宽)。
        *   主流型号 (数据中心): A100, H100, H200 (重点关注 VRAM 和带宽提升)。
        *   **最新架构: Blackwell (B100, B200, GB200 超级芯片)** - 新特性 (第二代 Transformer Engine 支持 FP4/FP6, 第五代 NVLink, 推理能力大幅提升)。
        *   消费级显卡: RTX 3090/4090 (大 VRAM 适合本地实验)。
        *   CUDA 生态系统: cuDNN, NCCL, TensorRT 等加速库。
    *   **其他加速器:** Google TPU (v4, v5e, v5p - 优化 TensorFlow/JAX), AMD GPU (MI300X - 对标 H100, ROCm 生态)。
    *   **选择因素:** VRAM 容量是最优先考虑的, 其次是性能, 成本, 生态系统支持 (CUDA 通常最完善), 功耗和散热。

*   **6.4 数据处理 (大规模) (Large-Scale Data Processing)**
    *   **Apache Spark:** 分布式数据处理框架。RDD/DataFrame API, SQL on Spark, MLlib (基础概念), 适用于大规模 ETL 和数据预处理。
    *   **Ray:** 分布式计算框架。Actor 模型, Task 并行, Ray AIR (Train, Tune, Serve - 统一的 AI 运行时), Ray Data (分布式数据加载与处理), 常用于大规模分布式训练和超参数搜索。
    *   **Dask:** 提供类 Pandas/NumPy/Scikit-learn API 的分布式计算库, 易于将现有 Python 代码扩展到多核或集群。

*   **6.5 负责任 AI (Responsible AI)**
    *   **公平性 (Fairness):** 理解偏见来源 (数据, 算法, 人类反馈), 常用度量指标 (如 Demographic Parity, Equal Opportunity), 缓解技术 (预处理数据, 后处理输出, In-processing 算法调整)。
    *   **可解释性/透明度 (Interpretability/Explainability/Transparency):**
        *   方法/工具: SHAP, LIME (原理, 局限性, 对复杂 LLM 的适用性有限), Attention 可视化 (有限)。
        *   文档化: 模型卡 (Model Cards), 数据表 (Datasheets for Datasets), 系统卡 (System Cards)。
    *   **鲁棒性/安全性 (Robustness/Safety):**
        *   对抗攻击: 理解对输入微小扰动导致模型失效的风险。
        *   内容安全: Prompt Injection 防范, Guardrails (输入过滤, 输出审核, 敏感话题检测), 越狱尝试检测。
        *   幻觉检测与缓解: 基于知识库的验证, 让模型承认“不知道”。
    *   **隐私 (Privacy):**
        *   数据处理: 数据匿名化/假名化技术。
        *   隐私增强技术: 差分隐私 (Differential Privacy - 基本概念 ε, δ), 联邦学习 (Federated Learning - 不共享原始数据)。
    *   **问责制 (Accountability):** 审计追踪, 风险评估框架, 明确责任主体。
    *   **法规与治理:** 了解关键法规 (如欧盟 AI Act, 中国生成式人工智能服务管理暂行办法) 的基本要求和影响。