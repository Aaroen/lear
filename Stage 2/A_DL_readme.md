《动手学深度学习》(Dive into Deep Learning, D2L)的部分实践实现:


**整体章节梳理**

1.  **Introduction:** 介绍深度学习背景。
2.  **Preliminaries:**
    *   **Data Manipulation:** NDArray/Tensor 操作 (创建、运算、广播、索引、内存)。**【实践基础】**
    *   **Data Preprocessing:** 数据处理技术 (读写文件、处理缺失值、转换为张量)。**【实践基础】**
    *   **Linear Algebra:** 深度学习中常用的线性代数概念及实现。**【实践基础】**
    *   **Calculus:** 自动求导 (Autograd)。**【实践基础 - 核心】**
    *   **Probability:** 概率论基础。
3.  **Linear Neural Networks:**
    *   **Linear Regression:** 从零实现和使用框架实现。**【核心实践 - 入门】**
    *   **Softmax Regression:** 从零实现和使用框架实现。**【核心实践 - 入门】**
4.  **Multilayer Perceptrons (MLP):**
    *   MLP 概念与从零实现/框架实现。**【核心实践】**
    *   模型选择、欠拟合与过拟合 (权重衰减、Dropout)。**【重要实践 - 技巧】**
5.  **Deep Learning Computation:**
    *   层与块 (Layers and Blocks)。
    *   参数管理。
    *   自定义层。
    *   读写文件 (模型保存与加载)。**【重要实践 - 工程】**
    *   GPU 使用。**【重要实践 - 工程】**
6.  **Convolutional Neural Networks (CNNs):**
    *   卷积层、池化层概念与实现。**【核心实践】**
    *   经典 CNN 架构: LeNet, AlexNet, VGG, NiN, GoogLeNet, ResNet, DenseNet。**【重要实践 - 理解架构】**
7.  **Recurrent Neural Networks (RNNs):**
    *   序列模型概念。
    *   文本预处理、语言模型。**【重要实践 - NLP 基础】**
    *   RNN 从零实现/框架实现。**【核心实践】**
    *   GRU, LSTM。**【核心实践】**
    *   深度 RNN, 双向 RNN。
8.  **Attention Mechanisms & Transformers:**
    *   Seq2Seq 模型。
    *   Attention 机制原理与实现。**【核心实践 - 前沿】**
    *   Transformer 架构。**【核心实践 - 前沿】**
9.  **Optimization Algorithms:**
    *   梯度下降及其变种 (SGD, Momentum, AdaGrad, RMSProp, Adam)。**【重要实践 - 理解优化】**
10. **Computational Performance:**
    *   编译器与后端。
    *   多 GPU 训练。**【进阶实践】**
    *   参数服务器。
11. **Computer Vision (CV):**
    *   图像增广 (Image Augmentation)。**【重要实践 - CV 技巧】**
    *   微调 (Fine-tuning)。**【核心实践 - 应用】**
    *   目标检测 (Object Detection)。**【进阶实践 - CV 应用】**
    *   语义分割 (Semantic Segmentation)。**【进阶实践 - CV 应用】**
    *   风格迁移 (Style Transfer)。
12. **Natural Language Processing (NLP):**
    *   词嵌入 (Word Embeddings: Word2Vec, GloVe, FastText)。**【重要实践 - NLP 基础】**
    *   情感分析 (Sentiment Analysis)。**【重要实践 - NLP 应用】**
    *   文本分类。
    *   BERT 等预训练模型。**【核心实践 - NLP 前沿】**
13. **Recommender Systems (RecSys):** (可能包含)
14. **Generative Adversarial Networks (GANs):** (可能包含) **【进阶实践】**
15. **Reinforcement Learning (RL):** (可能包含) **【进阶实践】**

**核心适合实践部分**


**实验 1: 基础数据操作与自动求导 (对应 D2L Ch 2)**

*   **核心目标:** 熟练掌握深度学习框架（以 PyTorch 为例）中 Tensor 的基本操作，并理解自动求导机制如何计算梯度。
*   **所需知识/库:** Python 基础, NumPy 基础 (有助于理解), PyTorch (或 TensorFlow/JAX)。
*   **详细步骤与思路:**
    1.  **Tensor 创建:**
        *   **步骤:** 使用 `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.randn()`, `torch.arange()` 等函数创建不同形状和数据类型的 Tensor。
        *   **思路:** 熟悉 Tensor 的基本属性（shape, dtype）。
        *   **示例:** `x = torch.arange(12, dtype=torch.float32).reshape((3, 4))`
    2.  **Tensor 索引与切片:**
        *   **步骤:** 使用 Python 的索引和切片语法访问 Tensor 的元素、行、列或子块。尝试负数索引、步长切片。
        *   **思路:** 理解 Tensor 的多维索引方式。
        *   **示例:** `x[1, 2]`, `x[0:2, :]`, `x[:, 1::2]`
    3.  **Tensor 运算:**
        *   **步骤:** 执行逐元素运算 (+, -, \*, /, \*\*), 矩阵运算 (`@` 或 `torch.matmul`), 聚合运算 (`sum()`, `mean()`, `norm()`)。注意区分逐元素乘法和矩阵乘法。
        *   **思路:** 掌握常用数学运算在 Tensor 上的实现。
        *   **示例:** `a + b`, `torch.exp(x)`, `torch.matmul(x, w)`, `x.sum(axis=0)`
    4.  **广播机制 (Broadcasting):**
        *   **步骤:** 创建两个形状不同但兼容的 Tensor，执行运算，观察结果的形状。例如，一个 (3,1) Tensor 和一个 (1,2) Tensor 相加。
        *   **思路:** 理解广播机制如何自动扩展 Tensor 维度以匹配运算。
        *   **示例:** `a = torch.arange(3).reshape((3, 1))`, `b = torch.arange(2).reshape((1, 2))`, `a + b`
    5.  **自动求导 (Autograd):**
        *   **步骤 a (准备):** 创建一个需要计算梯度的 Tensor `x` (设置 `requires_grad=True`)。定义一个关于 `x` 的函数 `y` (例如 `y = 2 * torch.dot(x, x)`)。
        *   **步骤 b (计算梯度):** 调用 `y.backward()`。
        *   **步骤 c (查看梯度):** 访问 `x.grad` 查看 `y` 对 `x` 的梯度。
        *   **步骤 d (验证):** 手动计算 `y` 对 `x` 的导数 (例子中是 `4*x`)，与 `x.grad` 对比。
        *   **步骤 e (控制梯度计算):** 尝试使用 `with torch.no_grad():` 或 `.detach()` 来阻止梯度跟踪。
        *   **思路:** 理解 `.backward()` 如何触发计算图的反向传播，梯度如何累积在叶节点 Tensor 的 `.grad` 属性中。
*   **预期结果/验证方法:** Tensor 操作结果符合预期 (形状、值)。自动求导计算出的梯度与手动推导的梯度一致。
*   **关键注意点:** Tensor 的数据类型 (dtype)、形状 (shape)、是否需要梯度 (`requires_grad`)。梯度默认会累积，每次 `backward()` 前通常需要清零 (`x.grad.zero_()` 或 `optimizer.zero_grad()`)。

---

**实验 2: 线性回归 (从零实现与框架实现) (对应 D2L Ch 3)**

*   **核心目标:** 掌握一个完整的模型训练流程：数据加载、模型定义、损失函数、优化算法、迭代训练。理解线性回归模型。
*   **所需知识/库:** 实验 1 内容, PyTorch (`torch.nn`, `torch.optim`, `torch.utils.data`)。
*   **详细步骤与思路:**
    1.  **数据准备:**
        *   **步骤:** 生成带噪声的合成数据 `y = Xw + b + epsilon`，其中 `w` 和 `b` 是已知的真实参数。将数据转换为 Tensor。
        *   **思路:** 创建一个简单的、已知最优解的问题，方便验证模型效果。
        *   **示例:** `true_w = torch.tensor([2, -3.4])`, `true_b = 4.2`, `X = torch.randn(1000, 2)`, `y = X @ true_w + true_b + torch.randn(1000) * 0.01`
    2.  **实现版本一：从零开始 (仅用 Tensor 和 Autograd)**
        *   **步骤 a (初始化参数):** 随机初始化权重 `w` 和偏置 `b`，设置 `requires_grad=True`。
        *   **步骤 b (定义模型):** `def linreg(X, w, b): return X @ w + b`。
        *   **步骤 c (定义损失函数):** `def squared_loss(y_hat, y): return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2` (均方误差)。
        *   **步骤 d (定义优化算法 - SGD):** `def sgd(params, lr, batch_size): with torch.no_grad(): for param in params: param -= lr * param.grad / batch_size; param.grad.zero_()`。
        *   **步骤 e (数据迭代器):** 实现一个函数，每次返回一小批量数据 `(batch_X, batch_y)`。
        *   **步骤 f (训练循环):**
            *   设置学习率 `lr` 和迭代次数 `num_epochs`。
            *   外循环 `for epoch in range(num_epochs):`
            *   内循环 `for X_batch, y_batch in data_iter(batch_size, features, labels):`
                *   计算预测 `l = squared_loss(linreg(X_batch, w, b), y_batch)`。
                *   计算梯度 `l.sum().backward()`。 (注意是对 batch 的损失和求梯度)
                *   更新参数 `sgd([w, b], lr, batch_size)`。
            *   (可选) 每个 epoch 结束后计算并打印当前损失。
    3.  **实现版本二：使用框架高级 API**
        *   **步骤 a (数据加载):** 使用 `TensorDataset` 和 `DataLoader` 创建数据迭代器。
        *   **步骤 b (定义模型):** `net = nn.Sequential(nn.Linear(2, 1))` (输入特征 2，输出 1)。
        *   **步骤 c (初始化参数):** (框架会自动初始化，也可手动设置) `net[0].weight.data.normal_(0, 0.01)`, `net[0].bias.data.fill_(0)`。
        *   **步骤 d (定义损失函数):** `loss = nn.MSELoss()`。
        *   **步骤 e (定义优化器):** `optimizer = torch.optim.SGD(net.parameters(), lr=0.03)`。
        *   **步骤 f (训练循环):**
            *   `for epoch in range(num_epochs):`
            *   `for X_batch, y_batch in data_loader:`
                *   `l = loss(net(X_batch), y_batch)`。
                *   `optimizer.zero_grad()`。
                *   `l.backward()`。
                *   `optimizer.step()`。
            *   (可选) 打印损失。
    4.  **结果验证:**
        *   **步骤:** 比较训练得到的 `w` 和 `b` (或 `net[0].weight.data`, `net[0].bias.data`) 与 `true_w`, `true_b` 是否接近。观察训练过程中的损失是否逐渐下降。
*   **预期结果/验证方法:** 训练后的模型参数接近真实参数。损失函数值显著下降并趋于稳定。两种实现方式都能达到相似的结果。
*   **关键注意点:** 学习率的选择。从零实现时注意梯度清零和 `torch.no_grad()` 的使用。框架实现中理解 `nn.Module`, `nn.Loss`, `torch.optim` 和 `DataLoader` 的作用。

---

**实验 3: Softmax 回归 (从零实现与框架实现) (对应 D2L Ch 3)**

*   **核心目标:** 理解 Softmax 函数处理多分类问题，掌握交叉熵损失函数，并复习训练流程。
*   **所需知识/库:** 实验 1, 2, PyTorch, torchvision (加载 Fashion-MNIST 数据集)。
*   **详细步骤与思路:**
    1.  **数据准备:**
        *   **步骤:** 加载 Fashion-MNIST 数据集 (使用 `torchvision.datasets.FashionMNIST` 和 `DataLoader`)。这是一个 10 分类的图像数据集。获取图像特征 (展平成 28*28=784 维向量) 和标签。
        *   **思路:** 使用真实的多分类数据集。
    2.  **实现版本一：从零开始**
        *   **步骤 a (初始化参数):** 初始化权重 `W` (784 x 10) 和偏置 `b` (1 x 10)，设置 `requires_grad=True`。
        *   **步骤 b (定义 Softmax 操作):** `def softmax(X): X_exp = torch.exp(X); partition = X_exp.sum(1, keepdim=True); return X_exp / partition`。
        *   **步骤 c (定义模型):** `def net(X): return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)`。
        *   **步骤 d (定义损失函数 - 交叉熵):** `def cross_entropy(y_hat, y): return - torch.log(y_hat[range(len(y_hat)), y])` (简洁实现，利用整数索引)。
        *   **步骤 e (定义准确率计算):** `def accuracy(y_hat, y): return (y_hat.argmax(axis=1) == y).float().mean().item()`。
        *   **步骤 f (定义优化算法 - SGD):** 同线性回归从零实现。
        *   **步骤 g (训练循环):** 类似线性回归从零实现，但使用交叉熵损失和准确率评估。
    3.  **实现版本二：使用框架高级 API**
        *   **步骤 a (数据加载):** 同版本一。
        *   **步骤 b (定义模型):** `net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))` (Flatten 层将图像展平)。
        *   **步骤 c (初始化参数):** (可选，框架会自动初始化)。
        *   **步骤 d (定义损失函数):** `loss = nn.CrossEntropyLoss()` (注意：这个损失函数内部包含了 Softmax 操作，所以模型输出层不需要 Softmax)。
        *   **步骤 e (定义优化器):** `optimizer = torch.optim.SGD(net.parameters(), lr=0.1)`。
        *   **步骤 f (训练循环):** 类似线性回归框架实现，但使用 `CrossEntropyLoss`。在每个 epoch 结束后计算训练集和测试集上的准确率。
    4.  **结果验证:** 观察训练和测试准确率是否提升，损失是否下降。
*   **预期结果/验证方法:** 损失下降，准确率提升（在 Fashion-MNIST 上应能达到 80% 以上）。两种实现结果相似。
*   **关键注意点:** 交叉熵损失的使用（框架版本通常已包含 Softmax）。理解 Softmax 将输出转换为概率分布。多分类问题的评估指标（准确率）。

---

**实验 4: 多层感知机 (MLP) (从零实现与框架实现) (对应 D2L Ch 4)**

*   **核心目标:** 构建并训练包含隐藏层和非线性激活函数的神经网络。
*   **所需知识/库:** 实验 1-3, PyTorch。
*   **详细步骤与思路:**
    1.  **数据准备:** 继续使用 Fashion-MNIST 数据集。
    2.  **实现版本一：从零开始**
        *   **步骤 a (初始化参数):** 初始化输入层到隐藏层的权重 `W1`, `b1`；隐藏层到输出层的权重 `W2`, `b2`。设置 `requires_grad=True`。
        *   **步骤 b (定义激活函数):** `def relu(X): return torch.max(X, torch.zeros_like(X))`。
        *   **步骤 c (定义模型):** `def net(X): X = X.reshape((-1, num_inputs)); H = relu(X @ W1 + b1); return (H @ W2 + b2)` (注意：这里输出层未加 Softmax，将在交叉熵损失中处理)。
        *   **步骤 d (定义损失函数):** 使用前面定义的 `cross_entropy`（或者直接用框架的 `CrossEntropyLoss`，但输入需要是 logits）。
        *   **步骤 e (训练循环):** 同 Softmax 回归从零实现，但模型换成 MLP。
    3.  **实现版本二：使用框架高级 API**
        *   **步骤 a (数据加载):** 同上。
        *   **步骤 b (定义模型):** `net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_hiddens), nn.ReLU(), nn.Linear(num_hiddens, num_outputs))`。
        *   **步骤 c (定义损失和优化器):** `loss = nn.CrossEntropyLoss()`, `optimizer = torch.optim.SGD(net.parameters(), lr=lr)`。
        *   **步骤 d (训练循环):** 同 Softmax 回归框架实现。
    4.  **结果验证:** 对比 MLP 和 Softmax 回归在 Fashion-MNIST 上的准确率，MLP 通常效果更好。
*   **预期结果/验证方法:** MLP 在测试集上的准确率通常优于线性模型（Softmax 回归）。损失下降，准确率提升。
*   **关键注意点:** 隐藏层神经元数量的选择（超参数）。激活函数的选择（ReLU 是常用选择）。理解添加隐藏层和非线性激活函数的作用。

---

**实验 5: Dropout 实现 (对应 D2L Ch 4)**

*   **核心目标:** 理解 Dropout 如何工作以减轻过拟合，并能手动实现或使用框架的 Dropout 层。
*   **所需知识/库:** 实验 1-4, PyTorch (`nn.Dropout`)。
*   **详细步骤与思路:**
    1.  **理解 Dropout 原理:**
        *   **步骤:** 回顾 Dropout 的概念：在训练期间，以一定概率 `p` 随机将神经网络单元（通常是隐藏层单元）的输出设置为零。为了补偿被丢弃的单元，剩余未被丢弃单元的输出会被放大 `1 / (1 - p)` (Inverted Dropout)。在测试期间，不进行 Dropout。
        *   **思路:** 理解 Dropout 通过引入随机性、强制网络学习冗余表示来防止对训练数据中特定特征的过度依赖，从而提高模型的泛化能力。
    2.  **实现 Dropout 层 (从零，理解性):**
        *   **步骤 a (定义函数):** `def dropout_layer(X, dropout_prob): assert 0 <= dropout_prob <= 1; if dropout_prob == 1: return torch.zeros_like(X); mask = (torch.rand(X.shape) > dropout_prob).float(); return mask * X / (1.0 - dropout_prob)`
        *   **步骤 b (训练/测试模式):** 注意，从零实现时需要手动控制只在训练时应用 dropout。框架层 (`nn.Dropout`) 会自动根据 `model.train()` 或 `model.eval()` 模式切换行为。
        *   **思路:** 掌握 Dropout 的核心计算逻辑和缩放因子。
    3.  **使用框架 `nn.Dropout`:**
        *   **步骤 a (集成到模型):** 在 MLP 的隐藏层激活函数之后添加 `nn.Dropout(dropout_prob)` 层。例如：`nn.Linear(in_features, hidden_features)`, `nn.ReLU()`, `nn.Dropout(p=0.5)`, `nn.Linear(hidden_features, out_features)`。
        *   **步骤 b (设置模式):** 确保在训练前调用 `model.train()`，在评估/测试前调用 `model.eval()`。这会控制 `nn.Dropout` 和 `nn.BatchNorm2d` 等层的行为。
        *   **步骤 c (训练与对比):** 使用和不使用 Dropout 分别训练同一个 MLP 模型（在可能过拟合的数据集或设置下），比较两者的训练损失、测试损失和测试准确率随时间的变化。
    *   **预期结果/验证方法:** 使用 Dropout 的模型，其训练准确率可能略低于不用 Dropout 的模型，但测试准确率（或损失）通常会更好，或者训练损失和测试损失之间的差距更小，表明过拟合得到了缓解。
    *   **关键注意点:** Dropout 概率 `p` 是一个超参数，需要调整。只在训练时使用 Dropout。理解 Inverted Dropout 的缩放机制。`nn.Dropout` 层需要通过 `model.train()` 和 `model.eval()` 控制其行为。

---

**实验 6: 卷积与池化基础 & LeNet 实现 (对应 D2L Ch 6)**

*   **核心目标:** 理解二维卷积和池化的基本运算，并使用框架实现经典的 LeNet-5 模型。
*   **所需知识/库:** 实验 1-4, PyTorch (`nn.Conv2d`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.ReLU`/`nn.Sigmoid`), torchvision。
*   **详细步骤与思路:**
    1.  **卷积基础 (理解性实践):**
        *   **步骤 a:** 创建一个简单的输入 Tensor (如 1x1x8x8) 和一个卷积核 Tensor (如 1x1x3x3)。
        *   **步骤 b:** 使用 `nn.Conv2d` 层对输入进行卷积，观察输出形状。
        *   **步骤 c:** 尝试手动实现简单的互相关运算（使用嵌套循环或 `unfold` 等技巧，可选），对比结果。
        *   **步骤 d:** 探索 `padding` 和 `stride` 参数对输出形状的影响。
        *   **思路:** 关注卷积操作如何改变特征图的尺寸和通道数。
    2.  **池化基础 (理解性实践):**
        *   **步骤 a:** 创建一个简单的输入 Tensor。
        *   **步骤 b:** 使用 `nn.MaxPool2d` 或 `nn.AvgPool2d` 层进行池化，观察输出形状。
        *   **步骤 c:** 理解池化层的作用（降低分辨率，提取鲁棒特征）。
    3.  **LeNet-5 实现 (框架):**
        *   **步骤 a (数据加载):** 加载 Fashion-MNIST 数据集。
        *   **步骤 b (定义模型):** 严格按照 LeNet-5 结构定义 `nn.Sequential` 模型：
            *   `nn.Conv2d` (in=1, out=6, kernel=5, padding=2) -> `nn.Sigmoid`/`nn.ReLU` -> `nn.AvgPool2d` (kernel=2, stride=2)
            *   `nn.Conv2d` (in=6, out=16, kernel=5) -> `nn.Sigmoid`/`nn.ReLU` -> `nn.AvgPool2d` (kernel=2, stride=2)
            *   `nn.Flatten()`
            *   `nn.Linear` (in=16*5*5, out=120) -> `nn.Sigmoid`/`nn.ReLU`
            *   `nn.Linear` (in=120, out=84) -> `nn.Sigmoid`/`nn.ReLU`
            *   `nn.Linear` (in=84, out=10)
        *   **步骤 c (调整以适应输入):** 注意 Fashion-MNIST 是 28x28，LeNet 原设计针对 32x32，可能需要调整或确认每层输出尺寸。D2L 的实现通常已做适配。
        *   **步骤 d (训练准备):** 定义损失 `nn.CrossEntropyLoss()` 和优化器 `torch.optim.Adam` 或 `SGD`。
        *   **步骤 e (GPU 训练，如果可用):** 将模型和数据移动到 GPU (`model.to(device)`, `data.to(device)`)。
        *   **步骤 f (训练循环):** 标准训练循环，记录训练和测试准确率。
*   **预期结果/验证方法:** 训练损失下降，在 Fashion-MNIST 上的准确率应高于 MLP。理解卷积和池化层如何改变数据形状。
*   **关键注意点:** 卷积核大小、填充、步幅对输出尺寸的影响。通道数的变化。池化层的作用。模型定义时确保各层维度匹配。

---

**实验 7: Batch Normalization & ResNet (对应 D2L Ch 7)**

*   **核心目标:** 理解 Batch Normalization (BN) 的原理和作用（加速收敛、提高泛化），掌握 ResNet 的残差块设计思想和实现。
*   **所需知识/库:** 实验 6, PyTorch (`nn.BatchNorm2d`, `nn.ReLU`)。
*   **详细步骤与思路:**
    1.  **Batch Normalization (BN) 应用:**
        *   **步骤 a:** 在 LeNet 或其他 CNN 模型的卷积层 (`nn.Conv2d`) 之后、激活函数 (`nn.ReLU`) 之前插入 `nn.BatchNorm2d` 层。
        *   **步骤 b:** 重新训练模型，观察训练速度（损失下降速度）和最终准确率是否有所改善。
        *   **步骤 c (理解):** 阅读 BN 相关内容，理解它在训练和测试时的行为差异（移动平均统计量的使用）。
        *   **思路:** 通过实验验证 BN 的效果。
    2.  **Residual Block (ResNet 核心块) 实现:**
        *   **步骤 a (基础块):** 定义一个 `Residual` 类 (继承 `nn.Module`)：
            *   `__init__`: 定义两个 `nn.Conv2d` 层 (通常 3x3)，每个后面跟一个 `nn.BatchNorm2d` 和 `nn.ReLU`。如果输入输出通道数或尺寸不同，额外定义一个 1x1 的 `nn.Conv2d` (可能带 BN) 用于 identity mapping (`self.conv3`)。
            *   `forward`: `Y = F(X)` (通过两个卷积层)，如果需要 `X = self.conv3(X)`，最后 `return nn.ReLU(X + Y)`。
        *   **步骤 b (测试块):** 创建一个输入 Tensor，传入 Residual 块实例，检查输出形状是否符合预期。
        *   **思路:** 实现 ResNet 的核心组件——能够让梯度直接流过的残差连接。
    3.  **构建 ResNet 模型:**
        *   **步骤 a (整体结构):**
            *   起始部分：一个较大的 `nn.Conv2d` + `nn.BatchNorm2d` + `nn.ReLU` + `nn.MaxPool2d`。
            *   主体部分：堆叠多个 Residual Block。通常按阶段划分，每个阶段开始时可能通过 stride=2 的卷积或池化减小尺寸、增加通道数。
            *   结束部分：全局平均池化 (`nn.AdaptiveAvgPool2d((1, 1))`) + `nn.Flatten()` + `nn.Linear` 输出层。
        *   **步骤 b (实现 ResNet-18):** 按照 ResNet-18 的结构（可在 D2L 书中或论文中找到）堆叠 Residual 块。
        *   **步骤 c (训练):** 在图像数据集 (如 CIFAR-10 或 ImageNet 子集) 上训练 ResNet 模型。
*   **预期结果/验证方法:** 加入 BN 后模型训练通常更快、效果更好。ResNet 可以有效训练更深的网络，并在标准数据集上取得高准确率。
*   **关键注意点:** BN 层的位置。Residual Block 中 identity mapping 的维度匹配（通道数和 H/W）。ResNet 不同层数变体的结构细节。

---

**实验 8: 微调 (Fine-tuning) 预训练模型 (对应 D2L Ch 11)**

*   **核心目标:** 掌握利用在大数据集上预训练好的模型（如 ImageNet 上的 ResNet 或 NLP 领域的 BERT）来提升在特定小数据集上任务性能的常用且高效的技术。
*   **所需知识/库:** 实验 1-7 (特别是 CNN/ResNet 或 Transformer 相关知识), PyTorch (`torchvision.models`, `transformers` 库), 特定任务的数据加载和处理知识。
*   **详细步骤与思路:**
    1.  **加载预训练模型:**
        *   **步骤 (CV 示例):** 使用 `torchvision.models` 加载一个预训练的 CNN 模型，例如 `model = torchvision.models.resnet18(pretrained=True)`。
        *   **步骤 (NLP 示例):** 使用 `transformers` 库加载一个预训练的 Transformer 模型，例如 `model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')`。
        *   **思路:** 获取已经学习了通用特征（图像的纹理、边缘；语言的语法、语义）的模型权重。
    2.  **修改模型以适应新任务:**
        *   **步骤 (CV):** 预训练模型的最后一层通常是针对原任务（如 ImageNet 1000 类分类）的全连接层。需要将其替换为适合新任务输出维度（例如，新任务是 10 类分类）的新全连接层。
            *   获取原分类器的输入特征数：`num_ftrs = model.fc.in_features` (对于 ResNet)。
            *   替换分类头：`model.fc = nn.Linear(num_ftrs, num_classes_new_task)`。
        *   **步骤 (NLP):** 对于 `transformers` 库中的 `ForSequenceClassification` 类模型，通常在加载时指定 `num_labels` 即可自动处理分类头。如果需要更复杂的修改（如添加自定义层），则需要访问模型的内部结构。
        *   **思路:** 重用模型的主体结构（特征提取器），仅替换或修改与特定任务相关的输出层。
    3.  **(可选) 冻结部分层参数:**
        *   **步骤:** 为了防止预训练权重在小数据集上被破坏，或为了加速训练，可以选择冻结模型的部分或大部分层（通常是靠近输入的层）。
            *   `for param in model.parameters(): param.requires_grad = False` (冻结所有)
            *   然后解冻需要训练的层，例如新添加的分类头：`for param in model.fc.parameters(): param.requires_grad = True`。
        *   **思路:** 只更新与新任务最相关的少数几层参数，保留通用的预训练知识。适合目标任务数据量较少的情况。如果数据量充足，可以解冻更多层甚至整个模型进行训练（通常使用较小的学习率）。
    4.  **准备数据:**
        *   **步骤:** 加载并预处理你的特定任务数据集。关键在于**预处理方式应尽可能与预训练模型所使用的方式保持一致**（例如，图像归一化的均值和标准差，文本的 Tokenization 方式）。查阅模型文档或来源了解预处理细节。
        *   **思路:** 确保输入数据符合预训练模型的预期格式和范围。
    5.  **训练 (微调):**
        *   **步骤:** 定义损失函数（通常是交叉熵）和优化器。将需要更新的参数传递给优化器（如果冻结了部分层，只传入 `requires_grad=True` 的参数）。
            *   `optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)`。
        *   **步骤:** 使用**较小的学习率**开始训练。因为预训练模型已经处于一个较好的状态，过大的学习率容易破坏预训练权重。可以尝试不同的学习率策略（如学习率衰减）。
        *   **步骤:** 运行标准的训练循环。
    *   **预期结果/验证方法:** 微调后的模型在目标任务上的性能通常远超从零开始训练的模型，尤其是在目标数据集较小的情况下。收敛速度通常也更快。对比微调模型和从零训练模型的测试准确率/损失。
    *   **关键注意点:** 选择合适的预训练模型（源任务与目标任务的相关性）。数据预处理与预训练模型保持一致。学习率的选择（通常需要更小）。理解何时冻结层以及冻结哪些层。

---

**实验 9: RNN/LSTM/GRU (从零理解与框架实现) (对应 D2L Ch 8, 9)**

*   **核心目标:** 理解循环神经网络处理序列数据的原理（隐藏状态传递），掌握 LSTM 和 GRU 门控机制的作用，并能使用框架实现序列模型。
*   **所需知识/库:** 实验 1-4, PyTorch (`nn.RNN`, `nn.LSTM`, `nn.GRU`, `nn.Embedding`)。
*   **详细步骤与思路:**
    1.  **文本预处理:**
        *   **步骤 a:** 加载文本数据集（如 D2L 提供的时光机器文本）。
        *   **步骤 b:** 进行词元化 (Tokenization)，将文本切分成单词或字符。
        *   **步骤 c:** 构建词表 (Vocabulary)，将每个词元映射到一个唯一的整数索引。
        *   **步骤 d:** 将文本数据转换为索引序列。
        *   **思路:** 将原始文本转换为模型可以处理的数值形式。
    2.  **简单 RNN 从零实现 (理解性):**
        *   **步骤 a:** 定义模型参数 (`W_xh`, `W_hh`, `b_h`, `W_hq`, `b_q`)。
        *   **步骤 b:** 实现前向传播逻辑：接收输入序列 `X` 和初始隐藏状态 `H_0`，循环遍历时间步 `t`，计算 `H_t` 和输出 `O_t`。
        *   **步骤 c:** 理解 BPTT (Backpropagation Through Time) 的概念（梯度如何沿时间步反向传播）。（框架会自动处理）
        *   **思路:** 深入理解 RNN 的状态更新机制。
    3.  **使用框架 API 实现 RNN/LSTM/GRU (语言模型任务):**
        *   **步骤 a (数据加载):** 将索引序列构造成小批量（通常形状为 `(批量大小, 时间步数)` 或 `(时间步数, 批量大小)`)。
        *   **步骤 b (模型定义):**
            *   `nn.Embedding` 层: 将词元索引映射为向量。
            *   `nn.LSTM` (或 `nn.RNN`, `nn.GRU`) 层: 处理嵌入向量序列。注意设置 `input_size`, `hidden_size`, `num_layers`。其输出包括 `output` 序列和最终的 `hidden_state` (及 `cell_state` for LSTM)。
            *   `nn.Linear` 层: 将 LSTM 的输出映射到词表大小，得到预测每个词的 logits。
        *   **步骤 c (训练准备):** 使用交叉熵损失。选择优化器。
        *   **步骤 d (初始化隐藏状态):** 在每个 epoch 开始或处理新序列前，初始化隐藏状态 `H_0` (和 `C_0` for LSTM) 为零 Tensor。
        *   **步骤 e (训练循环):**
            *   处理一个批量的序列数据。
            *   将隐藏状态从上一个批量 detach (`state.detach()`) 以阻止梯度无限传播。
            *   模型前向传播，得到输出 logits 和新的隐藏状态。
            *   计算损失（通常预测下一个词，需要对齐输入输出）。
            *   反向传播，更新参数。
            *   (可选) 使用梯度裁剪 (`nn.utils.clip_grad_norm_`) 防止梯度爆炸。
        *   **步骤 f (评估):** 使用困惑度 (Perplexity) 评估语言模型效果。
*   **预期结果/验证方法:** 从零实现的 RNN 状态更新符合预期。框架实现的语言模型损失（或困惑度）下降。LSTM/GRU 通常比简单 RNN 效果更好，能处理更长的依赖。
*   **关键注意点:** 序列数据的 batching 方式。隐藏状态的初始化和传递（跨 batch）。梯度裁剪的使用。理解 LSTM/GRU 的门控机制如何缓解梯度消失/爆炸。

---

**实验 10: Attention 机制 & Transformer 核心 (对应 D2L Ch 10, 11)**

*   **核心目标:** 理解注意力机制的计算过程（特别是 Scaled Dot-Product Attention），掌握 Multi-Head Attention 和 Transformer Encoder/Decoder 块的核心组件。
*   **所需知识/库:** 实验 1-7, 9, PyTorch (`nn.Linear`, `nn.Softmax`, `nn.LayerNorm`, `nn.Dropout`)。
*   **详细步骤与思路:**
    1.  **Scaled Dot-Product Attention 实现:**
        *   **步骤 a:** 定义函数，输入为 `queries` (Q), `keys` (K), `values` (V) 和可选的 `mask`。
        *   **步骤 b:** 计算 `scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)` (d_k 是 key 的维度)。
        *   **步骤 c:** 如果提供了 `mask`，将 mask 中为 True (或 1) 的位置对应的 `scores` 设置为一个非常小的负数 (如 -1e9)，以使其在 Softmax 后接近 0。
        *   **步骤 d:** 计算 `attention_weights = nn.functional.softmax(scores, dim=-1)`。
        *   **步骤 e:** 计算输出 `output = torch.matmul(attention_weights, V)`。
        *   **步骤 f:** 使用简单的 Q, K, V Tensor 测试函数，检查输出形状和注意力权重。
        *   **思路:** 实现注意力机制的核心计算步骤。
    2.  **Multi-Head Attention 实现:**
        *   **步骤 a:** 定义一个 `MultiHeadAttention` 类 (继承 `nn.Module`)。
        *   **步骤 b (`__init__`):** 定义 W_q, W_k, W_v 线性层用于投影输入 Q, K, V 到多头；定义最终的输出线性层 W_o。记录 `num_heads`, `d_k`, `d_v`。
        *   **步骤 c (`forward`):**
            *   将输入的 Q, K, V 分别通过 W_q, W_k, W_v 线性层。
            *   将投影后的 Q, K, V 张量 reshape 并 transpose，使其形状变为 `(batch_size, num_heads, seq_len, d_k/d_v)` 以便并行计算多头。
            *   调用前面实现的 Scaled Dot-Product Attention 函数（传入 reshape 后的 Q, K, V 和 mask）。
            *   将多头输出的 attention 结果重新 transpose 和 reshape，合并回 `(batch_size, seq_len, d_model)`。
            *   通过最终的 W_o 线性层。
        *   **步骤 d (测试):** 使用合适的输入 Tensor 测试该模块。
        *   **思路:** 将多个并行的 Scaled Dot-Product Attention 组合起来。
    3.  **Transformer Block 组件实现:**
        *   **步骤 a (Add & Norm):** 实现一个函数或层，输入 `X` 和一个子层 `sublayer`，计算 `LayerNorm(X + sublayer(X))`。需要用到 `nn.LayerNorm`。
        *   **步骤 b (Position-wise Feed-Forward Network):** 定义一个类，包含两个线性层 (`Linear -> ReLU -> Dropout -> Linear`)。
        *   **步骤 c (组合 Encoder Block):** 组合 Multi-Head Attention (自注意力), Add & Norm, Feed-Forward Network, Add & Norm。
        *   **步骤 d (组合 Decoder Block):** 组合 Masked Multi-Head Attention (自注意力), Add & Norm, Multi-Head Attention (Encoder-Decoder 注意力), Add & Norm, Feed-Forward Network, Add & Norm。
        *   **思路:** 将 Transformer 的标准构建块实现出来。
*   **预期结果/验证方法:** 各个组件的输出形状符合预期。理解 Attention 权重如何表示相关性。理解 Transformer 块的内部数据流。
*   **关键注意点:** 维度匹配（`d_model`, `num_heads`, `d_k`, `d_v`）。Attention Mask 的正确使用（padding mask, look-ahead mask）。Layer Normalization 的位置。

---

**实验 11: Adam 优化器 (对应 D2L Ch 12)**

*   **核心目标:** 理解 Adam 优化算法结合了 Momentum 和 RMSProp 的思想，并能从零实现其更新规则。
*   **所需知识/库:** 实验 1-2, PyTorch。
*   **详细步骤与思路:**
    1.  **Adam 更新规则理解:**
        *   **步骤:** 回顾 Adam 的公式，理解一阶矩估计 `m` (动量) 和二阶矩估计 `v` (学习率缩放) 的计算，以及偏差修正步骤。
    2.  **从零实现 Adam:**
        *   **步骤 a:** 定义一个函数 `adam_update(params, states, hyperparams)`。`params` 是模型参数列表，`states` 用于存储每个参数对应的 `m` 和 `v` 及时间步 `t`，`hyperparams` 包含 `lr`, `beta1`, `beta2`, `epsilon`。
        *   **步骤 b:** 在函数内部，遍历 `params`：
            *   获取对应参数的 `state`（如果不存在则初始化 `m=0, v=0, t=0`）。
            *   `state['t'] += 1`。
            *   获取当前参数的梯度 `grad = param.grad`。
            *   更新 `m`: `state['m'] = hyperparams['beta1'] * state['m'] + (1 - hyperparams['beta1']) * grad`。
            *   更新 `v`: `state['v'] = hyperparams['beta2'] * state['v'] + (1 - hyperparams['beta2']) * grad**2`。
            *   计算偏差修正后的矩: `m_hat = state['m'] / (1 - hyperparams['beta1']**state['t'])`, `v_hat = state['v'] / (1 - hyperparams['beta2']**state['t'])`。
            *   更新参数: `param.data -= hyperparams['lr'] * m_hat / (torch.sqrt(v_hat) + hyperparams['epsilon'])`。
    3.  **测试与对比:**
        *   **步骤 a:** 使用一个简单的优化问题（如线性回归或优化一个简单函数）和手动实现的 Adam 进行训练。
        *   **步骤 b:** 使用相同问题和 PyTorch 自带的 `torch.optim.Adam` 进行训练。
        *   **步骤 c:** 对比两种方式下参数收敛速度和最终结果。
*   **预期结果/验证方法:** 手动实现的 Adam 能够使损失下降并找到最优解。其表现（如收敛速度）应与框架自带的 Adam 相似（给定相同超参数）。
*   **关键注意点:** `states` 的正确管理（每个参数对应独立的 `m, v, t`）。超参数的选择。`epsilon` 的作用（防止除零）。
