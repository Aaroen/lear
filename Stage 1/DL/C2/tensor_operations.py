"""
Tensor运算 - 实验1.3
演示PyTorch中Tensor的各种运算操作
"""
import torch
import sys
import os

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入tensor_creation模块
from tensor_creation import create_tensors

def demonstrate_operations(x=None):
    """演示Tensor的各种运算操作"""
    if x is None:
        # 如果没有提供Tensor，则创建一个
        x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
        print("创建的Tensor x:")
        print(x)
    
    # 创建另一个Tensor用于演示
    y = torch.ones_like(x) * 2
    print("\n创建的Tensor y:")
    print(y)
    
    print("\n1. 基本算术运算:")
    print(f"x + y = \n{x + y}")
    print(f"x - y = \n{x - y}")
    print(f"x * y = \n{x * y}")  # 逐元素乘法
    print(f"x / y = \n{x / y}")
    print(f"x ** 2 = \n{x ** 2}")  # 平方
    
    print("\n2. 矩阵运算:")
    # 创建适合矩阵乘法的Tensor
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    print(f"a = \n{a}")
    print(f"b = \n{b}")
    print(f"矩阵乘法 a @ b = \n{a @ b}")  # 使用@运算符
    print(f"矩阵乘法 torch.matmul(a, b) = \n{torch.matmul(a, b)}")  # 使用matmul函数
    
    print("\n3. 聚合运算:")
    print(f"x的和: {x.sum()}")
    print(f"x沿行的和: {x.sum(axis=0)}")
    print(f"x沿列的和: {x.sum(axis=1)}")
    print(f"x的均值: {x.mean()}")
    print(f"x的最大值: {x.max()}")
    print(f"x的最小值: {x.min()}")
    print(f"x的L2范数: {x.norm()}")
    
    print("\n4. 数学函数:")
    print(f"exp(x) = \n{torch.exp(x)}")
    print(f"log(x+1) = \n{torch.log(x+1)}")  # 加1避免log(0)
    print(f"sin(x) = \n{torch.sin(x)}")
    
    print("\n5. 连接操作:")
    print(f"沿行连接 [x, y] = \n{torch.cat([x, y], dim=0)}")
    print(f"沿列连接 [x, y] = \n{torch.cat([x, y], dim=1)}")
    
    print("\n6. 形状变换:")
    print(f"x的转置 = \n{x.T}")
    print(f"x重塑为(4, 3) = \n{x.reshape(4, 3)}")
    print(f"x展平 = {x.flatten()}")
    
    return x, y

if __name__ == "__main__":
    print("===== Tensor运算演示 =====")
    # 从tensor_creation模块获取Tensor
    x = create_tensors()
    print("\n使用从tensor_creation获取的Tensor:")
    demonstrate_operations(x)