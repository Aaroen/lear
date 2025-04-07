"""
Tensor索引与切片 - 实验1.2
演示PyTorch中Tensor的索引和切片操作
"""
import torch
import sys
import os

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入tensor_creation模块
from tensor_creation import create_tensors

def demonstrate_indexing(x=None):
    """演示Tensor的索引和切片操作"""
    if x is None:
        # 如果没有提供Tensor，则创建一个
        x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
        print("创建的Tensor:")
        print(x)
    
    print("\n1. 访问单个元素:")
    print(f"x[1, 2] = {x[1, 2]}")  # 第2行第3列的元素
    
    print("\n2. 访问整行:")
    print(f"x[1] = {x[1]}")  # 第2行
    
    print("\n3. 访问整列:")
    print(f"x[:, 1] = {x[:, 1]}")  # 第2列
    
    print("\n4. 切片操作 - 前两行:")
    print(f"x[0:2, :] = \n{x[0:2, :]}")
    
    print("\n5. 切片操作 - 后两列:")
    print(f"x[:, 2:4] = \n{x[:, 2:4]}")
    
    print("\n6. 负索引 - 最后一行:")
    print(f"x[-1] = {x[-1]}")
    
    print("\n7. 步长切片 - 隔列取:")
    print(f"x[:, ::2] = \n{x[:, ::2]}")
    
    print("\n8. 复杂切片 - 第1行到第2行，第2列到第3列:")
    print(f"x[0:2, 1:3] = \n{x[0:2, 1:3]}")
    
    return x

if __name__ == "__main__":
    print("===== Tensor索引与切片演示 =====")
    # 从tensor_creation模块获取Tensor
    x = create_tensors()
    print("\n使用从tensor_creation获取的Tensor:")
    demonstrate_indexing(x)