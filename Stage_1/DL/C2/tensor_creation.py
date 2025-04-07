"""
Tensor创建 - 实验1.1
演示PyTorch中Tensor的基本创建方法和属性
"""
import torch
import sys
import os

# 添加当前目录到系统路径，以便其他模块可以导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_tensors():
    """创建不同类型和形状的Tensor"""
    print("1. 从Python列表创建Tensor:")
    x1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"x1 = {x1}")
    print(f"形状: {x1.shape}, 数据类型: {x1.dtype}")
    
    print("\n2. 创建指定形状的零Tensor:")
    x2 = torch.zeros((2, 3, 4))
    print(f"x2 = {x2}")
    print(f"形状: {x2.shape}, 数据类型: {x2.dtype}")
    
    print("\n3. 创建指定形状的一Tensor:")
    x3 = torch.ones((2, 3))
    print(f"x3 = {x3}")
    print(f"形状: {x3.shape}, 数据类型: {x3.dtype}")
    
    print("\n4. 创建随机Tensor:")
    x4 = torch.randn(3, 4)  # 标准正态分布
    print(f"x4 = {x4}")
    print(f"形状: {x4.shape}, 数据类型: {x4.dtype}")
    
    print("\n5. 创建等差数列Tensor:")
    x5 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    print(f"x5 = {x5}")
    print(f"形状: {x5.shape}, 数据类型: {x5.dtype}")
    
    print("\n6. 创建指定数据类型的Tensor:")
    x6 = torch.tensor([1.2, 3.4], dtype=torch.float64)
    print(f"x6 = {x6}")
    print(f"形状: {x6.shape}, 数据类型: {x6.dtype}")
    
    print("\n7. 创建与现有Tensor相同形状的Tensor:")
    x7 = torch.zeros_like(x5)
    print(f"x7 = {x7}")
    print(f"形状: {x7.shape}, 数据类型: {x7.dtype}")
    
    return x5  # 返回一个Tensor供其他模块使用

if __name__ == "__main__":
    print("===== Tensor创建演示 =====")
    create_tensors()
    
    # 设备检查
    print("\n===== 设备检查 =====")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"是否可用CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")