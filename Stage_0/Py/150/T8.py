# 导入 List 类型用于类型注解
from typing import List

class Solution:
    """
    解决“买卖股票的最佳时机 II”问题的类。
    """
    def maxProfit(self, prices: List[int]) -> int:
        """
        计算给定价格数组下可以获得的最大利润。

        Args:
            prices: 一个整数列表，prices[i] 表示第 i 天的股票价格。

        Returns:
            可以获得的最大利润。
        """
        # 检查边缘情况：如果没有价格数据或只有一天的数据，无法交易，利润为0
        if not prices or len(prices) < 2:
            return 0

        # 初始化最大利润为 0
        max_profit = 0

        # 从第二天开始遍历价格列表 (索引从 1 开始)
        for i in range(1, len(prices)):
            # 检查当天的价格是否高于前一天的价格
            if prices[i] > prices[i-1]:
                # 如果价格上涨，将差价（利润）累加到总利润中
                # 这相当于在前一天买入，当天卖出
                max_profit += prices[i] - prices[i-1]

        # 返回计算出的累计最大利润
        return max_profit

# --- 调用示例 ---

# 创建 Solution 类的实例
solver = Solution()

# 示例 1
prices1 = [7, 1, 5, 3, 6, 4]
profit1 = solver.maxProfit(prices1)
print(f"输入: prices = {prices1}") # 打印输入数组
print(f"输出: {profit1}") # 打印计算出的最大利润 (预期输出: 7)
print("-" * 20) # 分隔线

# 示例 2
prices2 = [1, 2, 3, 4, 5]
profit2 = solver.maxProfit(prices2)
print(f"输入: prices = {prices2}") # 打印输入数组
print(f"输出: {profit2}") # 打印计算出的最大利润 (预期输出: 4)
print("-" * 20) # 分隔线

# 示例 3
prices3 = [7, 6, 4, 3, 1]
profit3 = solver.maxProfit(prices3)
print(f"输入: prices = {prices3}") # 打印输入数组
print(f"输出: {profit3}") # 打印计算出的最大利润 (预期输出: 0)
print("-" * 20) # 分隔线

# 其他测试用例
prices4 = [] # 空列表
profit4 = solver.maxProfit(prices4)
print(f"输入: prices = {prices4}")
print(f"输出: {profit4}") # 预期输出: 0
print("-" * 20)

prices5 = [5] # 只有一个元素
profit5 = solver.maxProfit(prices5)
print(f"输入: prices = {prices5}")
print(f"输出: {profit5}") # 预期输出: 0
print("-" * 20)

prices6 = [3, 3, 5, 0, 0, 3, 1, 4] # 包含平台期和波动
profit6 = solver.maxProfit(prices6)
print(f"输入: prices = {prices6}")
print(f"输出: {profit6}") # 预期输出: (5-3) + (3-0) + (4-1) = 2 + 3 + 3 = 8
print("-" * 20)
