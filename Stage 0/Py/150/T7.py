from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        计算股票交易的最大利润。

        Args:
            prices: 一个列表，表示每天的股票价格。

        Returns:
            int: 可以获得的最大利润。如果不能获取任何利润，则返回 0 。
        """
        if not prices or len(prices) < 2: # 如果价格列表为空或者只有一天，无法进行交易，直接返回0
            return 0

        min_price = prices[0] # 初始化最低买入价格为第一天的价格
        max_profit = 0      # 初始化最大利润为0

        for price in prices[1:]: # 从第二天开始遍历价格列表
            min_price = min(min_price, price) # 更新最低买入价格，保持min_price为遍历到当前天为止的最低价格
            profit = price - min_price      # 计算如果今天卖出股票的利润 (今天的价格 - 最低买入价格)
            max_profit = max(max_profit, profit) # 更新最大利润，取当前最大利润和今天利润的较大值

        return max_profit # 返回计算得到的最大利润

# 调用示例
solution = Solution()
print(solution.maxProfit([7, 1, 5, 3, 6, 4]))  # 输出5
print(solution.maxProfit([7, 6, 4, 3, 1]))     # 输出0

# 示例代码调用
if __name__ == '__main__':
    solution = Solution()

    # 示例 1
    prices1 = [7, 1, 5, 3, 6, 4]
    profit1 = solution.maxProfit(prices1)
    print(f"输入: {prices1}, 输出: {profit1} (预期: 5)") # 输出: 输入: [7, 1, 5, 3, 6, 4], 输出: 5 (预期: 5)

    # 示例 2
    prices2 = [7, 6, 4, 3, 1]
    profit2 = solution.maxProfit(prices2)
    print(f"输入: {prices2}, 输出: {profit2} (预期: 0)") # 输出: 输入: [7, 6, 4, 3, 1], 输出: 0 (预期: 0)

    # 示例 3: 单天价格，无法交易
    prices3 = [5]
    profit3 = solution.maxProfit(prices3)
    print(f"输入: {prices3}, 输出: {profit3} (预期: 0)") # 输出: 输入: [5], 输出: 0 (预期: 0)

    # 示例 4: 两天价格，下跌
    prices4 = [5, 2]
    profit4 = solution.maxProfit(prices4)
    print(f"输入: {prices4}, 输出: {profit4} (预期: 0)") # 输出: 输入: [5, 2], 输出: 0 (预期: 0)

    # 示例 5: 两天价格，上涨
    prices5 = [2, 5]
    profit5 = solution.maxProfit(prices5)
    print(f"输入: {prices5}, 输出: {profit5} (预期: 3)") # 输出: 输入: [2, 5], 输出: 3 (预期: 3)
