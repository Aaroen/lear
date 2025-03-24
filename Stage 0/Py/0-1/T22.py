class Solution(object):
    def countOdds(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: int
        """
        # 计算high到0之间奇数的个数，减去low到0之间的奇数个数，得到low到high之间的奇数数目
        return (high + 1) // 2 - low // 2

# 调用示例
solution = Solution()
print(solution.countOdds(3, 7))   # 输出3
print(solution.countOdds(8, 10))  # 输出1
