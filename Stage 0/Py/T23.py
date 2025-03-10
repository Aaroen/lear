class Solution(object):
    def average(self, salary):
        """
        :type salary: List[int]
        :rtype: float
        """
        min_val = min(salary)    # 找到数组中的最小值
        max_val = max(salary)    # 找到数组中的最大值
        total = sum(salary)      # 计算数组元素总和
        # 去掉最大和最小值后的总和除以剩余元素个数（长度减2）
        return (total - min_val - max_val) / (len(salary) - 2)


# 创建Solution类的实例
solution = Solution()

# 示例1调用
print(solution.average([4000, 3000, 1000, 2000]))  # 输出：2500.0
# 解释：总和是10000，减去最小1000和最大4000后为5000，除以2得2500.0

# 示例2调用
print(solution.average([1000, 2000, 3000]))        # 输出：2000.0
# 解释：总和是6000，减去最小1000和最大3000后为2000，除以1得2000.0

# 示例3调用
print(solution.average([6000, 5000, 4000, 3000, 2000, 1000]))  # 输出：3500.0
# 解释：总和是21000，减去最小1000和最大6000后为14000，除以4得3500.0

# 示例4调用
print(solution.average([8000, 9000, 2000, 3000, 6000, 1000]))  # 输出：4750.0
# 解释：总和是29000，减去最小1000和最大9000后为19000，除以4得4750.0
