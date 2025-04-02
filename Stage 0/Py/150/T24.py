from typing import List

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        n = len(numbers)
        index_1 = 0
        index_2 = n - 1
        while index_1 < index_2: # 循环条件：index_1 必须小于 index_2
            current_sum = numbers[index_1] + numbers[index_2]
            if current_sum == target: # 情况 1: 和等于目标值，找到答案
                return [index_1 + 1, index_2 + 1]
            elif current_sum < target: # 情况 2: 和小于目标值，index_1 右移以增大和
                index_1 += 1
            else: # 情况 3: 和大于目标值，index_2 左移以减小和
                index_2 -= 1
        # 理论上不会执行到这里，因为题目保证有唯一解 (为了完整性保留)
        return []


# 示例 1
numbers1 = [2, 7, 11, 15]
target1 = 9
solution = Solution()
result1 = solution.twoSum(numbers1, target1)
print(f"输入: numbers = {numbers1}, target = {target1}")
print(f"输出: {result1}") # 输出: [1, 2]
# 解释:
# - 初始化 left = 0, right = 3
# - numbers[0] + numbers[3] = 2 + 15 = 17 > 9，right 减 1，right = 2
# - numbers[0] + numbers[2] = 2 + 11 = 13 > 9，right 减 1，right = 1
# - numbers[0] + numbers[1] = 2 + 7 = 9 == 9，返回 [0+1, 1+1] = [1, 2]


# 示例 2
numbers2 = [2, 3, 4]
target2 = 6
result2 = solution.twoSum(numbers2, target2)
print(f"输入: numbers = {numbers2}, target = {target2}")
print(f"输出: {result2}") # 输出: [1, 3]
# 解释:
# - 初始化 left = 0, right = 2
# - numbers[0] + numbers[2] = 2 + 4 = 6 == 6，返回 [0+1, 2+1] = [1, 3]


# 示例 3
numbers3 = [-1, 0]
target3 = -1
result3 = solution.twoSum(numbers3, target3)
print(f"输入: numbers = {numbers3}, target = {target3}")
print(f"输出: {result3}") # 输出: [1, 2]
# 解释:
# - 初始化 left = 0, right = 1
# - numbers[0] + numbers[1] = -1 + 0 = -1 == -1，返回 [0+1, 1+1] = [1, 2]

