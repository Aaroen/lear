from typing import List

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums) # 1. 转换为集合，去重并方便查找
        max_length = 0     # 2. 初始化最长长度

        for num in num_set:  # 3. 遍历集合中的每个数字
            if num - 1 not in num_set: # 4. 检查是否为序列的起点
                current_num = num
                current_length = 1

                while current_num + 1 in num_set: # 5. 扩展序列
                    current_num += 1
                    current_length += 1

                max_length = max(max_length, current_length) # 6. 更新最长长度

        return max_length # 7. 返回最长长度


solution = Solution()

# 示例 1
input1 = [100, 4, 200, 1, 3, 2]
output1 = solution.longestConsecutive(input1)
print(f"输入: {input1}, 输出: {output1}, 是否正确: {output1 == 4}")  # 输出: 输入: [100, 4, 200, 1, 3, 2], 输出: 4, 是否正确: True

# 示例 2
input2 = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
output2 = solution.longestConsecutive(input2)
print(f"输入: {input2}, 输出: {output2}, 是否正确: {output2 == 9}")  # 输出: 输入: [0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 输出: 9, 是否正确: True

# 示例 3
input3 = [1, 0, 1, 2]
output3 = solution.longestConsecutive(input3)
print(f"输入: {input3}, 输出: {output3}, 是否正确: {output3 == 3}")  # 输出: 输入: [1, 0, 1, 2], 输出: 3, 是否正确: True

# 空数组示例
input4 = []
output4 = solution.longestConsecutive(input4)
print(f"输入: {input4}, 输出: {output4}, 是否正确: {output4 == 0}") # 输出: 输入: [], 输出: 0, 是否正确: True
