from typing import List

class Solution:
    def jump(self, nums: List[int]) -> int:
        jump_nums = 0
        n = len(nums)
        if n == 1:
            return 0  # 只有一个元素, 无需跳跃

        current_reach = 0 # 当前跳跃次数下能到达的最远距离
        max_reach = 0     # 下一次跳跃能到达的最远距离

        for i in range(n - 1):
            max_reach = max(max_reach, i + nums[i]) # 更新 max_reach：从当前可达范围内的所有位置出发，下一步能到达的最远距离

            if i == current_reach: # 到达当前可达距离的边界，需要跳跃
                jump_nums += 1
                current_reach = max_reach # 更新 current_reach 为新的 max_reach

        return jump_nums

# 示例 1
nums1 = [2, 3, 1, 1, 4]
solution = Solution()
result1 = solution.jump(nums1)
print(f"输入: nums = {nums1}")
print(f"输出: {result1}")  # 输出: 2

# 示例 2
nums2 = [2, 3, 0, 1, 4]
solution = Solution()
result2 = solution.jump(nums2)
print(f"输入: nums = {nums2}")
print(f"输出: {result2}")  # 输出: 2

# 示例 3 (测试只有一个元素的情况)
nums3 = [0]
solution = Solution()
result3 = solution.jump(nums3)
print(f"输入: nums = {nums3}")
print(f"输出: {result3}")  # 输出: 0

# 示例 4 (测试第一个元素可以直接到达末尾的情况)
nums4 = [5, 1, 1, 1, 1]
solution = Solution()
result4 = solution.jump(nums4)
print(f"输入: nums = {nums4}")
print(f"输出: {result4}")  # 输出: 1
