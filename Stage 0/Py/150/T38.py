from typing import List
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_dict = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_dict:
                return [num_dict[complement], i]
            num_dict[num] = i
solution = Solution()

# 示例 1
nums1 = [2, 7, 11, 15]
target1 = 9
result1 = solution.twoSum(nums1, target1)
print(f"nums: {nums1}, target: {target1}, result: {result1}") # 输出: nums: [2, 7, 11, 15], target: 9, result: [0, 1]
print(f"示例 1 判断结果: {result1 == [0, 1]}") # 示例 1 判断结果: True

# 示例 2
nums2 = [3, 2, 4]
target2 = 6
result2 = solution.twoSum(nums2, target2)
print(f"nums: {nums2}, target: {target2}, result: {result2}") # 输出: nums: [3, 2, 4], target: 6, result: [1, 2]
print(f"示例 2 判断结果: {result2 == [1, 2]}") # 示例 2 判断结果: True