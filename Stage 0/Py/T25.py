from typing import List


class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        for i in range (len(nums)-1,1,-1):
            if nums[i-2] + nums[i-1] > nums[i]:
                return nums[i]+nums[i-1]+nums[i-2]

        return 0
            
# 示例 1
nums1 = [2, 1, 2]
solution = Solution()
result1 = solution.largestPerimeter(nums1)
print(f"输入: {nums1}, 输出: {result1}")  # 输出: 输入: [2, 1, 2], 输出: 5

# 示例 2
nums2 = [1, 2, 1, 10]
result2 = solution.largestPerimeter(nums2)
print(f"输入: {nums2}, 输出: {result2}")  # 输出: 输入: [1, 2, 1, 10], 输出: 0

# 更多测试用例
nums3 = [3, 2, 3, 4]
result3 = solution.largestPerimeter(nums3)
print(f"输入: {nums3}, 输出: {result3}")  # 输出: 输入: [3, 2, 3, 4], 输出: 10 (3+3+4)

nums4 = [1, 1, 1]
result4 = solution.largestPerimeter(nums4)
print(f"输入: {nums4}, 输出: {result4}")  # 输出: 输入: [1, 1, 1], 输出: 3

nums5 = [4, 5, 1, 2]
result5 = solution.largestPerimeter(nums5)
print(f"输入: {nums5}, 输出: {result5}")  # 输出: 输入: [4, 5, 1, 2], 输出:  2+4+5=11 
