from typing import List

# class Solution:
#     def arraySign(self, nums: List[int]) -> int:
#         count_neg = 0
#         for num in nums:
#             if num == 0:
#                 return 0
#             if num !=   0:
#                 count_neg += 1
#         return 1 if count_neg % 2 == 0 else -1


# # 创建 Solution 类的实例
# solution = Solution()

# # 示例 1
# nums1 = [-1, -2, -3, -4, 3, 2, 1]
# result1 = solution.arraySign(nums1)
# print(f"输入数组: {nums1}, 符号函数值: {result1}") # 输出: 输入数组: [-1, -2, -3, -4, 3, 2, 1], 符号函数值: 1
# # 解释:
# # nums1 的乘积是 (-1) * (-2) * (-3) * (-4) * 3 * 2 * 1 = 144
# # signFunc(144) = 1

# # 示例 2
# nums2 = [1, 5, 0, 2, -3]
# result2 = solution.arraySign(nums2)
# print(f"输入数组: {nums2}, 符号函数值: {result2}") # 输出: 输入数组: [1, 5, 0, 2, -3], 符号函数值: 0
# # 解释:
# # nums2 的乘积是 1 * 5 * 0 * 2 * (-3) = 0
# # signFunc(0) = 0

# # 示例 3
# nums3 = [-1, 1, -1, 1, -1]
# result3 = solution.arraySign(nums3)
# print(f"输入数组: {nums3}, 符号函数值: {result3}") # 输出: 输入数组: [-1, 1, -1, 1, -1], 符号函数值: -1
# # 解释:
# # nums3 的乘积是 (-1) * 1 * (-1) * 1 * (-1) = -1
# # signFunc(-1) = -1












class Solution:
    def arraySign(self, nums: List[int]) -> int:
        product = 1
        for num in nums:
            product *= num
        if product == 0:
            return 0 
        elif product > 0:
            return 1
        else:
            return -1
        
# 创建 Solution 类的实例
solution = Solution()

# 示例 1
nums1 = [-1, -2, -3, -4, 3, 2, 1]
result1 = solution.arraySign(nums1)
print(f"输入数组: {nums1}, 符号函数值: {result1}") # 输出: 输入数组: [-1, -2, -3, -4, 3, 2, 1], 符号函数值: 1
# 解释:
# nums1 的乘积是 (-1) * (-2) * (-3) * (-4) * 3 * 2 * 1 = 144
# signFunc(144) = 1

# 示例 2
nums2 = [1, 5, 0, 2, -3]
result2 = solution.arraySign(nums2)
print(f"输入数组: {nums2}, 符号函数值: {result2}") # 输出: 输入数组: [1, 5, 0, 2, -3], 符号函数值: 0
# 解释:
# nums2 的乘积是 1 * 5 * 0 * 2 * (-3) = 0
# signFunc(0) = 0

# 示例 3
nums3 = [-1, 1, -1, 1, -1]
result3 = solution.arraySign(nums3)
print(f"输入数组: {nums3}, 符号函数值: {result3}") # 输出: 输入数组: [-1, 1, -1, 1, -1], 符号函数值: -1
# 解释:
# nums3 的乘积是 (-1) * 1 * (-1) * 1 * (-1) = -1
# signFunc(-1) = -1
