from typing import List

# class Solution:
#     def plusOne(self, digits: List[int]) -> List[int]:
#         carry = 1  # 初始进位为1，因为要加一
#         # 从后往前遍历每一位
#         for i in range(len(digits)-1, -1, -1):
#             digits[i] += carry  # 当前位加上进位
#             carry = digits[i] // 10  # 计算新的进位
#             digits[i] %= 10         # 更新当前位的值
#             if carry == 0:          # 如果进位为0，提前结束循环
#                 break
#         # 处理最高位的进位
#         if carry == 1:
#             digits.insert(0, 1)
#         return digits

# # 调用示例
# solution = Solution()
# print(solution.plusOne([1,2,3]))   # 输出: [1,2,4]
# print(solution.plusOne([4,3,2,1])) # 输出: [4,3,2,2]
# print(solution.plusOne([9]))       # 输出: [1,0]
