from typing import List
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        num_map = {}  # 创建一个哈希表 (字典) 用于存储数字和索引

        for index, num in enumerate(nums): # 遍历数组，同时获取索引和数值
            if num in num_map: # 检查当前数字是否在哈希表中
                if index - num_map[num] <= k: # 如果存在，检查索引差是否小于等于 k
                    return True # 如果小于等于 k，找到符合条件的重复数字，返回 True
                else:
                    num_map[num] = index # 如果索引差大于 k，更新哈希表中该数字的索引为最新的索引
            else:
                num_map[num] = index # 如果数字不在哈希表中，将数字和索引存入哈希表

        return False # 遍历完数组后，没有找到符合条件的重复数字，返回 False


nums1 = [1, 2, 3, 1]
k1 = 3
solution = Solution()
result1 = solution.containsNearbyDuplicate(nums1, k1)
print(f"输入: nums = {nums1}, k = {k1}, 输出: {result1}") # 输出: 输入: nums = [1, 2, 3, 1], k = 3, 输出: True
print(result1 == True) # 输出: True (代码编写正确)
nums2 = [1, 0, 1, 1]
k2 = 1
result2 = solution.containsNearbyDuplicate(nums2, k2)
print(f"输入: nums = {nums2}, k = {k2}, 输出: {result2}") # 输出: 输入: nums = [1, 0, 1, 1], k = 1, 输出: True
print(result2 == True) # 输出: True (代码编写正确)
nums3 = [1, 2, 3, 1, 2, 3]
k3 = 2
result3 = solution.containsNearbyDuplicate(nums3, k3)
print(f"输入: nums = {nums3}, k = {k3}, 输出: {result3}") # 输出: 输入: nums = [1, 2, 3, 1, 2, 3], k = 2, 输出: False
print(result3 == False) # 输出: True (代码编写正确)
