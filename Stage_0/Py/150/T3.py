from typing import List
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        if not nums:
            return 0
        for i in range(1,len(nums)):
            if nums[i] != nums[i - 1]:
                nums[k] = nums[i]
                k +=1
        return k
    
# 调用示例 1
nums1 = [1, 1, 2]
solution = Solution()
k1 = solution.removeDuplicates(nums1)
print(f"示例 1 的新长度: {k1}, 去重后的 nums1: {nums1[:k1]}") # 输出: 示例 1 的新长度: 2, 去重后的 nums1: [1, 2]

# 调用示例 2
nums2 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
k2 = solution.removeDuplicates(nums2)
print(f"示例 2 的新长度: {k2}, 去重后的 nums2: {nums2[:k2]}") # 输出: 示例 2 的新长度: 5, 去重后的 nums2: [0, 1, 2, 3, 4]

# 调用示例 3 (包含重复元素的数组)
nums3 = [1, 1, 1, 2, 2, 3]
k3 = solution.removeDuplicates(nums3)
print(f"示例 3 的新长度: {k3}, 去重后的 nums3: {nums3[:k3]}") # 输出: 示例 3 的新长度: 3, 去重后的 nums3: [1, 2, 3]

# 调用示例 4 (空数组)
nums4 = []
k4 = solution.removeDuplicates(nums4)
print(f"示例 4 的新长度: {k4}, 去重后的 nums4: {nums4[:k4]}") # 输出: 示例 4 的新长度: 0, 去重后的 nums4: []

# 调用示例 5 (只有一个元素的数组)
nums5 = [5]
k5 = solution.removeDuplicates(nums5)
print(f"示例 5 的新长度: {k5}, 去重后的 nums5: {nums5[:k5]}") # 输出: 示例 5 的新长度: 1, 去重后的 nums5: [5]
