from typing import List
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        if n < 3:
            return []
        ans = []
        nums.sort()
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left , right = i + 1 , n - 1
            while right > left :
                sum = nums[i] + nums[right] + nums[left]
                if sum == 0:
                    ans.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif sum > 0:
                    right -= 1
                else:
                    left += 1
        return ans

# 示例1
nums1 = [-1, 0, 1, 2, -1, -4]
solution = Solution()
print(solution.threeSum(nums1))  # 输出：[[-1,-1,2], [-1,0,1]]

# 示例2
nums2 = [0, 1, 1]
print(solution.threeSum(nums2))  # 输出：[]

# 示例3
nums3 = [0, 0, 0]
print(solution.threeSum(nums3))  # 输出：[[0,0,0]]


# class Solution:
#     def threeSum(self, nums: list[int]) -> list[list[int]]:
#         n = len(nums)
#         if n < 3:
#             return []

#         nums.sort() # 排序数组
#         ans = []

#         for i in range(n - 2): # 只需要遍历到倒数第三个元素
#             # 去重：如果 nums[i] 和前一个数相同，则跳过
#             if i > 0 and nums[i] == nums[i - 1]:
#                 continue

#             left = i + 1
#             right = n - 1

#             while left < right:
#                 current_sum = nums[i] + nums[left] + nums[right]

#                 if current_sum == 0:
#                     ans.append([nums[i], nums[left], nums[right]])

#                     # 去重：跳过重复的 nums[left] 和 nums[right]
#                     while left < right and nums[left] == nums[left + 1]:
#                         left += 1
#                     while left < right and nums[right] == nums[right - 1]:
#                         right -= 1

#                     left += 1 # 找到一个解后，left 和 right 同时向中间移动，寻找下一个解
#                     right -= 1
#                 elif current_sum < 0:
#                     left += 1 # 和太小，left 右移增大和
#                 else: # current_sum > 0
#                     right -= 1 # 和太大，right 左移减小和

#         return ans
