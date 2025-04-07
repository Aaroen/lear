class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        current_num = 0
        over_num = 0
        res = []
        for i in range(len(nums)):
            if  nums[i] - 1 not in nums:
                current_num = nums[i]
            if  nums[i] + 1 not in nums:
                over_num = nums[i]
                if over_num == current_num:
                    res.append(f"{over_num}")
                else:
                    res.append(f"{current_num}->{over_num}")
        return res

# class Solution:
#     def summaryRanges(self, nums: List[int]) -> List[str]:
#         if not nums:
#             return []
#         result = []
#         start = nums[0]
#         for i in range(1, len(nums)):
#             if nums[i] != nums[i-1] + 1:
#                 if start == nums[i-1]:
#                     result.append(str(start))
#                 else:
#                     result.append(f"{start}->{nums[i-1]}")
#                 start = nums[i]
#         # 处理最后一个区间
#         if start == nums[-1]:
#             result.append(str(start))
#         else:
#             result.append(f"{start}->{nums[-1]}")
#         return result


# 示例1
nums = [0,1,2,4,5,7]
print(Solution().summaryRanges(nums))  # 输出: ["0->2","4->5","7"] → True

# 示例2
nums = [0,2,3,4,6,8,9]
print(Solution().summaryRanges(nums))  # 输出: ["0","2->4","6","8->9"] → True

# 空数组测试
nums = []
print(Solution().summaryRanges(nums))  # 输出: [] → True

# 单元素测试
nums = [1]
print(Solution().summaryRanges(nums))  # 输出: ["1"] → True
