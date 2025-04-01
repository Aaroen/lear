from typing import List
class Solution:
    def minSubArrayLen(self, target: int, nums: list[int]) -> int:
        left = 0
        current_sum = 0
        min_length = float('inf')  # 初始化为正无穷，表示还未找到有效子数组

        for right in range(len(nums)):
            current_sum += nums[right]  # 扩大窗口，将右边界元素加入窗口和

            while current_sum >= target:  # 当窗口和大于等于 target 时，尝试缩小窗口
                min_length = min(min_length, right - left + 1) # 更新最小长度
                current_sum -= nums[left]  # 缩小窗口，从窗口和中减去左边界元素
                left += 1  # 左边界右移

        if min_length == float('inf'): # 如果 min_length 还是初始值，说明没找到有效子数组
            return 0
        else:
            return min_length

target1 = 8
nums1 = [4, 3, 1, 2, 2, 4]
solution = Solution()
result1 = solution.minSubArrayLen(target1, nums1)
print(f"输入: target = {target1}, nums = {nums1}")
print(f"输出: {result1}")