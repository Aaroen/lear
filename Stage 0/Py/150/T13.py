from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        计算数组中每个元素之外其余元素的乘积。

        不要使用除法，且时间复杂度为 O(n)。

        参数:
            nums: 整数数组。

        返回:
            数组 answer，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
        """
        n = len(nums)
        answer = [0] * n  # 初始化 answer 数组，与 nums 长度相同，初始值设为0

        # 步骤 1: 计算前缀乘积
        # prefix_prod[i] 表示 nums[0] * nums[1] * ... * nums[i-1] 的乘积 (不包括 nums[i])
        prefix_prod = 1  # 初始化前缀乘积为 1
        for i in range(n):
            answer[i] = prefix_prod  # answer[i] 先存储 nums[i] 之前所有元素的乘积
            prefix_prod *= nums[i]     # 更新前缀乘积，包含 nums[i]

        # 步骤 2: 计算后缀乘积并结合前缀乘积
        # suffix_prod[i] 表示 nums[i+1] * nums[i+2] * ... * nums[n-1] 的乘积 (不包括 nums[i])
        suffix_prod = 1  # 初始化后缀乘积为 1
        for i in range(n - 1, -1, -1):  # 从后向前遍历数组
            answer[i] *= suffix_prod   # 将 answer[i] (已有的前缀乘积) 乘以后缀乘积，得到最终结果
            suffix_prod *= nums[i]     # 更新后缀乘积，包含 nums[i]

        return answer


# 调用示例
if __name__ == '__main__':
    sol = Solution()

    # 示例 1
    nums1 = [1, 2, 3, 4]
    result1 = sol.productExceptSelf(nums1)
    print(f"输入: nums = {nums1}")
    print(f"输出: answer = {result1}")  # 输出: [24, 12, 8, 6]

    # 示例 2
    nums2 = [-1, 1, 0, -3, 3]
    result2 = sol.productExceptSelf(nums2)
    print(f"输入: nums = {nums2}")
    print(f"输出: answer = {result2}")  # 输出: [0, 0, 9, 0, 0]

    # 更多测试用例 (可选)
    nums3 = [0, 0]
    result3 = sol.productExceptSelf(nums3)
    print(f"输入: nums = {nums3}")
    print(f"输出: answer = {result3}") # 输出: [0, 0]

    nums4 = [2, 2, 2]
    result4 = sol.productExceptSelf(nums4)
    print(f"输入: nums = {nums4}")
    print(f"输出: answer = {result4}") # 输出: [4, 4, 4]
