from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        answer = [1] * n
        
        # 计算左边的乘积
        for i in range(1, n):
            answer[i] = answer[i-1] * nums[i-1]
        
        # 计算右边的乘积并更新结果
        right_product = 1
        for i in range(n-1, -1, -1):
            answer[i] *= right_product
            right_product *= nums[i]
        
        return answer

# 调用示例
if __name__ == "__main__":
    solution = Solution()
    print(solution.productExceptSelf([1, 2, 3, 4]))  # 输出: [24, 12, 8, 6]
    print(solution.productExceptSelf([-1, 1, 0, -3, 3]))  # 输出: [0, 0, 9, 0, 0]
