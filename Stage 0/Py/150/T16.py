from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        n = len(height)
        left_max = [0] * n
        right_max = [0] * n
        
        # 计算每个位置左边的最大高度
        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        # 计算每个位置右边的最大高度
        right_max[-1] = height[-1]
        for i in range(n-2, -1, -1):
            right_max[i] = max(right_max[i+1], height[i])
        
        # 计算每个位置能接的雨水量并累加
        res = 0
        for i in range(n):
            res += min(left_max[i], right_max[i]) - height[i]
        
        return res

# 调用示例
if __name__ == "__main__":
    solution = Solution()
    
    # 示例1
    height1 = [0,1,0,2,1,0,1,3,2,1,2,1]
    print(solution.trap(height1))  # 输出6
    
    # 示例2
    height2 = [4,2,0,3,2,5]
    print(solution.trap(height2))  # 输出9
