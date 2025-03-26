from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach = 0  # 记录当前能到达的最远位置
        n = len(nums)   # 数组长度
        
        for i in range(n):
            if i > max_reach:
                # 如果当前位置i无法到达，则直接返回False
                return False
            # 计算从当前位置i能跳到的最远位置
            current = i + nums[i]
            if current > max_reach:
                max_reach = current  # 更新最远可达位置
            # 如果最远可达位置已超过或到达终点，提前返回True
            if max_reach >= n - 1:
                return True
        
        # 遍历结束后，检查是否能到达终点
        return max_reach >= n - 1

# 调用示例
solution = Solution()
print(solution.canJump([2,3,1,1,4]))  # 输出：True
print(solution.canJump([3,2,1,0,4]))  # 输出：False
