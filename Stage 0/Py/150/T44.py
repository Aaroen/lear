from typing import List

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        i = 0
        n = len(intervals)
        start, end = newInterval
        
        # 添加所有在新区间之前的无重叠区间
        while i < n and intervals[i][1] < start:
            result.append(intervals[i])
            i += 1
        
        # 合并重叠区间
        while i < n and intervals[i][0] <= end:
            start = min(start, intervals[i][0])
            end = max(end, intervals[i][1])
            i += 1
        result.append([start, end])
        
        # 添加剩余区间
        while i < n:
            result.append(intervals[i])
            i += 1
        
        return result

sol = Solution()
# 示例 1
print(sol.insert([[1,3],[6,9]], [2,5]) == [[1,5],[6,9]])  # True
# 示例 2
print(sol.insert([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]) == [[1,2],[3,10],[12,16]])  # True
