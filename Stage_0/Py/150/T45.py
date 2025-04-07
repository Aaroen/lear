from typing import List

class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0
        
        # 按照气球的结束坐标进行排序
        points.sort(key=lambda x: x[1])
        
        arrows = 1
        first_end = points[0][1]
        
        for x_start, x_end in points[1:]:
            # 如果当前气球的起始坐标大于上一支箭的位置，需要再射一支箭
            if x_start > first_end:
                arrows += 1
                first_end = x_end
                
        return arrows


solution = Solution()

# 示例1
points1 = [[10,16],[2,8],[1,6],[7,12]]
print(solution.findMinArrowShots(points1) == 2)  # 输出: True

# 示例2
points2 = [[1,2],[3,4],[5,6],[7,8]]
print(solution.findMinArrowShots(points2) == 4)  # 输出: True

# 示例3
points3 = [[1,2],[2,3],[3,4],[4,5]]
print(solution.findMinArrowShots(points3) == 2)  # 输出: True
