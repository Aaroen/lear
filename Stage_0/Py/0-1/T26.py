from typing import List

class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        # 如果点的数量小于等于2，直接返回True
        if len(coordinates) <= 2:
            return True
        
        # 获取第一个和第二个点的坐标
        x0, y0 = coordinates[0]
        x1, y1 = coordinates[1]
        # 计算基准向量的增量
        dx = x1 - x0
        dy = y1 - y0
        
        # 遍历剩余的点进行检查
        for point in coordinates[2:]:
            xi, yi = point
            # 检查叉积是否为零，即是否共线
            if (xi - x0) * dy != (yi - y0) * dx:
                return False
        
        return True

# 调用示例
solution = Solution()

# 示例1：应返回True
coordinates1 = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
print(solution.checkStraightLine(coordinates1))  # 输出：True

# 示例2：应返回False
coordinates2 = [[1,1],[2,2],[3,4],[4,5],[5,6],[7,7]]
print(solution.checkStraightLine(coordinates2))  # 输出：False
