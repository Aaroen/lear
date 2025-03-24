class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []
        
        m = len(matrix)          # 行数
        n = len(matrix[0])       # 列数
        top = 0                  # 上边界
        bottom = m - 1           # 下边界
        left = 0                 # 左边界
        right = n - 1            # 右边界
        res = []                 # 结果列表

        while len(res) < m*n :
            # 处理从左至右
            for i in range(left , right + 1):
                res.append(matrix[top][i])
            top += 1
            if top > bottom:
                break

            # 处理从上到下
            for i in range(top , bottom + 1):
                res.append(matrix[i][right])
            right -= 1
            if left > right:
                break

            # 处理从右至左
            for i in range (right, left-1,-1):
                res.append(matrix[bottom][i])
            bottom -= 1
            if top > bottom:
                break

            # 处理从下到上
            for  i in range (bottom, top-1,-1):
                res.append(matrix[i][left])
            left +=1

        return res
    
matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
solution = Solution()
print(solution.spiralOrder(matrix))  