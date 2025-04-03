from typing import List
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)

        # 1. 转置矩阵 (Transpose)
        for i in range(n):
            for j in range(i + 1, n): # 注意 j 从 i+1 开始，避免重复交换和越界
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        # 2. 水平翻转每一行 (Reverse each row)
        for i in range(n):
            matrix[i].reverse() #  Python 列表自带 reverse() 方法进行原地翻转



matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
expected_output1 = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

solution = Solution()
solution.rotate(matrix1)
output1 = matrix1  # matrix1 会被原地修改

print(output1 == expected_output1) # 判断输出是否与预期相同，应该输出 True
print(output1) # 打印输出结果

matrix2 = [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]
expected_output2 = [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]]

solution = Solution()
solution.rotate(matrix2)
output2 = matrix2

print(output2 == expected_output2) # 应该输出 True
print(output2) # 打印输出结果
