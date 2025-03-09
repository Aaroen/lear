class Solution(object):
    def diagonalSum(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: int
        """
        sum = 0
        for i in range(len(mat)):
            sum += mat[i][i]
            sum += mat[len(mat)-i-1][i]

        if len(mat)%2 == 1:
            sum -= mat[len(mat)//2][len(mat)//2]

        return sum
    
# 示例 1
mat1 = [[1,2,3],
        [4,5,6],
        [7,8,9]]
solution = Solution() # 创建Solution类的实例
result1 = solution.diagonalSum(mat1) # 调用diagonalSum方法，传入矩阵mat1，计算对角线和
print(f"矩阵 {mat1} 的对角线和为: {result1}") # 输出结果。f-string格式化输出，更易读

# 示例 2
mat2 = [[1,1,1,1],
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1]]
result2 = solution.diagonalSum(mat2)
print(f"矩阵 {mat2} 的对角线和为: {result2}")

# 示例 3
mat3 = [[5]]
result3 = solution.diagonalSum(mat3)
print(f"矩阵 {mat3} 的对角线和为: {result3}")
