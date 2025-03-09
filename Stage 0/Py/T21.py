class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        if not matrix:  # 检查 matrix 是否为空列表
            return  # 如果为空，直接返回 (不做任何修改)
        m = len(matrix)
        n = len(matrix[0])
        m = len(matrix)
        n = len(matrix[0])
        place_0 = []
        # 记录0的位置
        for i, nums_list in enumerate(matrix):
            for j, num in enumerate(nums_list):
                if num == 0:
                    place_0.append((i, j))
        
        # 将所有0所在行，列全设为0
        for place in place_0: # 注意这里应该遍历 place_0 中的每个元素，每个元素代表一个 0 的位置
            row_index = place[0] # 获取行索引
            col_index = place[1] # 获取列索引

            for j in range(n): # 将整行设为 0
                matrix[row_index][j] = 0 # 使用正确的行索引 row_index

            for i in range(m): # 将整列设为 0
                matrix[i][col_index] = 0 # 使用正确的列索引 col_index

        return matrix

    
# 调用示例 1
matrix1 = [[1,1,1],[1,0,1],[1,1,1]]
solution = Solution()
solution.setZeroes(matrix1)
print(matrix1)  # 输出：[[1, 0, 1], [0, 0, 0], [1, 0, 1]]

# 调用示例 2
matrix2 = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
solution.setZeroes(matrix2)
print(matrix2)  # 输出：[[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]

# 示例 3: 空矩阵
matrix3 = []
solution.setZeroes(matrix3)
print(matrix3) # 输出: []

# 示例 4: 单行矩阵
matrix4 = [[1, 0, 1]]
solution.setZeroes(matrix4)
print(matrix4) # 输出: [[0, 0, 0]]

# 示例 5: 单列矩阵
matrix5 = [[1], [0], [1]]
solution.setZeroes(matrix5)
print(matrix5) # 输出: [[0], [0], [0]]
