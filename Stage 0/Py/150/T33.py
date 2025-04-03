from typing import List
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        directions = [(-1,-1), (-1,0), (-1,1),
                      (0,-1),          (0,1),
                      (1,-1),  (1,0), (1,1)]
        
        # 第一遍遍历：标记需要改变状态的细胞
        for i in range(m):
            for j in range(n):
                live_neighbors = 0
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n and abs(board[ni][nj]) == 1:
                        live_neighbors += 1
                
                # 规则1和规则3：活细胞死亡
                if board[i][j] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                    board[i][j] = -1
                # 规则4：死细胞复活
                if board[i][j] == 0 and live_neighbors == 3:
                    board[i][j] = 2
        
        # 第二遍遍历：更新状态
        for i in range(m):
            for j in range(n):
                if board[i][j] == -1:
                    board[i][j] = 0
                elif board[i][j] == 2:
                    board[i][j] = 1


# 示例1
board1 = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
Solution().gameOfLife(board1)
print(board1 == [[0,0,0],[1,0,1],[0,1,1],[0,1,0]])  # 应输出 True

# 示例2
board2 = [[1,1],[1,0]]
Solution().gameOfLife(board2)
print(board2 == [[1,1],[1,1]])  # 应输出 True
