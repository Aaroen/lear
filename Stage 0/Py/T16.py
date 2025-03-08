class Solution(object):
    def tictactoe(self, moves):
        """
        :type moves: List[List[int]]
        :rtype: str
        """
        game = [[' ' for _ in range (3)]for _ in range (3)]
        Players = ['A','B']

        # 输入棋盘元素
        for i,move in enumerate(moves):
            player = Players[i % 2]
            r , c = move
            game[r][c] = player

        # 判断行是否构成胜利条件
        for h in range(3):
            if all((game[h][v]) == player for v in range(3)):
                return player
            
        # 判断列是否构成胜利条件
        for v in range(3):
            if all((game[h][v])== player for h in range(3)):
                return player
        
         # 判断斜对角是否构成胜利条件
        if all(game[i][i] == player for i in range(3)) or all(game[i][2-i] == player for i in range(3)):
            return player
        
        # 判断游戏是否结束
        if len(moves) == 9:
            return 'Draw'
        else:
            return 'Pending'
       
        
        # 示例 1
moves1 = [[0,0],[2,0],[1,1],[2,1],[2,2]]
solution = Solution() # 创建 Solution 类的实例。
result1 = solution.tictactoe(moves1) # 调用 tictactoe 方法，传入 moves1 作为参数，并将返回值赋值给 result1。
print(f"输入: moves = {moves1}, 输出: {result1}") # 打印输入和输出结果。
# 输出: 输入: moves = [[0, 0], [2, 0], [1, 1], [2, 1], [2, 2]], 输出: A


# 示例 2
moves2 = [[0,0],[1,1],[0,1],[0,2],[1,0],[2,0]]
result2 = solution.tictactoe(moves2) # 调用 tictactoe 方法，传入 moves2 作为参数，并将返回值赋值给 result2。
print(f"输入: moves = {moves2}, 输出: {result2}") # 打印输入和输出结果。
# 输出: 输入: moves = [[0, 0], [1, 1], [0, 1], [0, 2], [1, 0], [2, 0]], 输出: B


# 示例 3
moves3 = [[0,0],[1,1],[2,0],[1,0],[1,2],[2,1],[0,1],[0,2],[2,2]]
result3 = solution.tictactoe(moves3) # 调用 tictactoe 方法，传入 moves3 作为参数，并将返回值赋值给 result3。
print(f"输入: moves = {moves3}, 输出: {result3}") # 打印输入和输出结果。
# 输出: 输入: moves = [[0, 0], [1, 1], [2, 0], [1, 0], [1, 2], [2, 1], [0, 1], [0, 2], [2, 2]], 输出: Draw


# 示例 4 (Pending 示例)
moves4 = [[0,0],[1,1]]
result4 = solution.tictactoe(moves4) # 调用 tictactoe 方法，传入 moves4 作为参数，并将返回值赋值给 result4。
print(f"输入: moves = {moves4}, 输出: {result4}") # 打印输入和输出结果。
# 输出: 输入: moves = [[0, 0], [1, 1]], 输出: Pending
