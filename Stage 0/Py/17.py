class Solution(object):
    def isRobotBounded(self, instructions):
        """
        :type instructions: str
        :rtype: bool
        """
        place = [0,0]
        new_place = [0,0]
        direct = 0
        for op in instructions:
            if op == 'G':
                if direct == 0:
                    new_place[1] += 1
                elif direct == 1:
                    new_place[0] -= 1
                elif direct == 2:
                    new_place[1] -= 1
                elif direct == 3:
                    new_place[0] += 1
            elif op == 'L': 
                direct = (direct + 1) % 4
            elif op == 'R':
                direct = (direct - 1) % 4
        if new_place == place or direct != 0:
            return True
        else:
            return False


# 示例1：输入 "GGLLGG"，输出 True
solution = Solution()
print(solution.isRobotBounded("GGLLGG"))

# 示例2：输入 "GG"，输出 False
print(solution.isRobotBounded("GG"))


print(solution.isRobotBounded("RGL"))
