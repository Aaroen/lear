# class Solution:
#     def convert(self, s: str, numRows: int) -> str:
#         if numRows == 1 or len(s) <= numRows:
#             return s
#         rou = numRows*2 - 2
#         str_z= ['']*numRows
#         for i,char in enumerate(s):
#             location = i % rou
#             if location < numRows:
#                 str_z[location] +=char
#             else:
#                 str_z[rou - location] += char
#         return ''.join(str_z)



class Solution:
    def convert(self, s: str, numRows: int) -> str:
        # 处理特殊情况：当行数为1或字符串长度不足行数时，直接返回原字符串
        if numRows == 1 or len(s) <= numRows:
            return s
        
        # 初始化每行的字符串列表，共有numRows行
        rows = [''] * numRows
        cycle = 2 * numRows - 2  # 计算Z字形的周期长度
        
        for i, c in enumerate(s):
            mod = i % cycle  # 当前字符在周期中的位置
            if mod < numRows:
                # 在向下移动的阶段，直接添加到对应行
                rows[mod] += c
            else:
                # 在向上移动的阶段，计算对应的行号并添加
                rows[cycle - mod] += c
        
        # 将所有行的字符串拼接成最终结果
        return ''.join(rows)

# 示例1：输入"PAYPALISHIRING"，行数3，输出"PAHNAPLSIIGYIR"
solution = Solution()
print(solution.convert("PAYPALISHIRING", 3))  # 输出：PAHNAPLSIIGYIR

# 示例2：输入"PAYPALISHIRING"，行数4，输出"PINALSIGYAHRPI"
print(solution.convert("PAYPALISHIRING", 4))  # 输出：PINALSIGYAHRPI

# 示例3：输入"A"，行数1，输出"A"
print(solution.convert("A", 1))  # 输出：A
