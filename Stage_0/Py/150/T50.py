class Solution:
    def calculate(self, s: str) -> int:
        stack = []  # 栈，用于处理括号
        res = 0     # 当前结果
        num = 0     # 当前数字
        sign = 1    # 当前符号，1为正，-1为负

        for char in s:
            if char.isdigit():
                num = num * 10 + int(char) # 构建多位数字
            elif char == '+':
                res += sign * num      # 将之前的数字加到结果
                num = 0                # 重置数字
                sign = 1               # 设置符号为正
            elif char == '-':
                res += sign * num      # 将之前的数字加到结果
                num = 0                # 重置数字
                sign = -1              # 设置符号为负
            elif char == '(':
                stack.append(res)      # 保存当前结果到栈
                stack.append(sign)     # 保存当前符号到栈
                res = 0                # 重置结果，开始计算括号内的表达式
                sign = 1               # 括号内的符号默认为正
            elif char == ')':
                res += sign * num      # 将括号前的数字加入结果
                num = 0                # 重置数字
                prev_sign = stack.pop() # 弹出之前的符号
                prev_res = stack.pop()  # 弹出之前的结果
                res = prev_res + prev_sign * res # 合并括号外的结果和括号内的结果
            elif char == ' ':
                pass                   # 忽略空格

        res += sign * num # 处理最后一个数字
        return res


# 示例调用
solution = Solution()

# 示例 1
s1 = "1 + 1"
output1 = solution.calculate(s1)
print(f"输入: s = '{s1}', 输出: {output1}") # 输出: 2
print(output1 == 2) # 示例 1 是否正确: True

# 示例 2
s2 = " 2-1 + 2 "
output2 = solution.calculate(s2)
print(f"输入: s = '{s2}', 输出: {output2}") # 输出: 3
print(output2 == 3) # 示例 2 是否正确: True

# 示例 3
s3 = "(1+(4+5+2)-3)+(6+8)"
output3 = solution.calculate(s3)
print(f"输入: s = '{s3}', 输出: {output3}") # 输出: 23
print(output3 == 23) # 示例 3 是否正确: True

# 更多测试用例
s4 = "0"
output4 = solution.calculate(s4)
print(f"输入: s = '{s4}', 输出: {output4}") # 输出: 0
print(output4 == 0)

s5 = "-2+ 3" # 测试负数开头
output5 = solution.calculate(s5)
print(f"输入: s = '{s5}', 输出: {output5}") # 输出: 1
print(output5 == 1)

s6 = "-(3-2)" # 测试括号和负号
output6 = solution.calculate(s6)
print(f"输入: s = '{s6}', 输出: {output6}") # 输出: -1
print(output6 == -1)

s7 = "- (3 - 2)" # 测试括号，负号和空格
output7 = solution.calculate(s7)
print(f"输入: s = '{s7}', 输出: {output7}") # 输出: -1
print(output7 == -1)

s8 = "( -3 + 2 )" # 测试括号内有负数和空格
output8 = solution.calculate(s8)
print(f"输入: s = '{s8}', 输出: {output8}") # 输出: -1
print(output8 == -1)
