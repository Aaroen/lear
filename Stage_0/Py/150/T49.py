class Solution:
    def evalRPN(self, tokens: list[str]) -> int:
        stack = [] # 初始化一个空栈，用于存储数字和中间结果
        operators = {"+", "-", "*", "/"} # 定义运算符集合，方便快速判断token是否为运算符
        for token in tokens: # 遍历输入的 tokens 列表
            if token in operators: # 判断当前 token 是否是运算符
                operand2 = stack.pop() # 如果是运算符，从栈中弹出第二个操作数 (栈顶)
                operand1 = stack.pop() # 从栈中弹出第一个操作数 (栈顶的下一个)
                if token == "+": # 加法运算
                    result = operand1 + operand2
                elif token == "-": # 减法运算
                    result = operand1 - operand2
                elif token == "*": # 乘法运算
                    result = operand1 * operand2
                elif token == "/": # 除法运算
                    result = int(operand1 / operand2) # 整数除法，向零截断
                stack.append(result) # 将计算结果压入栈中
            else: # 如果 token 不是运算符，说明是数字
                stack.append(int(token)) # 将 token 转换为整数并压入栈中
        return stack.pop() # 遍历完所有 token 后，栈顶元素就是最终结果，弹出并返回


# 示例调用代码和判断
solution = Solution()

# 示例 1
tokens1 = ["2","1","+","3","*"]
output1 = solution.evalRPN(tokens1)
expected_output1 = 9
result1 = output1 == expected_output1
print(f"示例 1: tokens = {tokens1}, 输出 = {output1}, 预期输出 = {expected_output1}, 判断结果: {result1}")

# 示例 2
tokens2 = ["4","13","5","/","+"]
output2 = solution.evalRPN(tokens2)
expected_output2 = 6
result2 = output2 == expected_output2
print(f"示例 2: tokens = {tokens2}, 输出 = {output2}, 预期输出 = {expected_output2}, 判断结果: {result2}")

# 示例 3
tokens3 = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
output3 = solution.evalRPN(tokens3)
expected_output3 = 22
result3 = output3 == expected_output3
print(f"示例 3: tokens = {tokens3}, 输出 = {output3}, 预期输出 = {expected_output3}, 判断结果: {result3}")
