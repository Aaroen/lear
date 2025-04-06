class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        for char in s:
            if char in mapping:
                # 右括号的情况
                if not stack:
                    return False
                top_element = stack.pop()
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)
        return not stack


solution = Solution()

# 示例 1
s1 = "()"
result1 = solution.isValid(s1)
print(f"输入: s = '{s1}', 输出: {result1}, 代码编写是否正确: {result1 == True}") # 输出: 输入: s = '()', 输出: True, 代码编写是否正确: True

# 示例 2
s2 = "()[]{}"
result2 = solution.isValid(s2)
print(f"输入: s = '{s2}', 输出: {result2}, 代码编写是否正确: {result2 == True}") # 输出: 输入: s = '()[]{}', 输出: True, 代码编写是否正确: True

# 示例 3
s3 = "(]"
result3 = solution.isValid(s3)
print(f"输入: s = '{s3}', 输出: {result3}, 代码编写是否正确: {result3 == False}") # 输出: 输入: s = '(]', 输出: False, 代码编写是否正确: True

# 示例 4
s4 = "([])"
result4 = solution.isValid(s4)
print(f"输入: s = '{s4}', 输出: {result4}, 代码编写是否正确: {result4 == True}") # 输出: 输入: s = '([])', 输出: True, 代码编写是否正确: True

# 更多测试用例 (您可以自行添加更多测试)
s5 = "{[]}"
result5 = solution.isValid(s5)
print(f"输入: s = '{s5}', 输出: {result5}, 代码编写是否正确: {result5 == True}") # 输出: 输入: s = '{[]}', 输出: True, 代码编写是否正确: True

s6 = "([)]"
result6 = solution.isValid(s6)
print(f"输入: s = '{s6}', 输出: {result6}, 代码编写是否正确: {result6 == False}") # 输出: 输入: s = '([)]', 输出: False, 代码编写是否正确: True

s7 = "]"
result7 = solution.isValid(s7)
print(f"输入: s = '{s7}', 输出: {result7}, 代码编写是否正确: {result7 == False}") # 输出: 输入: s = ']', 输出: False, 代码编写是否正确: True

s8 = "["
result8 = solution.isValid(s8)
print(f"输入: s = '{s8}', 输出: {result8}, 代码编写是否正确: {result8 == False}") # 输出: 输入: s = '[', 输出: False, 代码编写是否正确: True
