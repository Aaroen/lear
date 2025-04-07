class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        s_to_p = {}
        p_to_s = {}
        s_word = s.split()
        if len(pattern) != len(s_word):
            return False
        for sc, pc in zip(s_word, pattern):
            if sc in s_to_p:
                if s_to_p[sc] != pc:
                    return False
            else:
                if pc in p_to_s:
                    return False
                s_to_p[sc] = pc
                p_to_s[pc] = sc
        return True


solution = Solution()

# 示例 1
pattern1 = "abba"
s1 = "dog cat cat dog"
result1 = solution.wordPattern(pattern1, s1)
print(f"pattern: '{pattern1}', s: '{s1}', result: {result1}") # 输出: pattern: 'abba', s: 'dog cat cat dog', result: True
print(f"示例 1 判断结果: {result1 == True}") # 示例1 判断结果: True

# 示例 2
pattern2 = "abba"
s2 = "dog cat cat fish"
result2 = solution.wordPattern(pattern2, s2)
print(f"pattern: '{pattern2}', s: '{s2}', result: {result2}") # 输出: pattern: 'abba', s: 'dog cat cat fish', result: False
print(f"示例 2 判断结果: {result2 == False}") # 示例 2 判断结果: False

# 示例 3
pattern3 = "aaaa"
s3 = "dog cat cat dog"
result3 = solution.wordPattern(pattern3, s3)
print(f"pattern: '{pattern3}', s: '{s3}', result: {result3}") # 输出: pattern: 'aaaa', s: 'dog cat cat dog', result: False
print(f"示例 3 判断结果: {result3 == False}") # 示例 3 判断结果: False

# 更多测试用例 (可选)
pattern4 = "abcabc"
s4 = "dog cat fish dog cat fish"
result4 = solution.wordPattern(pattern4, s4)
print(f"pattern: '{pattern4}', s: '{s4}', result: {result4}") # 输出: pattern: 'abcabc', s: 'dog cat fish dog cat fish', result: True
print(f"示例 4 判断结果: {result4 == True}") # 示例 4 判断结果: True

pattern5 = "abc"
s5 = "dog dog dog"
result5 = solution.wordPattern(pattern5, s5)
print(f"pattern: '{pattern5}', s: '{s5}', result: {result5}") # 输出: pattern: 'abc', s: 'dog dog dog', result: {False}
print(f"示例 5 判断结果: {result5 == False}") # 示例 5 判断结果: False
