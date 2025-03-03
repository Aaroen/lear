# class Solution:
#     def findTheDifference(self, s: str, t: str) -> str:
#         char_count_s = {}
#         char_count_t = {}

#         for char in s:
#             char_count_s[char] = char_count_s.get(char, 0) + 1

#         for char in t:
#             char_count_t[char] = char_count_t.get(char, 0) + 1

#         for char, count in char_count_t.items():
#             if char not in char_count_s or char_count_s[char] != count:
#                 if char not in char_count_s or char_count_s[char] < count: # 确保是 t 比 s 多出的那个字符
#                     return char
#         return "" # 理论上不会到达这里，因为题目保证 t 比 s 多一个字符

# solution = Solution()

# word1 = "abc"

# word2 = "abcr"

# result = solution.findTheDifference(word1, word2)

# print(result)

class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        ascode_s = sum(ord(char) for char in s)
        ascode_t = sum(ord(char) for char in t)
        return chr(ascode_t-ascode_s)
    
solution = Solution()

word1 = "abc"

word2 = "bcaj"

result = solution.findTheDifference(word1, word2)

print(result)