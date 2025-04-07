# class Solution:
#     def strStr(self, haystack: str, needle: str) -> int:
#         """
#         在 haystack 字符串中找出 needle 字符串的第一个匹配项的下标。

#         参数:
#         haystack: 要搜索的字符串。
#         needle: 要查找的字符串。

#         返回值:
#         needle 在 haystack 中第一个匹配项的下标。如果 needle 不是 haystack 的一部分，则返回 -1。
#         """
#         if not needle:  # 如果 needle 是空字符串，根据题目描述，应该返回 0
#             return 0

#         for i in range(len(haystack) - len(needle) + 1): # 遍历 haystack 所有可能的起始位置
#             if haystack[i:i + len(needle)] == needle: # 检查从当前位置开始的子字符串是否与 needle 匹配
#                 return i
#         return -1 # 如果循环结束都没有找到匹配项，返回 -1

# solution = Solution()

# haystack = "b5adbutsad"

# needle = "sad"

# result = solution.strStr(haystack, needle)

# print(result)


# class Solution:
#     def strStr(self, haystack: str, needle: str) -> int:
#         for i in range(len(haystack) - len(needle)+1):
#             if(haystack[i:i+len(needle)] == needle):
#                 return i

# solution = Solution()

# haystack = "utsad"

# needle = "sad"

# result = solution.strStr(haystack, needle)

# print(result)

class Solution:
    def compute_prefix_function(self, pattern):
        m = len(pattern)
        pi = [0] * m
        length = 0
        for i in range(1, m):
            while length > 0 and pattern[i] != pattern[length]:
                length = pi[length - 1]
            if pattern[i] == pattern[length]:
                length += 1
            pi[i] = length
        return pi

    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0
        pi = self.compute_prefix_function(needle)
        n = len(haystack)
        m = len(needle)
        j = 0
        for i in range(n):
            while j > 0 and haystack[i] != needle[j]:
                j = pi[j - 1]
            if haystack[i] == needle[j]:
                j += 1
            if j == m:
                return i - m + 1
        return -1

solution = Solution()

haystack = "utsad"

needle = "sad"

result = solution.strStr(haystack, needle)

print(result)