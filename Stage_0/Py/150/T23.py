class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        n = len(s)
        m = len(t)
        if n > m :
            return False
        if n == 0:
            return True
        location = 0
        for i in range(m):
            if location < n and t[i] == s[location]:
                location += 1
        if location == n:
            return True
        else:
            return False

# class Solution:
#     def isSubsequence(self, s: str, t: str) -> bool:
#         i = 0  # 指针 i 指向字符串 s
#         j = 0  # 指针 j 指向字符串 t

#         while i < len(s) and j < len(t): # 确保 i 和 j 都不超出字符串长度
#             if s[i] == t[j]:  # 如果 s[i] 和 t[j] 字符相等
#                 i += 1       # s 的指针 i 向后移动，匹配下一个字符
#             j += 1           # t 的指针 j 始终向后移动

#         return i == len(s)  # 如果 i 移动到 s 的末尾，说明 s 是 t 的子序列

import collections
import bisect # 导入二分查找模块

# class Solution:
#     def isSubsequence(self, s: str, t: str) -> bool:
#         char_indices = collections.defaultdict(list) # 使用 defaultdict 初始化，方便添加索引
#         for index, char in enumerate(t):
#             char_indices[char].append(index) # 构建字符索引

#         t_index = -1 # 初始化 t 中已匹配到的索引

#         for char_s in s:
#             if char_s not in char_indices: # 如果 s 中的字符不在 t 中，直接返回 False
#                 return False

#             indices = char_indices[char_s] # 获取字符 char_s 在 t 中的索引列表
#             # 使用 bisect_right 找到第一个大于 t_index 的索引
#             next_index_pos = bisect.bisect_right(indices, t_index) # 二分查找
#             if next_index_pos == len(indices): # 如果没有找到更大的索引，说明无法匹配
#                 return False

#             t_index = indices[next_index_pos] # 更新 t_index 为找到的索引

#         return True # 成功匹配 s 的所有字符


# 示例 1
s1 = "abc"
t1 = "ahbgdc"
sol = Solution()
result1 = sol.isSubsequence(s1, t1)
print(f"s = '{s1}', t = '{t1}', isSubsequence = {result1}") # 输出: s = 'abc', t = 'ahbgdc', isSubsequence = True

# 示例 2
s2 = "axc"
t2 = "ahbgdc"
result2 = sol.isSubsequence(s2, t2)
print(f"s = '{s2}', t = '{t2}', isSubsequence = {result2}") # 输出: s = 'axc', t = 'ahbgdc', isSubsequence = False
