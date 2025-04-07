class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        s_to_t_map = {} # s 字符到 t 字符的映射
        t_to_s_map = {} # t 字符到 s 字符的映射

        for i in range(len(s)):
            s_char = s[i]
            t_char = t[i]

            if s_char in s_to_t_map: # 如果 s_char 已经有映射
                if s_to_t_map[s_char] != t_char: # 检查之前的映射是否与当前 t_char 相同
                    return False
            else: # 如果 s_char 还没有映射
                if t_char in t_to_s_map: # 检查 t_char 是否已经被映射到其他 s 字符
                    return False
                else: # 建立新的双向映射
                    s_to_t_map[s_char] = t_char
                    t_to_s_map[t_char] = s_char

        return True


solution = Solution()

# 示例 1
s1 = "egg"
t1 = "add"
result1 = solution.isIsomorphic(s1, t1)
print(f'"{s1}" 和 "{t1}" 是否同构: {result1}') # 输出: true

# 示例 2
s2 = "foo"
t2 = "bar"
result2 = solution.isIsomorphic(s2, t2)
print(f'"{s2}" 和 "{t2}" 是否同构: {result2}') # 输出: false

# 示例 3
s3 = "paper"
t3 = "title"
result3 = solution.isIsomorphic(s3, t3)
print(f'"{s3}" 和 "{t3}" 是否同构: {result3}') # 输出: true

# 更多测试用例 (用于验证代码正确性)
s4 = "badc"
t4 = "baba"
result4 = solution.isIsomorphic(s4, t4)
print(f'"{s4}" 和 "{t4}" 是否同构: {result4}') # 输出: false (因为 'd' 和 'c' 都映射到了 'a')

s5 = "bbbaaaba"
t5 = "aaabbbaa"
result5 = solution.isIsomorphic(s5, t5)
print(f'"{s5}" 和 "{t5}" 是否同构: {result5}') # 输出: false (因为 'b' 映射到 'a', 'a' 映射到 'b', 但是在 t 中，'a' 和 'b' 的相对顺序和 s 中 'b' 和 'a' 的相对顺序不一样)

s6 = "aabb"
t6 = "abab"
result6 = solution.isIsomorphic(s6, t6)
print(f'"{s6}" 和 "{t6}" 是否同构: {result6}') # 输出: true

s7 = ""
t7 = ""
result7 = solution.isIsomorphic(s7, t7)
print(f'"{s7}" 和 "{t7}" 是否同构: {result7}') # 输出: true

s8 = "a"
t8 = "a"
result8 = solution.isIsomorphic(s8, t8)
print(f'"{s8}" 和 "{t8}" 是否同构: {result8}') # 输出: true

s9 = "ab"
t9 = "aa"
result9 = solution.isIsomorphic(s9, t9)
print(f'"{s9}" 和 "{t9}" 是否同构: {result9}') # 输出: false (因为 'b' 也被映射到了 'a', 与 'a' 映射到 'a' 冲突，违反了一对一映射)
