import collections
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t) # 统计 t 中字符的需求数量
        window = collections.Counter() # 窗口中字符的计数
        left, right = 0, 0
        valid = 0 # 窗口中满足 need 要求的字符种类数量
        start = 0 # 最小覆盖子串的起始索引
        length = float('inf') # 最小覆盖子串的长度，初始为无穷大
        while right < len(s) :
            char = s[right]
            right += 1
            if char in need:
                window[char] += 1
                if window[char] == need[char] :
                    valid += 1
            while valid == len(need) :
                if right - left < length:
                    length = right - left
                    start = left
                char_l = s[left]
                left += 1
                if char_l in need:
                    if window[char_l] == need[char_l]:
                        valid -= 1
                    window[char_l] -= 1
        return "" if length == float('inf') else s[start:start + length]



solution = Solution()

# 示例1
print(solution.minWindow("ADOBECODEBANC", "ABC") == "BANC")  # True

# 示例2
print(solution.minWindow("a", "a") == "a")  # True

# 示例3
print(solution.minWindow("a", "aa") == "")  # True

