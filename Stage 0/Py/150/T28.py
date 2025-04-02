class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        left = 0
        strs = set()
        for right in range(len(s)):
            while s[right] in strs:
                strs.remove(s[left])
                left += 1
            strs.add(s[right])
            max_len = max(right - left + 1,max_len)
        return max_len



solution = Solution()

# 示例1
print(solution.lengthOfLongestSubstring("abcabcbb") == 3)  # True

# 示例2
print(solution.lengthOfLongestSubstring("bbbbb") == 1)    # True

# 示例3
print(solution.lengthOfLongestSubstring("pwwkew") == 3)   # True

# 空字符串测试
print(solution.lengthOfLongestSubstring("") == 0)         # True

# 所有字符唯一测试
print(solution.lengthOfLongestSubstring("abcdef") == 6)   # True
