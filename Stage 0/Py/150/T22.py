class Solution:
    def isPalindrome(self, s: str) -> bool:
        processed = ''.join(c.lower() for c in s if c.isalnum())
        n = len(processed)
        if n<2:
            return True
        for i in range(n):
            if processed[i] == processed[n-i-1]:
                continue
            else:
                return False
        return True

# class Solution:
#     def isPalindrome(self, s: str) -> bool:
#         # 处理字符串，仅保留字母数字并转为小写
#         processed = ''.join(c.lower() for c in s if c.isalnum())
#         # 比较处理后的字符串与其反转是否相同
#         return processed == processed[::-1]

solution = Solution()
print(solution.isPalindrome("A man, a plan, a canal: Panama"))  # 输出: True
print(solution.isPalindrome("race a car"))  # 输出: False
print(solution.isPalindrome(" "))  # 输出: True
