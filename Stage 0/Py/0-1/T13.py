class Solution(object):
    def toLowerCase(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s.lower()
    
# 示例 1
# 创建 Solution 类的实例
solution = Solution()
s1 = "Hello"
result1 = solution.toLowerCase(s1)
print(f"输入: s = '{s1}'")
print(f"输出: '{result1}'")
