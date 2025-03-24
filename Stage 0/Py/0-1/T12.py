class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        word = s.split()
        return len(word[-1])
    
    
    # 调用示例
solution = Solution()
print(solution.lengthOfLastWord("Hello World"))          # 输出5
print(solution.lengthOfLastWord("   fly me   to   the moon  "))  # 输出4
print(solution.lengthOfLastWord("luffy is still joyboy"))  # 输出6