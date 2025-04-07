class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        magazine_counts = {}  # 用字典存储 magazine 中每个字符的计数

        # 统计 magazine 中每个字符的出现次数
        for char in magazine:
            magazine_counts[char] = magazine_counts.get(char, 0) + 1

        # 遍历 ransomNote 的每个字符，检查是否能在 magazine 中找到足够的字符
        for char in ransomNote:
            if char not in magazine_counts or magazine_counts[char] == 0:
                return False  # 如果字符不在 magazine 中，或者 magazine 中该字符已经用完，则返回 False
            magazine_counts[char] -= 1 # 用掉一个字符，计数减 1

        return True  # 如果 ransomNote 中所有字符都能在 magazine 中找到，则返回 True

solution = Solution()

# 示例 1
ransomNote1 = "a"
magazine1 = "b"
result1 = solution.canConstruct(ransomNote1, magazine1)
print(f"输入: ransomNote = '{ransomNote1}', magazine = '{magazine1}', 输出: {result1} (预期: false) - {result1 == False}") # 示例输入直接判断并返回该代码编写是否正确

# 示例 2
ransomNote2 = "aa"
magazine2 = "ab"
result2 = solution.canConstruct(ransomNote2, magazine2)
print(f"输入: ransomNote = '{ransomNote2}', magazine = '{magazine2}', 输出: {result2} (预期: false) - {result2 == False}")

# 示例 3
ransomNote3 = "aa"
magazine3 = "aab"
result3 = solution.canConstruct(ransomNote3, magazine3)
print(f"输入: ransomNote = '{ransomNote3}', magazine = '{magazine3}', 输出: {result3} (预期: true) - {result3 == True}")

# 更多测试用例
ransomNote4 = "bg"
magazine4 = "efjbdfbdgfjhhaiigfhbaejahgfbbgbjagbddfgdiaigdadhcfcj"
result4 = solution.canConstruct(ransomNote4, magazine4)
print(f"输入: ransomNote = '{ransomNote4}', magazine = '{magazine4}', 输出: {result4} (预期: true) - {result4 == True}")

ransomNote5 = "fihjjjjei"
magazine5 = "hjibagacbhadfaefbhgcebdagdgiahfaefhbhfdjeegdi"
result5 = solution.canConstruct(ransomNote5, magazine5)
print(f"输入: ransomNote = '{ransomNote5}', magazine = '{magazine5}', 输出: {result5} (预期: false) - {result5 == False}")
