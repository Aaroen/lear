class Solution:
    def reverseWords(self, s: str) -> str:
        # 1. 使用 split() 方法分割字符串 s 成单词列表。
        #    split() 默认以空格作为分隔符，并且会自动处理多个空格以及字符串首尾的空格。
        words = s.split()

        # 2. 反转单词列表。
        words.reverse()
        # 或者可以使用 words = words[::-1] 达到相同的反转效果。

        # 3. 使用 " ".join() 方法将反转后的单词列表连接成字符串，单词之间用单个空格分隔。
        return " ".join(words)

# 调用示例
if __name__ == '__main__':
    solution = Solution()

    # 示例 1
    s1 = "the sky is blue"
    result1 = solution.reverseWords(s1)
    print(f"输入: '{s1}', 输出: '{result1}'")  # 输出: 输入: 'the sky is blue', 输出: 'blue is sky the'

    # 示例 2
    s2 = "  hello world  "
    result2 = solution.reverseWords(s2)
    print(f"输入: '{s2}', 输出: '{result2}'")  # 输出: 输入: '  hello world  ', 输出: 'world hello'

    # 示例 3
    s3 = "a good   example"
    result3 = solution.reverseWords(s3)
    print(f"输入: '{s3}', 输出: '{result3}'")  # 输出: 输入: 'a good   example', 输出: 'example good a'

    # 包含前导、尾随和中间多余空格的复杂示例
    s4 = "  leading and trailing spaces  and  multiple    spaces   "
    result4 = solution.reverseWords(s4)
    print(f"输入: '{s4}', 输出: '{result4}'")  # 输出: 输入: '  leading and trailing spaces  and  multiple    spaces   ', 输出: 'spaces multiple and spaces trailing and leading'

    # 单个单词的示例
    s5 = "word"
    result5 = solution.reverseWords(s5)
    print(f"输入: '{s5}', 输出: '{result5}'")  # 输出: 输入: 'word', 输出: 'word'
