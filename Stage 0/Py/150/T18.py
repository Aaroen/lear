from typing import List

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # 如果输入列表为空，直接返回空字符串
        if not strs:
            return ""
        # 获取所有字符串中的最小长度
        min_len = min(len(s) for s in strs)
        # 用于存储公共前缀的字符列表
        prefix = []
        # 遍历每个字符位置
        for i in range(min_len):
            # 获取第一个字符串的当前字符作为基准
            current_char = strs[0][i]
            # 检查其他字符串在相同位置的字符是否相同
            for s in strs[1:]:
                if s[i] != current_char:
                    return ''.join(prefix)
            # 所有字符串在该位置的字符相同，加入前缀
            prefix.append(current_char)
        # 返回最终拼接的公共前缀
        return ''.join(prefix)

# 调用示例
solution = Solution()
print(solution.longestCommonPrefix(["flower","flow","flight"]))  # 输出: "fl"
print(solution.longestCommonPrefix(["dog","racecar","car"]))     # 输出: ""
