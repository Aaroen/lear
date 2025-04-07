import collections # 导入 collections 模块，用于使用 Counter 类
from typing import List # 导入 List 类型提示

class Solution: # 定义解决方案类
    def findSubstring(self, s: str, words: List[str]) -> List[int]: # 定义方法，接收字符串 s 和单词列表 words，返回整数列表
        """
        在字符串 s 中查找所有 words 列表中单词串联形成的子串的起始索引。

        Args:
            s: 主字符串。
            words: 单词列表，所有单词长度相同。

        Returns:
            一个包含所有串联子串起始索引的列表。
        """
        # 1. 初始化和处理边缘情况
        if not s or not words: # 检查输入字符串 s 或单词列表 words 是否为空
            return [] # 如果任一为空，无法形成串联子串，返回空列表

        word_len = len(words[0]) # 获取列表中第一个单词的长度，假设所有单词长度相同
        num_words = len(words) # 获取单词列表中的单词总数
        total_len = word_len * num_words # 计算串联后子串的总长度
        s_len = len(s) # 获取主字符串 s 的长度

        if s_len < total_len: # 如果主字符串 s 的长度小于串联子串的总长度
            return [] # 不可能包含串联子串，返回空列表

        words_count = collections.Counter(words) # 使用 Counter 统计 words 列表中每个单词需要出现的次数
        result = [] # 初始化一个空列表，用于存储找到的串联子串的起始索引

        # 2. 按余数分组进行滑动窗口扫描
        # 因为单词长度固定，有效的串联子串的起始位置模 word_len 的结果必然是 0 到 word_len-1 中的一个
        # 所以我们对这 word_len 种可能的起始偏移量分别进行扫描
        for offset in range(word_len):
            left = offset # 初始化当前扫描偏移量下的窗口左边界
            window_counts = collections.Counter() # 初始化用于存储当前窗口内单词计数的 Counter
            words_found = 0 # 初始化当前窗口中已找到的有效单词（在 words_count 中且数量未超限）的数量

            # 3. 滑动窗口
            # right 指针表示当前考虑的单词块的起始位置
            right = offset # 右边界从当前偏移量开始
            while right + word_len <= s_len: # 当右边界加上一个单词长度不超过 s 的总长度时，继续滑动
                # 获取窗口右侧（即将加入窗口）的单词
                word = s[right : right + word_len]
                # 将右指针向前移动一个单词的长度，准备下一次迭代
                right += word_len

                # 4. 处理新加入的单词 word
                if word in words_count: # 检查这个新单词是否是目标单词之一
                    # 如果是目标单词
                    window_counts[word] += 1 # 在当前窗口的计数器中增加该单词的计数
                    words_found += 1 # 增加窗口中找到的有效单词总数

                    # 5. 处理窗口中单词数量超标的情况
                    # 如果刚才加入的 word 导致其在窗口中的数量超过了 words_count 中要求的数量
                    while window_counts[word] > words_count[word]:
                        # 从窗口左侧移除单词，直到 word 的数量恢复正常
                        left_word = s[left : left + word_len] # 获取最左侧的单词
                        window_counts[left_word] -= 1 # 在窗口计数器中减少最左侧单词的计数
                        words_found -= 1 # 减少窗口中找到的有效单词总数
                        left += word_len # 将左边界向右移动一个单词的长度

                    # 6. 检查是否找到了一个完整的串联子串
                    # 如果窗口中找到的有效单词数等于目标单词总数 num_words
                    # 这意味着当前窗口 [left, right) 包含了所有目标单词，且数量正确
                    if words_found == num_words:
                        result.append(left) # 将当前窗口的起始索引 left 添加到结果列表中
                        # 为了查找下一个可能的匹配，需要将当前窗口最左侧的单词移除
                        # 这模拟了窗口向右滑动一个单词单位的过程
                        left_word = s[left : left + word_len] # 获取最左侧的单词
                        window_counts[left_word] -= 1 # 减少其在窗口中的计数
                        words_found -= 1 # 减少有效单词数
                        left += word_len # 将左边界右移

                else:
                    # 7. 如果新加入的单词 word 不是目标单词
                    # 这意味着从 left 开始到当前 right 之前的窗口不可能形成串联子串
                    # 需要重置窗口状态
                    window_counts.clear() # 清空窗口计数器
                    words_found = 0 # 重置找到的有效单词数
                    # 将左边界直接跳到当前无效单词之后的位置，开始寻找新的可能窗口
                    left = right

        # 8. 返回所有找到的起始索引
        return result # 返回包含所有串联子串起始索引的列表




# 示例 1
s1 = "barfoothefoobarman"
words1 = ["foo", "bar"]
solver = Solution()
output1 = solver.findSubstring(s1, words1)
print(f"示例 1 输入: s = \"{s1}\", words = {words1}")
print(f"示例 1 输出: {output1}")
print(f"示例 1 预期输出: [0, 9] or [9, 0]")
# 判断代码对示例1是否正确 (忽略顺序)
is_correct1 = sorted(output1) == sorted([0, 9])
print(f"示例 1 代码编写是否正确: {is_correct1}\n") # true

# 示例 2
s2 = "wordgoodgoodgoodbestword"
words2 = ["word", "good", "best", "word"]
output2 = solver.findSubstring(s2, words2)
print(f"示例 2 输入: s = \"{s2}\", words = {words2}")
print(f"示例 2 输出: {output2}")
print(f"示例 2 预期输出: []")
is_correct2 = sorted(output2) == sorted([])
print(f"示例 2 代码编写是否正确: {is_correct2}\n") # true

# 示例 3
s3 = "barfoofoobarthefoobarman"
words3 = ["bar", "foo", "the"]
output3 = solver.findSubstring(s3, words3)
print(f"示例 3 输入: s = \"{s3}\", words = {words3}")
print(f"示例 3 输出: {output3}")
print(f"示例 3 预期输出: [6, 9, 12] (任意顺序)")
is_correct3 = sorted(output3) == sorted([6, 9, 12])
print(f"示例 3 代码编写是否正确: {is_correct3}\n") # true

# 附加测试用例：包含重复单词
s4 = "lingmindraboofooowingdingbarrwingmonkeypoundcake"
words4 = ["fooo","barr","wing","ding","wing"]
output4 = solver.findSubstring(s4, words4)
print(f"附加测试 输入: s = \"{s4}\", words = {words4}")
print(f"附加测试 输出: {output4}")
print(f"附加测试 预期输出: [13]")
is_correct4 = sorted(output4) == sorted([13])
print(f"附加测试 代码编写是否正确: {is_correct4}\n") # true

# 附加测试用例：单词不出现
s5 = "aaaaaaaaaaaaaa"
words5 = ["aa","aa"]
output5 = solver.findSubstring(s5, words5)
print(f"附加测试 输入: s = \"{s5}\", words = {words5}")
print(f"附加测试 输出: {output5}")
print(f"附加测试 预期输出: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]") # 预期长度 4, aa 在 s[0:2], s[2:4]; s[1:3], s[3:5] ...
is_correct5 = sorted(output5) == sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"附加测试 代码编写是否正确: {is_correct5}\n") # true (注意这里预期输出计算要小心, 长度为2的"aa"有13个，组合长度为4，所以最多可以从0到10开始)

