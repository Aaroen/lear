import math
from typing import List

class Solution:
    # 主函数，接受单词列表和最大宽度
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # result 用于存储最终排版好的所有行
        result = []
        # i 是当前处理到的单词在 words 列表中的索引
        i = 0
        # n 是单词总数
        n = len(words)

        # 循环处理所有单词，每次循环处理一行
        while i < n:
            # --- 1. 确定当前行能容纳哪些单词 ---
            # line_words 存储当前行的单词
            line_words = []
            # current_length 记录当前行单词的总长度（不含单词间必须的空格）
            current_length = 0
            # start_index 记录当前行第一个单词的索引
            start_index = i

            # 尝试将 words[i] 加入当前行
            # 检查条件：
            # a. i < n: 确保还有单词可以处理
            # b. current_length == 0: 如果是行的第一个单词，直接加
            # c. current_length + len(words[i]) + len(line_words) <= maxWidth:
            #    - current_length: 已有单词的总长
            #    - len(words[i]): 新单词的长度
            #    - len(line_words): 已有单词数，也是至少需要的空格数 (n个单词需要n-1个空格)
            #    如果这三者之和不超过 maxWidth，就可以添加新单词
            while i < n and (current_length == 0 or current_length + len(words[i]) + len(line_words) <= maxWidth):
                # 将当前单词 words[i] 加入 line_words
                line_words.append(words[i])
                # 更新当前行单词的总长度
                current_length += len(words[i])
                # 移动到下一个单词的索引
                i += 1

            # --- 2. 格式化当前行 ---
            # line_str 用于构建当前行的字符串
            line_str = ""
            # num_words_on_line 是当前行的单词数量
            num_words_on_line = len(line_words)
            # is_last_line 判断当前行是否是文本的最后一行
            is_last_line = (i == n)
            # is_single_word 判断当前行是否只有一个单词
            is_single_word = (num_words_on_line == 1)

            # --- 2a. 处理最后一行或只有一个单词的行 (左对齐) ---
            if is_last_line or is_single_word:
                # 使用单个空格连接当前行的所有单词
                line_str = " ".join(line_words)
                # 计算右侧需要填充的空格数量
                trailing_spaces = maxWidth - len(line_str)
                # 在行末尾添加所需数量的空格
                line_str += " " * trailing_spaces
            # --- 2b. 处理需要两端对齐的行 ---
            else:
                # 计算单词之间的总空格数
                total_spaces_needed = maxWidth - current_length
                # 计算单词之间的间隙数量 (n个单词有n-1个间隙)
                num_gaps = num_words_on_line - 1
                # 计算每个间隙至少应有多少个空格
                base_spaces = total_spaces_needed // num_gaps
                # 计算有多少个间隙需要额外多分配一个空格（从左到右分配）
                extra_spaces = total_spaces_needed % num_gaps

                # 开始构建行字符串，从第一个单词开始
                line_str = line_words[0]
                # 遍历剩余的单词 (从第二个单词开始)
                for k in range(1, num_words_on_line):
                    # 计算当前间隙需要添加的空格数
                    spaces_to_add = base_spaces
                    # 如果还有额外的空格需要分配，则当前间隙多加一个空格
                    if extra_spaces > 0:
                        spaces_to_add += 1
                        # 减少一个待分配的额外空格
                        extra_spaces -= 1
                    # 将计算好的空格数和下一个单词拼接到行字符串
                    line_str += " " * spaces_to_add + line_words[k]

            # 将格式化好的当前行字符串添加到结果列表中
            result.append(line_str)

        # 返回包含所有格式化行的列表
        return result

# --- 调用示例 ---
# 创建 Solution 类的实例
solver = Solution()

# 示例 1
words1 = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth1 = 16
output1 = solver.fullJustify(words1, maxWidth1)
print("示例 1 输入:")
print("words =", words1)
print("maxWidth =", maxWidth1)
print("示例 1 输出:")
for line in output1:
    print(f'"{line}"') # 打印时加上引号以明确显示空格
print("-" * 20)

# 示例 2
words2 = ["What","must","be","acknowledgment","shall","be"]
maxWidth2 = 16
output2 = solver.fullJustify(words2, maxWidth2)
print("示例 2 输入:")
print("words =", words2)
print("maxWidth =", maxWidth2)
print("示例 2 输出:")
for line in output2:
    print(f'"{line}"')
print("-" * 20)

# 示例 3
words3 = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"]
maxWidth3 = 20
output3 = solver.fullJustify(words3, maxWidth3)
print("示例 3 输入:")
print("words =", words3)
print("maxWidth =", maxWidth3)
print("示例 3 输出:")
for line in output3:
    print(f'"{line}"')
print("-" * 20)

# 额外测试：只有一个单词的行
words4 = ["Listen","to","many,","speak","to","a","few."]
maxWidth4 = 6
output4 = solver.fullJustify(words4, maxWidth4)
print("额外测试 输入:")
print("words =", words4)
print("maxWidth =", maxWidth4)
print("额外测试 输出:")
for line in output4:
    print(f'"{line}"')
print("-" * 20)

# 额外测试：最后一行
words5 = ["ask","not","what","your","country","can","do","for","you"]
maxWidth5 = 10
output5 = solver.fullJustify(words5, maxWidth5)
print("额外测试 输入:")
print("words =", words5)
print("maxWidth =", maxWidth5)
print("额外测试 输出:")
for line in output5:
    print(f'"{line}"')
print("-" * 20)
