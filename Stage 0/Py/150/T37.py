from typing import List # 导入 List 类型提示，增加代码可读性

class Solution: # 定义 Solution 类
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 定义 groupAnagrams 方法，输入是字符串列表 strs，输出是字符串列表的列表

        anagram_groups = {}  # 初始化一个空字典 anagram_groups，用于存储字母异位词分组。
                             # 键(key)将是排序后的字符串，值(value)将是原始字符串的列表。

        for str_word in strs: # 遍历输入的字符串列表 strs 中的每个字符串，命名为 str_word
            sorted_word = "".join(sorted(str_word)) # 1. 对当前字符串 str_word 进行排序。
                                                    #    sorted(str_word) 将字符串转换为字符列表并排序。
                                                    #    "".join(...) 将排序后的字符列表重新连接成一个字符串。
                                                    #    sorted_word 就是排序后的标准形式的字符串。

            if sorted_word in anagram_groups: # 2. 检查 sorted_word 是否已经作为键存在于 anagram_groups 字典中。
                anagram_groups[sorted_word].append(str_word) # 3. 如果 sorted_word 已经存在，说明我们之前遇到过它的字母异位词。
                                                            #    将当前的原始字符串 str_word 添加到 anagram_groups 中键为 sorted_word 的值 (列表) 中。
            else: # 4. 如果 sorted_word 不存在于 anagram_groups 中，说明这是第一次遇到这种字母构成的词。
                anagram_groups[sorted_word] = [str_word] # 5. 在 anagram_groups 中创建一个新的键值对：
                                                            #    键为 sorted_word，值为一个新的列表 [str_word]，列表中只包含当前的原始字符串 str_word。

        return list(anagram_groups.values()) # 6. 遍历完所有输入的字符串后，anagram_groups 字典中已经包含了所有字母异位词的分组。
                                            #    anagram_groups.values() 返回字典中所有的值 (在这里是列表的集合)。
                                            #    list(...) 将这些值转换为一个列表，最终返回分组后的结果，即列表的列表。

# 示例 1
strs1 = ["eat", "tea", "tan", "ate", "nat", "bat"]
solution = Solution()
output1 = solution.groupAnagrams(strs1)
expected_output1 = [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
output1_sorted = sorted([sorted(group) for group in output1]) # 为了比较，将输出和预期输出都排序，忽略组的顺序和组内单词的顺序
expected_output1_sorted = sorted([sorted(group) for group in expected_output1])
is_correct1 = output1_sorted == expected_output1_sorted
print(f"输入: {strs1}, 输出: {output1}, 预期输出: {expected_output1}, 是否正确: {is_correct1}") # 输出结果并判断是否正确


# 示例 2
strs2 = [""]
output2 = solution.groupAnagrams(strs2)
expected_output2 = [[""]]
output2_sorted = sorted([sorted(group) for group in output2])
expected_output2_sorted = sorted([sorted(group) for group in expected_output2])
is_correct2 = output2_sorted == expected_output2_sorted
print(f"输入: {strs2}, 输出: {output2}, 预期输出: {expected_output2}, 是否正确: {is_correct2}")

# 示例 3
strs3 = ["a"]
output3 = solution.groupAnagrams(strs3)
expected_output3 = [["a"]]
output3_sorted = sorted([sorted(group) for group in output3])
expected_output3_sorted = sorted([sorted(group) for group in expected_output3])
is_correct3 = output3_sorted == expected_output3_sorted
print(f"输入: {strs3}, 输出: {output3}, 预期输出: {expected_output3}, 是否正确: {is_correct3}")
