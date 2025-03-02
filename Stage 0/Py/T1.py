# class Solution:
#     def mergeAlternately(self, word1: str, word2: str) -> str:
#         merged_string = ""
#         len1 = len(word1)
#         len2 = len(word2)
#         i = 0
#         j = 0
#         while i < len1 and j < len2:
#             merged_string += word1[i]
#             merged_string += word2[j]
#             i += 1
#             j += 1

#         while i < len1:
#             merged_string += word1[i]
#             i += 1

#         while j < len2:
#             merged_string += word2[j]
#             j += 1

#         return merged_string

# # 创建 Solution 类的实例
# solution = Solution()

# # 定义输入字符串
# word1 = "abc"
# word2 = "pqr"

# # 调用 mergeAlternately 方法
# result = solution.mergeAlternately(word1, word2)

# # 打印结果
# print(result)



class mer:
    def merge(self,word1:str,word2:str) -> str:
        mergeall = ""
        len1 = len(word1)
        len2 = len(word2)
        i = 0
        j = 0
        while i<len1 and j<len2:
            mergeall += word1[i]
            mergeall += word2[j]
            i += 1
            j += 1
        while i<len1:
            mergeall += word1[i]
            i +=1
        while j<len2:
            mergeall += word2[j]
            j +=1

        return mergeall


solo = mer()
#定义输入字符串
word1 = "abc"
word2 = "pqr"
#调用
result = solo.merge(word1, word2)
#打印结果
print(result)