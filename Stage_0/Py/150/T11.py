from typing import List

class Solution:
    def hIndex(self, citations: List[int]) -> int:
        # 将引用次数数组降序排序
        citations.sort(reverse=True)
        n = len(citations)
        h = 0
        for i in range(n):
            # 检查当前论文的引用次数是否大于等于当前的论文数量（i+1）
            if citations[i] >= i + 1:
                h = i + 1  # 更新h为可能的更大值
            else:
                # 后面的引用次数更小，无法满足条件，提前终止循环
                break
        return h

# 调用示例
if __name__ == "__main__":
    solution = Solution()
    # 示例1
    print(solution.hIndex([3,0,6,1,5]))  # 输出3
    # 示例2
    print(solution.hIndex([1,3,1]))      # 输出1
