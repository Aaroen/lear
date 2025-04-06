class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []  
        # 1. 按起始位置排序
        intervals.sort(key=lambda x: x[0])
        res = []
        start = intervals[0][0]
        end = intervals[0][1]
        for i in range(1, len(intervals)):
            current_start = intervals[i][0]
            current_end = intervals[i][1]
            if current_start <= end:  # 与当前合并区间重叠，扩展 end
                end = max(end, current_end)
            else:  # 不重叠，将当前合并区间加入结果，并开始新合并
                res.append([start, end])
                start = current_start
                end = current_end
        # 4. 循环结束后，将最后一个合并区间加入结果
        res.append([start, end])
        return res


# 示例 1
intervals1 = [[1,3],[2,6],[8,10],[15,18]]
solution = Solution()
output1 = solution.merge(intervals1)
expected_output1 = [[1,6],[8,10],[15,18]]
print(f"输入: {intervals1}, 输出: {output1}, 是否正确: {output1 == expected_output1}")

# 示例 2
intervals2 = [[1,4],[4,5]]
output2 = solution.merge(intervals2)
expected_output2 = [[1,5]]
print(f"输入: {intervals2}, 输出: {output2}, 是否正确: {output2 == expected_output2}")

# 额外示例 3：包含不重叠和重叠，以及需要多次合并的情况
intervals3 = [[1,3],[6,9],[2,5],[10,12]]
output3 = solution.merge(intervals3)
expected_output3 = [[1,5],[6,9],[10,12]]
print(f"输入: {intervals3}, 输出: {output3}, 是否正确: {output3 == expected_output3}")

# 额外示例 4：空输入
intervals4 = []
output4 = solution.merge(intervals4)
expected_output4 = []
print(f"输入: {intervals4}, 输出: {output4}, 是否正确: {output4 == expected_output4}")

# 额外示例 5：单个区间
intervals5 = [[1,5]]
output5 = solution.merge(intervals5)
expected_output5 = [[1,5]]
print(f"输入: {intervals5}, 输出: {output5}, 是否正确: {output5 == expected_output5}")
