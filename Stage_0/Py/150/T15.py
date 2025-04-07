from typing import List
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [1]*n # 初始化每个孩子至少一个糖果
        # 从左往右遍历，分数增加则糖果加一，否则为一
        for i in range(1,n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1

        # 从右向左遍历，修正糖果分配
        for i in range(n-2,-1,-1):
            if ratings[i] > ratings[i+1]:
                candies[i] = max(candies[i],candies[i+1]+1)
        return sum(candies)


# 调用示例
if __name__ == '__main__':
    solution = Solution()

    # 示例 1
    ratings1 = [1, 0, 2]
    result1 = solution.candy(ratings1)
    print(f"输入: ratings = {ratings1}, 输出: {result1}, 解释: {result1 == 5}") # 预期输出 5

    # 示例 2
    ratings2 = [1, 2, 2]
    result2 = solution.candy(ratings2)
    print(f"输入: ratings = {ratings2}, 输出: {result2}, 解释: {result2 == 4}") # 预期输出 4

    # 更多测试用例
    ratings3 = [1,2,87,87,87,2,1]
    result3 = solution.candy(ratings3)
    print(f"输入: ratings = {ratings3}, 输出: {result3}, 解释: {result3 == 13}") # 预期输出 13

    ratings4 = [1,3,2,2,1]
    result4 = solution.candy(ratings4)
    print(f"输入: ratings = {ratings4}, 输出: {result4}, 解释: {result4 == 7}") # 预期输出 7