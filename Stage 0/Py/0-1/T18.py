class Solution(object):
    def maximumWealth(self, accounts):
        """
        :type accounts: List[List[int]]
        :rtype: int
        """
        max_money = 0
        for account in accounts:
            all_money = 0
            for money in account:
                all_money +=money
            
            max_money = max(max_money,all_money)
        
        return max_money
    
    # 示例1：
solution = Solution()
print(solution.maximumWealth([[1,2,3],[3,2,1]]))  # 输出6
# 解释：
# 第1位客户资产总和：1+2+3=6
# 第2位客户资产总和：3+2+1=6
# 最大值为6

# 示例2：
print(solution.maximumWealth([[1,5],[7,3],[3,5]]))  # 输出10
# 解释：
# 第1位客户总和：6，第2位：10，第3位：8 → 最大值10

# 示例3：
print(solution.maximumWealth([[2,8,7],[7,1,3],[1,9,5]]))  # 输出17
# 解释：
# 第1位总和：17，第2位：11，第3位：15 → 最大值17
