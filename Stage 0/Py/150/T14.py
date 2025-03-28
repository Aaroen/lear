from typing import List
# class Solution:
#     def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
#         n = len(gas)
#         for i in range(n):
#             # 遍历判断每个加油站的油量是否足够支持出发
#             if gas[i] >= cost[i]:
#                 # 足够出发则循环判断下一加油站是否支持继续行驶
#                 k = 1
#                 j = (i+1)%n
#                 now_gas = gas[i] - cost[i]
#                 while now_gas + gas[j] >= cost[j] and k < n:
#                     now_gas = now_gas + gas[j] - cost[j]
#                     j =(j+1)%n
#                     k +=1
#                 # 成功循环一周则返回起点位置，否则继续寻找
#                 if k == n:
#                     return i
#         return -1


class Solution:
    def canCompleteCircuit(self, gas: list[int], cost: list[int]) -> int:
        n = len(gas)
        total_tank = 0  # 记录环路总油量盈余/亏损
        current_tank = 0 # 记录当前油箱油量
        start_station = 0 # 潜在的起始加油站索引

        for i in range(n):
            total_tank += gas[i] - cost[i] # 累加每站的油量盈余/亏损
            current_tank += gas[i] - cost[i] # 更新当前油箱油量

            if current_tank < 0: # 如果当前油箱油量为负，说明从当前start_station出发无法到达当前加油站i+1
                start_station = i + 1 # 将起始加油站设置为下一个加油站的下一个站 (实际上是 i+1, 因为循环会继续 i+1)
                current_tank = 0      # 重置当前油箱油量，因为我们从新的起始站重新开始

        if total_tank < 0: # 如果环路总油量是亏损的，说明无解
            return -1
        else:
            return start_station # 否则，返回找到的起始加油站索引




# 调用示例
if __name__ == '__main__':
    solution = Solution()

    # 示例 1
    gas1 = [1, 2, 3, 4, 5]
    cost1 = [3, 4, 5, 1, 2]
    result1 = solution.canCompleteCircuit(gas1, cost1)
    print(f"示例 1 的结果: {result1} (期望: 3)") # 输出: 示例 1 的结果: 3 (期望: 3)

    # 示例 2
    gas2 = [2, 3, 4]
    cost2 = [3, 4, 3]
    result2 = solution.canCompleteCircuit(gas2, cost2)
    print(f"示例 2 的结果: {result2} (期望: -1)") # 输出: 示例 2 的结果: -1 (期望: -1)

    # 更多测试用例 (可选)
    gas3 = [5,1,2,3,4]
    cost3 = [4,4,1,5,1]
    result3 = solution.canCompleteCircuit(gas3, cost3)
    print(f"示例 3 的结果: {result3} (期望: 4)") # 输出: 示例 3 的结果: 4 (期望: 4)

    gas4 = [1]
    cost4 = [2]
    result4 = solution.canCompleteCircuit(gas4, cost4)
    print(f"示例 4 的结果: {result4} (期望: -1)") 
