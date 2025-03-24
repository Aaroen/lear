from typing import List

# class Solution:
#     def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
#         n = len(arr)
#         if n <=2:
#             return True
#         arr.sort()
#         d= arr[1] - arr[0]
#         for i in range (2,n):
#             if d != arr[i] - arr[i-1]:
#                 return False
#         return True
    

class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        n = len(arr)
        if n <=2:
            return True
        arr.sort()
        d= arr[0] - arr[1]
        for i in range (1,n-1):
            if d != arr[i] - arr[i+1]:
                return False
        return True
    


# 调用示例代码
solution = Solution()

# 示例 1
arr1 = [3, 5, 1]
result1 = solution.canMakeArithmeticProgression(arr1)
print(f"数组 {arr1} 是否可以形成等差数列: {result1}") # 输出: 数组 [3, 5, 1] 是否可以形成等差数列: True

# 示例 2
arr2 = [1, 2, 4]
result2 = solution.canMakeArithmeticProgression(arr2)
print(f"数组 {arr2} 是否可以形成等差数列: {result2}") # 输出: 数组 [1, 2, 4] 是否可以形成等差数列: False

# 示例 3 （更多测试用例）
arr3 = [1, 2, 3, 4, 5]
result3 = solution.canMakeArithmeticProgression(arr3)
print(f"数组 {arr3} 是否可以形成等差数列: {result3}") # 输出: 数组 [1, 2, 3, 4, 5] 是否可以形成等差数列: True

arr4 = [10, 5, 0]
result4 = solution.canMakeArithmeticProgression(arr4)
print(f"数组 {arr4} 是否可以形成等差数列: {result4}") # 输出: 数组 [10, 5, 0] 是否可以形成等差数列: True

arr5 = [0, 0, 0]
result5 = solution.canMakeArithmeticProgression(arr5)
print(f"数组 {arr5} 是否可以形成等差数列: {result5}") # 输出: 数组 [0, 0, 0] 是否可以形成等差数列: True

arr6 = [-1, 5, 3]
result6 = solution.canMakeArithmeticProgression(arr6)
print(f"数组 {arr6} 是否可以形成等差数列: {result6}") # 输出: 数组 [-1, 5, 3] 是否可以形成等差数列: True
