class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()  # 用于记录已经出现过的数字
        while n != 1:
            if n in seen:  # 如果数字重复出现，说明进入了循环
                return False
            seen.add(n)
            # 计算下一个数字
            next_n = 0
            while n > 0:
                digit = n % 10
                next_n += digit * digit
                n = n // 10
            n = next_n
        return True


# 示例 1
n1 = 19
solution = Solution()
result1 = solution.isHappy(n1)
print(f"输入: n = {n1}, 输出: {result1}  (预期: true,  代码结果: {result1})") # 预期 true

# 示例 2
n2 = 2
result2 = solution.isHappy(n2)
print(f"输入: n = {n2}, 输出: {result2}  (预期: false, 代码结果: {result2})") # 预期 false

# 更多测试用例 (可选)
n3 = 7
result3 = solution.isHappy(n3)
print(f"输入: n = {n3}, 输出: {result3}  (预期: true,  代码结果: {result3})") # 预期 true

n4 = 4
result4 = solution.isHappy(n4)
print(f"输入: n = {n4}, 输出: {result4}  (预期: false, 代码结果: {result4})") # 预期 false
