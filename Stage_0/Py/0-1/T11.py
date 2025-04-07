class Solution:
    def romanToInt(self, s: str) -> int:
        roma_nums = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        x = roma_nums[s[len(s)-1]]
        for i in range(len(s)-2,-1,-1):
            if roma_nums[s[i]] < roma_nums[s[i+1]]:
                x -= roma_nums[s[i]]
            else:
                x += roma_nums[s[i]]
        return x

# 创建 Solution 类的实例
solution = Solution()

# 测试用例
roman_numerals = ["III", "IV", "IX", "LVIII", "MCMXCIV"]

# 循环遍历测试用例并打印结果
for roman in roman_numerals:
    integer_value = solution.romanToInt(roman)
    print(f"罗马数字: {roman}, 整数: {integer_value}")
