class Solution:
    def intToRoman(self, num: int) -> str:
        # 定义数值与罗马符号的对应列表，按数值从大到小排序
        value_symbols = [
            (1000, 'M'),
            (900, 'CM'),
            (500, 'D'),
            (400, 'CD'),
            (100, 'C'),
            (90, 'XC'),
            (50, 'L'),
            (40, 'XL'),
            (10, 'X'),
            (9, 'IX'),
            (5, 'V'),
            (4, 'IV'),
            (1, 'I'),
        ]
        res = []  # 使用列表提高字符串拼接效率
        for value, symbol in value_symbols:
            # 当当前数值小于等于剩余数字时，重复添加符号并减去对应值
            while num >= value:
                res.append(symbol)
                num -= value
            if num == 0:
                break  # 提前终止循环
        return ''.join(res)

# 调用示例
solution = Solution()
print(solution.intToRoman(3749))  # 输出: "MMMDCCXLIX"
print(solution.intToRoman(58))    # 输出: "LVIII"
print(solution.intToRoman(1994))  # 输出: "MCMXCIV"
