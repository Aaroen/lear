class Solution:
    def addBinary(self, a: str, b: str) -> str:
        i = len(a) - 1  # 初始化指针i指向a的最后一个字符
        j = len(b) - 1  # 初始化指针j指向b的最后一个字符
        carry = 0       # 进位初始化为0
        result = []     # 存储每一位的结果，初始为空列表
        
        while i >= 0 or j >= 0 or carry > 0:  # 循环直到两个指针都处理完且进位为0
            digitA = int(a[i]) if i >= 0 else 0  # 获取当前a的位，如果i已到头则为0
            digitB = int(b[j]) if j >= 0 else 0  # 同上处理b的当前位
            sum_val = digitA + digitB + carry    # 当前位的总和加上进位
            current = sum_val % 2                # 当前位的结果（0或1）
            carry = sum_val // 2                # 新的进位值（总和除以2的商）
            result.append(current)               # 将当前位的结果添加到列表
            i -= 1                              # 指针左移
            j -= 1
        
        # 循环结束后，将结果反转并转为字符串
        return ''.join(str(x) for x in reversed(result))
