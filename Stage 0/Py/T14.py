class Solution(object):
    def calPoints(self, operations):
        """
        :type operations: List[str]
        :rtype: int
        """
        point = []
        new = 0
        for op in operations:
            if op == "C":
                point.pop()
            elif op == "D":
                new = point[-1] * 2
                point.append(new)
            elif op == "+":
                new = point[-1] + point[-2]
                point.append(new)
            else:
                new = int(op)
                point.append(new)

        return sum(point)
    

    # 示例代码调用
if __name__ == "__main__":
    sol = Solution()
    # 示例 1
    ops1 = ["5","2","C","D","+"]
    output1 = sol.calPoints(ops1)
    print(f"输入: ops = {ops1}, 输出: {output1} (预期: 30)")
    # 示例 2
    ops2 = ["5","-2","4","C","D","9","+","+"]
    output2 = sol.calPoints(ops2)
    print(f"输入: ops = {ops2}, 输出: {output2} (预期: 27)")
    # 示例 3
    ops3 = ["1"]
    output3 = sol.calPoints(ops3)
    print(f"输入: ops = {ops3}, 输出: {output3} (预期: 1)")