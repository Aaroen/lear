class MinStack:

    def __init__(self):
        """
        初始化堆栈对象。
        """
        self.data_stack = []  # 主栈，存储所有元素
        self.min_stack = []   # 辅助栈，存储最小值序列

    def push(self, val: int) -> None:
        """
        将元素val推入堆栈。
        """
        self.data_stack.append(val)
        # 如果 min_stack 为空 或者 val 小于等于 min_stack 栈顶元素，则将 val 也推入 min_stack
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        """
        删除堆栈顶部的元素。
        """
        if not self.data_stack: # 题目已说明 pop/top/getMin 在非空栈上调用，此处为了代码完整性可以加入判断
            return

        pop_val = self.data_stack.pop()
        # 如果弹出的元素和 min_stack 栈顶元素相等，也需要从 min_stack 弹出
        if pop_val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        """
        获取堆栈顶部的元素。
        """
        if not self.data_stack: # 题目已说明 pop/top/getMin 在非空栈上调用，此处为了代码完整性可以加入判断
            return None # 或者抛出异常，根据实际需求

        return self.data_stack[-1]

    def getMin(self) -> int:
        """
        获取堆栈中的最小元素。
        """
        if not self.min_stack: # 题目已说明 pop/top/getMin 在非空栈上调用，此处为了代码完整性可以加入判断
            return None # 或者抛出异常，根据实际需求

        return self.min_stack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin() == -3)  # True
minStack.pop()
print(minStack.top() == 0)     # True
print(minStack.getMin() == -2)  # True
