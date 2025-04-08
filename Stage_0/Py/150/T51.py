# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

from typing import Optional

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next: # 链表为空或者只有一个节点，不可能有环
            return False

        slow = head
        fast = head.next # 快指针初始时比慢指针快一步，这样在环的起始位置就能更快开始追逐

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True

        return False


# 辅助函数：根据列表创建链表 (用于测试)
def create_linked_list(nodes: list, pos: int = -1) -> Optional[ListNode]:
    if not nodes:
        return None
    head = ListNode(nodes[0])
    current = head
    node_list = [head] # 保存所有节点，方便环的连接
    for i in range(1, len(nodes)):
        new_node = ListNode(nodes[i])
        current.next = new_node
        current = new_node
        node_list.append(new_node)

    if pos != -1: # 创建环
        current.next = node_list[pos] # 尾节点指向 pos 位置的节点

    return head

# 调用示例 1
head1 = create_linked_list([3, 2, 0, -4], pos=1)
solution = Solution()
result1 = solution.hasCycle(head1)
print(f"示例 1 结果: {result1} (期望: True) - 判断结果: {result1 == True}")

# 调用示例 2
head2 = create_linked_list([1, 2], pos=0)
result2 = solution.hasCycle(head2)
print(f"示例 2 结果: {result2} (期望: True) - 判断结果: {result2 == True}")

# 调用示例 3
head3 = create_linked_list([1], pos=-1)
result3 = solution.hasCycle(head3)
print(f"示例 3 结果: {result3} (期望: False) - 判断结果: {result3 == False}")

# 调用示例 4: 没有环的链表
head4 = create_linked_list([1, 2, 3, 4], pos=-1)
result4 = solution.hasCycle(head4)
print(f"示例 4 结果: {result4} (期望: False) - 判断结果: {result4 == False}")

# 调用示例 5: 空链表
head5 = create_linked_list([], pos=-1)
result5 = solution.hasCycle(head5)
print(f"示例 5 结果: {result5} (期望: False) - 判断结果: {result5 == False}")
