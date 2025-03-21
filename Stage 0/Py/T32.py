# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
from typing import Optional

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode(0)
        curr = head
        carr = 0

        while l1 or l2 or carr :
            val1 = l1.val if l1 else 0  # 如果 l1 不为空，取当前节点值，否则取 0
            val2 = l2.val if l2 else 0  # 如果 l2 不为空，取当前节点值，否则取 0
            sum = val1 + val2 + carr # 使用 val1 和 val2 计算 sum_val
            carr = sum // 10
            res  = sum % 10
            curr.next = ListNode(res)
            curr = curr.next
            if l1: l1 = l1.next
            if l2: l2 = l2.next

        return head.next