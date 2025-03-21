# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
from typing import Optional
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        stk1 = []
        stk2 = []

        while l1:
            stk1.append(l1.val)
            l1 = l1.next
        while l2:
            stk2.append(l2.val) 
            l2 = l2.next

        carr = 0
        head = None

        while stk1 or stk2 or carr :
            sum = carr
            if stk1 :
                sum += stk1.pop()
            if stk2 :
                sum += stk2.pop()
            carr = sum // 10
            sum = sum % 10
            curr = ListNode(sum)
            curr.next = head
            head = curr

        return head