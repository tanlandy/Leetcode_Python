[206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
两个指针，一个一个修改指向

时间：O(N)
空间：O(1)
```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head

        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        
        return prev

```

[21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)


```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        pre = dummy
        
        while list1 and list2:
            if list1.val <= list2.val:
                pre.next = list1
                list1 = list1.next
            else:
                pre.next = list2
                list2 = list2.next
            pre = pre.next
        
        if list1:
            pre.next = list1
        elif list2:
            pre.next = list2
        
        return dummy.next
```

[143. Reorder List](https://leetcode.com/problems/reorder-list/)
Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]

分两半，把第二部分reverse，然后一个一个相互练到一起

时间：O(N)
空间：O(1)
```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        # find middle
        slow, fast = head, head,next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        second = slow.next # 第二部分的起点
        slow.next = None # 不要忘记分隔开

        # reverse second half
        prev = None
        while second:
            nxt = second.next
            second.next = prev
            prev = second
            second = nxt
        
        # merge two halfs
        second = prev # prev是最后一个点
        first = head

        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first = tmp1
            second = tmp2



```