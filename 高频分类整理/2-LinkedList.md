[206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
两个指针，一个一个得修改指向

时间：O(N)
空间：O(1)
```py
"""
[1,2,3,4,5] -> [5,4,3,2,1]
"""
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

[92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

```py
"""
[1,2,3,4,5], left = 2, right = 4
-> [1,4,3,2,5]
"""
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if not head:
            return head
        
        dummy = ListNode(-1)
        dummy.next = head
        left_prev = dummy
        cur = head
        
        # 找到起点，以及起点之前的哪个点
        for i in range(left - 1):
            left_prev = left_prev.next
            cur = cur.next
        
        # 在left和right之间reverse
        prev = None
        for i in range(right - left + 1):
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        
        # 把整个连接起来
        left_prev.next.next = cur # left_prev = 1, 把2->5
        left_prev.next = prev # 把1->4
        
        return dummy.next

```

[25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)


```py
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(-1)
        dummy.next = head
        
        # 找到group size==k的终点
        def getKth(cur, k):
            while cur and k > 0:
                cur = cur.next
                k -= 1
            return cur
        
        group_prev = dummy
        while True:
            # 获得这次要reverse的group的终点
            kth = getKth(group_prev, k)
            if not kth:
                break
            
            group_next = kth.next # 存下来这个group交换之后的终点的下一个点
            prev, cur = kth.next, group_prev.next # prev是下一个group的起点， cur是这个group的起点
            
            while cur != group_next: # 当cur走到这个group的终点
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt
            
            # 把2个group连接起来
            tmp = group_prev.next 
            group_prev.next = kth
            group_prev = tmp
        
        return dummy.next
```

[24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

```py
"""
[1,2,3,4,5] -> [2,1,4,3,5]
"""
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        dummy.next = head
        pre = dummy
        
        while pre.next and pre.next.next: # 因为用到了pre.next.next.next所以pre.next.next不能为空
            first = pre.next
            second = pre.next.next # 取得接下来的2个数
            
            first.next = second.next # 1->3
            second.next = first # 2->1
            
            pre.next = second # pre_group_end->2
            pre = first # pre站到1(接下来两个是3,4)
        
        return dummy.next
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

[19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
走两遍，第一遍计算出来总长度，第二遍就走到对应的位置再删除就可以。因为要删除，所以要用dummy node

```py
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(-1)
        length = 0
        dummy.next = head
        cur = dummy
        
        while cur:
            cur = cur.next
            length += 1

        target = length - n - 1
        
        cur = dummy
        while target > 0:
            cur = cur.next
            target -= 1
        
        cur.next = cur.next.next
        
        return dummy.next

```


[138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

Two Passes: 第一遍只复制node，不管指针，形成一个map{old : new}；第二遍把node的指针连起来；注意连的map里没考虑最后是None的情况，所以一开始map={ None : None}；遍历是while cur

时间：O(N)
空间：O(N)

```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        oldToCopy = {None : None} # 为了在复制的时候，如果cur.next是None， copy.next也可以是None

        cur = head
        while cur:
            copy = Node(cur.val)
            oldToCopy[cur] = copy
            cur = cur.next
        
        cur = head
        while cur:
            copy = oldToCopy[cur]
            copy.next = oldToCopy[cur.next]
            copy.random = oldToCopy[cur.random]
            cur = cur.next

        return oldToCopy[head]
```

