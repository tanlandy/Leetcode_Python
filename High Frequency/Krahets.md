# Krahets精选题

[Krahets精选88题](https://leetcode.cn/studyplan/selected-coding-interview/)

# 链表

[21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        用一个虚拟头节点，不停往后接上list1和list2的节点，最后返回头节点即可
        当你需要创造一条新链表的时候，可以使用虚拟头结点简化边界情况的处理。
        """
        dummy = ListNode(-1)
        pre = dummy

        while list1 and list2:
            if list1.val < list2.val:
                pre.next = list1
                list1 = list1.next
            else:
                pre.next = list2
                list2 = list2.next
            pre = pre.next
        
        pre.next = list1 if list1 else list2
        
        return dummy.next

```

[206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur, pre = head, None

        while cur:

            nxt = cur.next  # 先把后面的存下来，因为改变.next之后就找不到了
            cur.next = pre
            pre = cur
            cur = nxt
        
        return pre
```

[86. 分隔链表](https://leetcode.cn/problems/partition-list/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        """
        新建2个链表。把比x小的放第一个，其他的放在第二个，最后把两个链表连接起来
        """
        # 要提前存2个头，一个用来链接，另一个用来return
        before = before_head =  ListNode(-1)
        after = after_head = ListNode(-1)
        
        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next
        
        after.next = None  # 这一步很重要，不然会有环
        before.next = after_head.next
        
        return before_head.next

```


[237. Delete Node in a Linked List](https://leetcode.cn/problems/delete-node-in-a-linked-list/description/?envType=study-plan-v2&envId=selected-coding-interview)

```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        实际上是删掉了node.next这个node
        """
        
        node.val = node.next.val  # 把node的值变成node.next
        node.next = node.next.next  # 把node.next删掉
```

[138. Copy List with Random Pointer](https://leetcode.cn/problems/copy-list-with-random-pointer/description/?envType=study-plan-v2&envId=selected-coding-interview)

```Python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        Two Passes: 
        第一遍只复制node，不管指针，形成一个map{old : new}；
        第二遍把node的指针连起来；
        注意连的map里没考虑最后是None的情况，所以一开始map={ None : None}；遍历是while cur

        时间：O(N)
        空间：O(N)
        """
        old_new = {None: None}  # 为了在复制的时候，如果cur.next是None， copy.next也可以是None

        cur = head
        while cur:
            new = Node(cur.val)
            old_new[cur] = new
            cur = cur.next
        
        cur = head
        while cur:
            new = old_new[cur]
            new.next = old_new[cur.next]
            new.random = old_new[cur.random]
            cur = cur.next

        return old_new[head]   

```