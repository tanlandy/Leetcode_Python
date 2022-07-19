# Educative
## Linked List
```py
class Node:
  def __init__(self, data):
    self.data = data
    self.next = None

class LinkedList:
  def __init__(self):
    self.head = None
  
  def print_list(self):
    cur_node = self.head
    while cur_node:
      print(cur_node.data)
      cur_node = cur_node.next

  def append(self, data):
    new_node = Node(data)
    if self.head is None:
      self.head = new_node
      return
    last_node = self.head
    while last_node.next:
      last_node = last_node.next
    last_node.next = new_node

  def prepend(self, data):
    new_node = Node(data)

    new_node.next = self.head
    self.head = new_node
  
  def insert_after_node(self, prev_node, data):
    if not prev_node:
      print("Previous node does not exist.")
      return
    new_node = Node(data)

    new_node.next = prev_node.next
    prev_node.next = new_node

  
  
llist = LinkedList()
llist.append("A")
llist.append("B")
llist.append("C")


llist.insert_after_node(llist.head.next, "D")

llist.print_list()  
```

1. Swap nodes in pairs
2. Reverse
3. Merge Two Sorted Linked Lists
4. Remove duplicates


[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)

```py
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
        
        node.val = node.next.val # 把node的值变成node.next
        node.next = node.next.next # 把node.next删掉
```

## Doubly Linked List
Implementation
```py
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        if self.head is None:
            new_node = Node(data)
            self.head = new_node
        else:
            new_node = Node(data)
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = new_node
            new_node.prev = cur

    def prepend(self, data):
        if self.head is None:
            new_node = Node(data)
            self.head = new_node
        else:
            new_node = Node(data)
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node

    def print_list(self):
        cur = self.head
        while cur:
            print(cur.data)
            cur = cur.next



```



# 基础题目
[206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

```py
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        [1,2,3,4,5] -> [5,4,3,2,1]
        Use two ptrs. each time assign the cur.next to prev ptr, then move both cur and pre to their next node

        Time: O(N)
        Space: O(1)
        """
        prev, cur = None, head

        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        
        return prev
```

[876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)

```py
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        slow and fast pointers
        Time: O(N)
        Space: O(1)
        """
        slow = fast = head
        
        while fast and fast.next: # inside the while function, there is a fast.next.next, so have to check fast.next
            fast = fast.next.next
            slow = slow.next
        
        return slow
```

# 进阶题目

[160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

```py
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        Use two pointers, let them traverse at most the length of M+N when they interact at the end
        
        Time: O(M+N)
        Space: O(1)
        """
        pA, pB = headA, headB
        
        while pA != pB:
            pA = headB if pA is None else pA.next
            pB = headA if pB is None else pB.next
        
        return pA
```



## 成环问题Floyd's

[141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

```py
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """用快慢指针找到相遇的点，相遇了就说明有环"""
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        
        return False
```

[142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)


```py
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        先用快慢指针找到相遇的点。
        然后利用Floyd's算法找到成环的点；Floyd's算法是head和快慢指针相遇点往后走，彼此相遇的点就是成环的点

        时间：O(N)
        空间：O(1)
        """
        slow = fast = head
        
        # 快慢指针找相遇点，如果fast.next为空，就说明没有环
        while slow and fast:
            if not fast.next or not fast.next.next: # 最下面用到了slow.next相当于是fast.next.next.next，所以要not fast.next.next
                return None
            
            slow = slow.next
            fast = fast.next.next # 这里用到了fast.next.next，所以条件要有not fast.next
            if slow == fast:
                break

        cur = head        
        while cur:
            if cur == slow:
                return cur
            
            cur = cur.next
            slow = slow.next
```

[287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

```py
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = 0, 0
        # 找到相遇的节点
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        
        # 找到cycle的起点
        slow2 = 0
        while True:
            slow = nums[slow]
            slow2 = nums[slow2]
            if slow == slow2:
                return slow
```

[2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)
计算出来每次的数字，然后相加，最后更新指针；注意while循环的条件

时间：O(max(l1, l2))
空间：O(max(l1, l2))
```py
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1) # edge case when insert into LinkedList
        cur = dummy

        carry = 0
        while l1 or l2 or carry: # 这个条件非常重要
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            
            num = (carry + v1 + v2) % 10
            carry = (carry + v1 + v2) // 10
            
            cur.next = ListNode(num) # 形成LinkedList
            
            # 更新指针
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        
        return dummy.next

```


[92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

```py

class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        """
        [1,2,3,4,5], left = 2, right = 4
        -> [1,4,3,2,5]
        """
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

[328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/)

```py
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        create odd and even lists seperately
        link the end of odd list to the beginning of even list
        """
        if not head:
            return head
        
        odd = head
        even = head.next
        even_head = even
        
        while even and even.next: # 需要even.next; 因为odd.next = even.next，同时even.next = odd.next，也就是说
            odd.next = even.next # 1->3
            odd = odd.next # odd is 3, 3->4->5
            
            even.next = odd.next # 2->4 这里的odd.next相当于even.next.next
            even = even.next
        
        odd.next = even_head
        
        return head
```

# Other problems

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
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        [1,2,3,4,5] -> [2,1,4,3,5]
        """
        dummy = ListNode(-1)
        dummy.next = head
        pre = dummy
        
        while pre.next and pre.next.next: # 因为用到了pre.next.next.next所以pre.next.next不能为空
            first = pre.next
            second = pre.next.next # 取得接下来的2个数
            
            first.next = second.next # 1->3 # 这里的second.next相当于pre.next.next.next
            second.next = first # 2->1
            
            pre.next = second # pre_group_end->2
            pre = first # pre站到1(接下来两个是3,4)
        
        return dummy.next
```

[234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

```py
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """
        find mid, reverse second half, and compare one by one
        """
        
        # find mid
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # 这个时候slow就是中间的点
        
        # reverse second half
        pre = None
        cur = slow
        
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        # 这个时候cur走到了None，pre指向原list的最后一个点，也就是reversed second half的起点

        # compare
        first = head
        second = pre
        
        while second:
            if first.val != second.val:
                return False
            first = first.next
            second = second.next
        
        return True
```

[21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

```py
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

[23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

```py
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Heap：先把所有list的第一个node放进minHeap，然后一个一个形成res。每从minHeap弹出一个当前的最小值，就把该list的下一个值放进minHeap里

        时间：O(NlogK), N is # of final nodes, K is len(lists)
        空间：O(N) for creating final result; O(K) for creating minHeap
        """
        dummy = ListNode(-1)
        cur = dummy
        minHeap = []
        
        # add the first node of each list into the minHeap
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(minHeap, (lists[i].val, i))
                lists[i] = lists[i].next # 每个list指向第二个节点
        
        # add to res one by one
        while minHeap: 
            val, i = heapq.heappop(minHeap) # 取出来最小值
            cur.next = ListNode(val) # 新建node并连接到当前的node
            cur = cur.next # 移动ptr
            if lists[i]: # 把更新后的点加进minHeap
                heapq.heappush(minHeap, (lists[i].val, i))
                lists[i] = lists[i].next # 处理的list指向下一个节点
        
        return dummy.next

```


```py
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Divide and conquer

        Time: O(NlogK) N is len(one_list), K is num of lists
        Space: O(N) in total: one_merge take O(N); O(1) for merge_two()
        """
        # edge case
        if not lists or len(lists) == 0:
            return None
        
        while len(lists) > 1:
            one_merge = []
            
            for i in range(0, len(lists), 2): # each time the step is 2
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None # check for the odd condition
                one_merge.append(self.merge_two(l1, l2))
                
            lists = one_merge
        
        return lists[0]
    
    def merge_two(self, l1, l2):
        """
        Same as Leetcode Q21
        """
        dummy = ListNode(-1) # dummy node to avoid empty ptr
        pre = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                pre.next = l1
                l1 = l1.next
            else:
                pre.next = l2
                l2 = l2.next
            pre = pre.next # don't forget to move ptr
        
        # append the remaining nodes in l1 or l2
        if l1:
            pre.next = l1
        elif l2:
            pre.next = l2
        
        return dummy.next
```

[143. Reorder List](https://leetcode.com/problems/reorder-list/)
Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]

分两半，把第二部分reverse，然后一个一个相互连到一起

时间：O(N)
空间：O(1)
```py
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

```py
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        One pass
        ptr1 move k times, then ptr2 start from the beginning.
        When moving together, ptr1 moves n-k to the end, so ptr2 moves n-k times

        Time: O(N)
        Space: O(1)
        """
        dummy = ListNode(-1)
        dummy.next = head
        ptr1 = ptr2 = dummy
        
        while n > 0: 
            ptr1 = ptr1.next
            n -= 1
        
        ptr1 = ptr1.next # ptr1多走一步，从而之后ptr2就刚好在要移除的节点的前一个
        while ptr1:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        
        ptr2.next = ptr2.next.next
        
        return dummy.next
```

```py
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Two Pass
        走两遍，第一遍计算出来总长度，第二遍就走到对应的位置再删除就可以。因为要删除，所以要用dummy node

        时间：O(N)
        空间：O(1)
        """
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

```python
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

[83. Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)

```py
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        类似LC26，快慢指针，当不一样的时候就可以加进来
        """
        if not head:
            return head
        
        slow = fast = head
        while fast:
            if slow.val != fast.val:
                slow.next = fast
                slow = fast
            fast = fast.next
        slow.next = None # 别忘了断开后面的连接
        return head         
```

[203. Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)
```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if not head:
            return None
        
        dummy = ListNode(-1)
        dummy.next = head
        cur = head
        pre = dummy
        
        while cur:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre = cur
            cur = cur.next
        
        return dummy.next
```