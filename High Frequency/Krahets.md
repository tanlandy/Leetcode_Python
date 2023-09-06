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

# 栈与队列

[20. Valid Parentheses](https://leetcode.cn/problems/valid-parentheses/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def isValid(self, s: str) -> bool:
        """
        when open: append to stack
        when close: use a dictionary to map close to open, and check if new one matches the top, if so pop the old
        """
        stack = []
        close_open = { ")":"(", "]":"[", "}":"{" } # 非常好的解决比对好几次的情况

        for c in s:
            if c in "({[":
                stack.append(c)
            elif stack and stack[-1] == close_open[c]: # 比较栈顶，总是要看看stack是否为空
                stack.pop()
            else:
                return False
        
        return stack == []  # 最后要检查stack是否为空
```

[155. Min Stack](https://leetcode.cn/problems/min-stack/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class MinStack:
    """
    用两个stack就可以，重点是min_stack的插入数据方式：每次先找到最小值，然后放入
    删除时候都删
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val) # check if min_stack is empty
        self.min_stack.append(min_val) # each time add the minimum value into min_stack

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]

```

[232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class MyQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
        """to push, only push into stack1"""
        self.stack1.append(x)

    def pop(self) -> int:
        # """to pop, get from stack2; if empty, push the entire stack1 into stack2"""
        # if not self.stack2:
        #     while self.stack1:
        #         self.stack2.append(self.stack1.pop())
        # return self.stack2.pop()
        """直接peek一下把stack1压栈到stack2，然后返回最上面的那个值"""
        self.peek()  
        return self.stack2.pop()

    def peek(self) -> int:
        """to peek, get from stack2; if empty, push the entire stack1 into stack2"""
        # 当且只有stack2为空的时候，才把stack1全部压栈到stack2，从而保证顺序的一致
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self) -> bool:
        if self.stack1 or self.stack2:
            return False
        return True
```

[394. 字符串解码](https://leetcode.cn/problems/decode-string/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def decodeString(self, s: str) -> str:
        """
        When we hit an open bracket, we know we have parsed k for the contents of the bracket, so 
        push (current_string, k) to the stack, so we can pop them on closing bracket to duplicate
        the enclosed string k times.
        """
        stack = []
        res = ""
        k = 0

        for char in s:
            if char == "[":
                # Just finished parsing this k, save current string and k for when we pop
                stack.append((res, k))
                # Reset current_string and k for this new frame
                res = ""  # res一旦遇到[就全新开始
                k = 0
            elif char == "]":
                # We have completed this frame, get the last current_string and k from when the frame 
                last_string, last_k = stack.pop(-1)
                res = last_string + last_k * res
            elif char.isdigit():
                k = k * 10 + int(char)
            else:
                res += char

        return res
```

[295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class MedianFinder:

    def __init__(self):
        # A保存较大的一半，B保存较小的一半，同时A size比B大1或和B相等
        self.A, self.B = [], []

    def addNum(self, num: int) -> None:
        # B[0]是较小的一半的最大值，只要是和B有关的操作，都要取反
        if len(self.A) != len(self.B):  # 这个数加到B
            heappush(self.A, num)
            heappush(self.B, -heappop(self.A))
        else:  # 这个数加到A
            heappush(self.B, -num)
            heappush(self.A, -heappop(self.B))

    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0
```


# 哈希表

[242. 有效的字母异位词](https://leetcode.cn/problems/valid-anagram/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)

```

[387. 字符串中的第一个唯一字符](https://leetcode.cn/problems/first-unique-character-in-a-string/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def firstUniqChar(self, s: str) -> int:
        """遍历两遍。第一遍构建频数，第二遍找第一个是1的索引"""
        ch_num = collections.defaultdict(int)
        for ch in s:
            ch_num[ch] += 1
        for idx, ch in enumerate(s):
            if ch_num[ch] == 1:
                return idx
        return -1

```

[205. 同构字符串](https://leetcode.cn/problems/isomorphic-strings/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        s_t = dict()
        used = set()  # 用used保证ch_t是唯一的

        for i in range(len(s)):
            ch_s = s[i]
            ch_t = t[i]
            if ch_s not in s_t:
                if ch_t in used:
                    return False
                s_t[ch_s] = ch_t
                used.add(ch_t)
            else:
                if s_t[ch_s] != ch_t:
                    return False
        
        return True

```

[266. Palindrome Permutation](https://leetcode.com/problems/palindrome-permutation/editorial/)

```python

class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        freqs = Counter(s)
        seen_odd = False

        for count in freqs.values():
            if count % 2 == 1:
                if seen_odd:
                    return False
                else:
                    seen_odd = True

        return True

```


[409. 最长回文串](https://leetcode.cn/problems/longest-palindrome/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def longestPalindrome(self, s: str) -> int:
        freqs = Counter(s)
        res = 0
        odd = 0

        for count in freqs.values():
            remainder = count % 2
            res += count - remainder  # 偶数直接加，奇数则-1再加
            if remainder == 1:
                odd = 1  # 奇数多出来的可以放在正中间

        return res + odd

```
