# Krahets精选题
Pomotroid
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
        freqs = Counter(s)

        for idx, ch in enumerate(s):
            if freqs[ch] == 1:
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

# 双指针


[392. 判断子序列](https://leetcode.cn/problems/is-subsequence/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        p1 = p2 = 0
        while p1 < len(s) and p2 < len(t):
            if s[p1] == t[p2]:
                p1 += 1
                p2 += 1
            else:
                p2 += 1
        
        return p1 == len(s)
```

[876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        return slow

```

[160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pA, pB = headA, headB
        
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        
        return pA

```

[167. 两数之和 II - 输入有序数组](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1

        while l < r:
            cur_sum = numbers[l] + numbers[r]
            if cur_sum < target:
                l += 1
            elif cur_sum > target:
                r -= 1
            else:
                return [l + 1, r + 1]

```

[142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        先用快慢指针找到相遇的点。
        然后利用Floyd's算法找到成环的点；Floyd's算法是head和快慢指针相遇点往后走，彼此相遇的点就是成环的点
        相同时间走的距离：slow: a + b  fast: a + k * circle + b
        走的距离有倍数关系：2 * (a + b) = a + k * circle + b
        want a
        等式化简得到 a = k * circle - b
        此时cur=head和slow继续走，相遇时，cur走过的距离就是等式左边a，slow走过的距离就是等式右边k * circle - b
        
        时间：O(N)
        空间：O(1)
        """
        slow = fast = head
        
        # 快慢指针找相遇点，如果fast或fast.next为空，就说明没有环
        while 1:
            if not fast or not fast.next:
                return
            slow = slow.next
            fast = fast.next.next 
            if slow == fast:
                break
                
        cur = head
        while cur != slow:
            cur = cur.next
            slow = slow.next  
        
        return cur

```

[151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()

        i = j = len(s) - 1  # 从右往左的同向双指针
        res = []
        while i >= 0:
            while i >= 0 and s[i] != ' ':  # 从右往左找空格
                i -= 1
            res.append(s[i + 1: j + 1])  # 左闭右开区间
            while i >= 0 and s[i] == ' ':
                i -= 1
            j = i
        return ' '.join(res)

```

[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        用set记录当前sliding window的数据；如果s[r]在set里，移动窗口直到不在并且在set中删去

        时间：O(N)
        空间：O(N)
        """
        chars = set()
        l = r = 0
        res = 0

        while r < len(s):
            while s[r] in chars:
                chars.remove(s[l])
                l += 1
            chars.add(s[r])
            res = max(res, r - l + 1)
            r += 1

        return res
```

[15. 三数之和](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        排序之后固定一个指针，用双指针
        Time: O(N^2)
        Space: O(logN), depending on how to sort
        """
        nums.sort()
        res = []

        for i, n in enumerate(nums):
            if i > 0 and nums[i - 1] == nums[i]:  # avoid duplicate
                continue

            if nums[i] > 0:  # 如果最小的数都>0，那肯定没有能够满足的结果
                break
            
            l, r = i + 1, len(nums) - 1
            while l < r:
                cur_sum = nums[i] + nums[l] + nums[r]
                if cur_sum < 0:
                    l += 1
                elif cur_sum > 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    # keep moving to find other possibilities
                    l += 1
                    r -= 1
                    while l < r and nums[l - 1] == nums[l]:  # avoid duplicate
                        l += 1
            
        return res
```

[239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        单调递减栈
        时间：O(N)
        空间：O(N)
        """
        res = []
        queue = deque()  # queue存index, nums[queue[0]]总是窗口的最大值
        l = r = 0

        while r < len(nums):
            while queue and nums[r] > nums[queue[-1]]:  # 比较次大值，保证是单调递减栈
                queue.pop()
            queue.append(r)
        
            if l > queue[0]:  # 便于比较queue[0]对应的数是否还在窗口里面
                queue.popleft()  # 如果不在，那就把queue[0]删除掉，此时次大值queue[1]变成最大值
            
            if r - l + 1 == k:
                res.append(nums[queue[0]])
                l += 1
            r += 1
        
        return res
```

# 查找

[704. 二分查找](https://leetcode.cn/problems/binary-search/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        
        return -1

```

[278. 第一个错误的版本](https://leetcode.cn/problems/first-bad-version/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def firstBadVersion(self, n: int) -> int:
        # [True, True, True, False, False, False, False]
        # Find the first False version, which is the most left valid one
        l, r = 1, n

        while l <= r:
            mid = (l + r) // 2
            if isBadVersion(mid):
                r = mid - 1
            else:
                l = mid + 1
        
        return l

```

[724. 寻找数组的中心下标](https://leetcode.cn/problems/find-pivot-index/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        r_sum = sum(nums)
        l_sum = 0

        for i in range(len(nums)):
            r_sum -= nums[i]
            if l_sum == r_sum:
                return i
            l_sum += nums[i]  # 因为比较时候不包括nums[i]，所以先比较，再加到l_sum
        
        return -1

```

[287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # 建立一个 idx -> nums[idx] -> nums[nums[idx]] 的映射，因为存在重复的数，所以会有两个数都跳转到同一个idx下，从而在第二次跳转到哪个idx时出现循环
        # 如果数组中有重复的数，以数组 [1,3,4,2,2] 为例,我们将数组下标 n 和数 nums[n] 建立一个映射关系 f(n)f(n)f(n)， 其映射关系 n->f(n) 为： 0->1 1->3 2->4 3->2 4->2 同样的，我们从下标为 0 出发，根据 f(n)f(n)f(n) 计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推产生一个类似链表一样的序列。 0->1->3->2->4->2->4->2->

        slow, fast = 0, 0
        slow = nums[slow]
        fast = nums[nums[fast]]

        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]

        pre = 0
        while pre != slow:
            pre = nums[pre]
            slow = nums[slow]

        return pre 

```

[153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/)

```py
class Solution:
    def findMin(self, nums: List[int]) -> int:
        """
        最小值一直往右都小于等于nums[-1]，相当于找最左满足的数

        时间：O(logN)
        空间：O(1)
        """
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2
            if nums[mid] <= nums[-1]:
                r = mid - 1
            else:
                l = mid + 1
        
        return nums[l]
```

```python

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < nums[r]:  # m右侧有序
                r = mid
            else:  # m左侧有序
                l = mid + 1
        
        return nums[l-1]  # 因为l=r+1，所以l-1是最小值

```

[154. 寻找旋转排序数组中的最小值 II](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < nums[r]:  # m右侧有序
                r = mid
            elif nums[mid] == nums[r]:  # 不知道哪边有序
                r -= 1
            else:  # m左侧有序
                l = mid + 1
        return nums[l]

```

# 搜索

[102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = collections.deque([root])
        res = []

        while queue:
            size = len(queue)
            level_res = []
            for _ in range(size):
                node = queue.popleft()
                level_res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level_res)
        
        return res

```

[103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        反转list: oneRes.reverse()；翻转isOdd: isOdd = not isOdd
        """
        if not root:
            return []
        odd = True
        res = []
        queue = collections.deque([root])

        while queue:
            one_res = []
            size = len(queue)

            for _ in range(size):
                node = queue.popleft()
                one_res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            if not odd:
                one_res.reverse()
            odd = not odd
            res.append(one_res)

        return res

```

[236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        解题思路：每个节点要知道什么、做什么：什么时候做
        遍历or递归
        要知道自己的子树里是否有这两个数字->递归
        要做什么：返回自己子树是否有这两个数字->递归
        什么时候做：后序遍历，传递子树信息

        自下而上，这个函数就返回自己左右子树满足条件的node：返回自己或者不为None的一边。base case就是找到了
        如果一个节点能够在它的左右子树中分别找到 p 和 q，则该节点为 LCA 节点。

        时间：O(N)
        空间：O(N)
        """
        if root is None: # base case 走到了根节点
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # 后序遍历
        if root == p or root == q: # Case 1：公共祖先就是我自己，也可以放在前序位置（要确保p,q在树中）
            return root
        
        if left and right: # Case 2：自己子树包含这两个数
            return root
        else:
            return left or right # Case 3：其中一个子树包含节点 
```

[235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        不需要去遍历子树，由于 BST 左小右大的性质，将当前节点的值与 val1 和 val2 作对比即可判断当前节点是不是 LCA

        Time: O(H)
        Space: O(1)
        """
        cur = root
        
        while cur:
            # cur太小就往右
            if p.val > cur.val and q.val > cur.val:
                cur = cur.right
            # cur太大就往左
            elif p.val < cur.val and q.val < cur.val:
                cur = cur.left
            else: # p.val <= cur.val <= q.val
                return cur
```

[230. 二叉搜索树中第K小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        res = []

        def dfs(root):
            if not root:
                return
            dfs(root.left)
            res.append(root.val)
            if len(res) == k:
                return 
            dfs(root.right)
        
        dfs(root)
        return res[-1]

```


[426. Convert Binary Search Tree to Sorted Doubly Linked List](https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)

```python

class Solution:
    def treeToDoublyList(self, root: 'Optional[Node]') -> 'Optional[Node]':
        """
        中序遍历
        因为建立了新链表，所以使用dummy避免边界问题
        """
        if not root:
            return root

        dummy = Node(-1)
        pre = dummy

        def inorder(node):
            nonlocal pre

            if not node:
                return

            inorder(node.left)
            
            node.left = pre  # 把两个节点连接起来
            pre.right = node
            pre = node  # 站到自己这个节点，即中序遍历的上一个节点
            
            inorder(node.right)
        
        inorder(root)
        dummy.right.left = pre  # dummy.right是个node，就是有着最小值的node
        pre.right = dummy.right

        return dummy.right
```

[104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        分治：从下到上返回当前节点的最大深度
        """
        if not root:
            return 0
        left_max = self.maxDepth(root.left)
        right_max = self.maxDepth(root.right)
        
        return 1 + max(left_max, right_max)
```

[226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return

        node = root.left
        root.left = self.invertTree(root.right)
        root.right = self.invertTree(node)

        return root

```

[101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        对于每个节点来说：看自己的左右节点是否对称，看自己的子树是否对称->返回自己是否满足对称
        遍历一遍不可以，需要知道自己的子节点是否对称这一信息
        -> 递归，同时看两个节点，然后左左右右，左右右左看对称
        """
        if not root:
            return True

        def dfs(left, right):
            if not left and not right:
                return True
            
            if not left or not right or left.val != right.val:
                return False
            
            return dfs(left.right, right.left) and dfs(left.left, right.right)
        
        return dfs(root, root)
```

[110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """
        站在每个节点：知道两边子树的高度差，并比较；
        返回什么：要返回当前节点的高度
        -> 后序遍历，返回当前高度

        时间：O(N)
        空间：O(N)
        """
        def node_height(node):
            if not node:
                return 0
            left_h = node_height(node.left)
            right_h = node.height(node.right)

            if left_h == -1 or right_h == -1:
                return -1

            if abs(left_h - right_h) > 1:
                return -1
            
            return 1 + max(left_h, right_h)
        
        return node_height(root) != -1
```

[113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """
        execute the dfs and maintain the running sum of node traversed and the list of those nodes
        
        Time: O(N^2)
        Space: O(N)
        需要注意，如果是问root to node的话，就不需要满足位置关系。
        如果不全是positive value的话，不能提前break，一定要找到底
        """
        res = []
        def dfs(node, cur_sum, cur_path):
            if not node:
                return
            cur_sum += node.val
            cur_path.append(node.val)
            
            if cur_sum == targetSum and not node.left and not node.right: # 同时满足大小和位置关系
                res.append(cur_path.copy())
            else:
                dfs(node.left, cur_sum, cur_path)
                dfs(node.right, cur_sum, cur_path)
            
            cur_path.pop()
        
        dfs(root, 0, [])
        return res
```

[105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        """
        Preorder的第一个是root，第二个数是左子树的root
        Inorder的root左边的值都在左子树，root右边的都是右子树
        时间：O(N)
        空间：O(N)
        """
        # base case
        if not preorder or not inorder:
            return None
        
        root = TreeNode(preorder[0])
        mid = inorder.index(root.val) # 找到root在inorder的index

        # preorder：根据左子树的数量，root之后[1:mid+1]左闭右开都是左子树，[mid+1:]都是右子树
        # inorder的root左边都是左子树，右边都是右子树   
        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid]) # 右开
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])

        return root

```

[297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Codec:

    def serialize(self, root):
        """
        用一个list记录，最后转为string导出：
        前序遍历，空节点计作N，然后用","连接
        """
        res = []
        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        return ",".join(res)

    def deserialize(self, data):
        """
        先确定根节点 root，然后遵循前序遍历的规则，递归生成左右子树
        """
        vals = data.split(",")
        self.i = 0
        
        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node
        
        return dfs()
```

[509. 斐波那契数](https://leetcode.cn/problems/fibonacci-number/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

```

[70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def climbStairs(self, n: int) -> int:
        """
        bottom-up DP，从最后往最前面
        站在最后一部，思考如何计算，从而找到递推表达式f(n) = f(n - 1) + f(n - 2)
        """
        if n == 1:
            return 1
        dp = [0] * (n + 1)  # dp[i] is the number of ways to climb to i
        dp[1] = 1  # base case
        dp[2] = 2  # base case
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]  # recurrence relation          
        
        return dp[n]
```

```py
class Solution:
    def climbStairs(self, n: int) -> int:
        """
        可以不把所有的中间结果存下来，只用两个变量来记录

        时间：O(N)
        空间：O(1)
        """
        one = two = 1
        
        for i in range(n - 1):
            tmp = one
            one = one + two
            two = tmp
        
        return one
```        

[1480. 一维数组的动态和](https://leetcode.cn/problems/running-sum-of-1d-array/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        dp = [0] * len(nums)
        dp[0] = nums[0]

        for i in range(1, len(dp)):
            dp[i] = dp[i - 1] + nums[i]
        return dp

```

[121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """对于每一个price来说，最大收益是当前price-之前的最小收益"""
        cost, profit = float('inf'), 0

        for price in prices:
            cost = min(price, cost)
            profit = max(profit, price - cost)
        
        return profit

```

[122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """只要今天相对于昨天有收益，昨天就买入"""
        profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i - 1]
            if tmp > 0:
                profit += tmp
        return profit

```

[64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        dp[r][c] is the minumum path sum to (r, c)
        dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1])
        
        Time: O(M*N)
        Space: O(M*N)
        """
        
        rows, cols = len(grid), len(grid[0])
        
        dp = [[0 for _ in range(cols)] for _ in range(rows)]
        
        dp[0][0] = grid[0][0]
        for r in range(1, rows):
            dp[r][0] = grid[r][0] + dp[r-1][0]
        for c in range(1, cols):
            dp[0][c] = grid[0][c] + dp[0][c-1]
            
        for r in range(1, rows):
            for c in range(1, cols):
                dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1])
        
        return dp[-1][-1]
```

[53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp[i]: 以i结尾的最大子数组和
        # 最终返回值是max(dp)

        dp = [float('-inf') for _ in range(len(nums))]

        dp[0] = nums[0]
        for i in range(1, len(dp)):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])  # 根据递推关系，找到dp的定义

        return max(dp)

```

[198. 打家劫舍](https://leetcode.cn/problems/house-robber/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def rob(self, nums: List[int]) -> int:
        # dp[i] 表示前i个房间，能偷到的最大金额
        # 最终结果就是dp[-1]
        if len(nums) == 1:
            return nums[0]

        dp = [0 for _ in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = max(dp[0], nums[1])

        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        return dp[-1]

```

[213. 打家劫舍 II]

```python

class Solution:
    def rob(self, nums: List[int]) -> int:
        """跑两遍"""
        def rob_single(nums):
            if len(nums) == 1:
                return nums[0]
            dp = [0 for _ in range(len(nums))]
            dp[0] = nums[0]
            dp[1] = max(dp[0], nums[1])

            for i in range(2, len(dp)):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
            return dp[-1]
        
        if len(nums) == 1:
            return nums[0]

        return max(rob_single(nums[:-1]), rob_single(nums[1:]))

```

[300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        dp[i]: 以s[i]为结尾的，最长子序列的长度
        res: max(dp)
        Time: O(N^2)
        Space: O(N)
        """
        dp = [1 for _ in range(len(nums))]

        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:  # nums[i]可以接在nums[j]之后
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
```


# 回溯

[46. 全排列](https://leetcode.cn/problems/permutations/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        满树问题，剪枝条件是跳过重复使用的值：可以用used[]来记录使用过的值，也可以每次判断nums[i] in one_res
        """
        res = []
        used = [False] * len(nums)

        def backtrack(one_res):
            # 添加条件： 长度
            if len(one_res) == len(nums):
                res.append(one_res.copy())
                return
            
            # 满树问题：i从0开始
            for i in range(len(nums)):
                # 跳过不合法的选择，否则结果有[1,1,1],[1,1,2]...
                if used[i]:
                    continue
                
                used[i] = True
                one_res.append(nums[i])
                backtrack(one_res)
                one_res.pop()
                used[i] = False
        
        backtrack([])
        return res
```

[47. 全排列 II](https://leetcode.cn/problems/permutations-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        used = [False] * len(nums)
        
        def backtrack(one_res):
            if len(one_res) == len(nums):
                res.append(one_res.copy())
                return
        
            for i in range(len(nums)):
                if used[i]:
                    continue
                    
                # 新添加的剪枝逻辑，固定相同的元素在排列中的相对位置
                # not used[i-1]保证相同元素在排列中的相对位置保持不变。
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                    
                used[i] = True
                one_res.append(nums[i])
                backtrack()
                one_res.pop()
                used[i] = False
        
        backtrack([])
        return res
```


[39. 组合总和](https://leetcode.cn/problems/combination-sum/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort() # 元素可重复，所以要排序
        
        def backtrack(one_res, start, target):
            if target < 0:
                return
            
            if target == 0:
                res.append(one_res.copy())
                return
            
            for i in range(start, len(candidates)):
                # 与LC40不同点：不怕重复，所以不需要选择条件
                one_res.append(candidates[i])
                target -= candidates[i]
                backtrack(i, target) # 与LC40不同点，可以重复，所以从i再用一次
                one_res.pop()
                target += candidates[i]
        
        backtrack([], 0, target)
        return res
```

[40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """类似LC90，只是base case不同"""
        candidates.sort() # 元素可以重复，所以要排序
        res = []
        
        def backtrack(one_res, start, target):
            if target < 0:
                return
            
            if target == 0:
                res.append(one_res.copy())
            
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                
                one_res.append(candidates[i])
                target -= candidates[i]
                backtrack(one_res, i + 1, target)  # 从i+1开始，因为每个元素只能用一次
                one_res.pop()
                target += candidates[i]
        
        backtrack([], 0, target)
        return res

```





# 贪心

[240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)


```py
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        start from top-right to bottom-left
        
        Time: O(M+N)
        Space: O(1)
        """
        rows, cols = len(matrix), len(matrix[0])
        r, c = 0, cols - 1
        
        while r < rows and c >= 0:
            if matrix[r][c] == target:
                return True
            elif matrix[r][c] < target:
                r += 1
            else:
                c -= 1
        
        return False
```

[11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def maxArea(self, height: List[int]) -> int:
        """
        Time: O(N)
        Space: O(1)
        """
        l, r = 0, len(height) - 1
        area = 0
        
        while l < r:
            cur_area = (r - l) * min(height[l], height[r])
            area = max(area, cur_area)
            
            # 移动短边，消去的面积组合肯定比当前的小
            if height[l] < height[r]: # move the smaller edge, to make the area larger
                l += 1
            else:
                r -= 1
        
        return area
```

[179. 最大数](https://leetcode.cn/problems/largest-number/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        """
        use a comparator when sorting the nums

        Time: O(NlogN)
        Space: O(N)
        """

        def cmp_func(x, y):
            """
            Sorted by value of concatenated string increasingly.
            For case [3, 30], will return 330 instead of 303
            """
            if x + y > y + x:
                return 1
            elif x == y:
                return 0
            else:
                return -1

        # Build nums contains all numbers in the String format.
        nums = [str(num) for num in nums]

        # Sort nums by cmp_func decreasingly.
        nums.sort(key=cmp_to_key(cmp_func), reverse=True)

        res = "0" if nums[0] == "0" else "".join(nums)
        return res
```

[135. 分发糖果](https://leetcode.cn/problems/candy/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def candy(self, ratings: List[int]) -> int:
        """
        A在B左边
        左规则：当Rating_A < Rating_B时，A糖果需要 < B
        右规则：当Rating_A > Rating_B时，A糖果需要 > B
        从左右根据左右规则各遍历一次，取各自的最大值

        时间：O(N)
        空间：O(N)
        """
        n = len(ratings)
        left = [1] * n
        right = left.copy()

        # 左规则
        for i in range(1, n):
            if ratings[i-1] < ratings[i]:
                left[i] = left[i-1] + 1
        count = left[-1]  # basecase

        # 右规则
        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                right[i] = right[i+1] + 1
            count += max(left[i], right[i])  # 取两者最大值
        return count

```

[768. 最多能完成排序的块 II](https://leetcode.cn/problems/max-chunks-to-make-sorted-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        """单调栈stack存入每个块的最大值，最终stack的大小则是块的数量

        时间：O(N)
        空间：O(N)
        """
        stack = []
        for num in arr:
            if stack and num < stack[-1]:  # 只要当前的数字比栈顶小，不得不进行合并
                head = stack.pop()
                while stack and num < stack[-1]:  # 合并块
                    stack.pop()  
                stack.append(head)  # 新排序块的最大值还是head
            else:  # 当前数字比栈顶大或相等，新建排序块
                stack.append(num)  # 新排序块
        return len(stack)

```


# 位运算

[191. 位1的个数](https://leetcode.cn/problems/number-of-1-bits/description/?envType=study-plan-v2&envId=selected-coding-interview)

`a`           = 1 0 1 0 1 0 0 0
`a - 1`       = 1 0 1 0 0 1 1 1  # 把a最后一个1变成0，最后一个1的后面的0变成1
`a & (a - 1)` = 1 0 1 0 0 0 0 0  # 把a最后一个1变成0，其他不变

```python

class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res += 1
            n = n & (n - 1)
        return res

```

[231. 2 的幂](https://leetcode.cn/problems/power-of-two/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        """把最后一个1变成0之后，就是0了"""
        return n > 0 and n & (n - 1) == 0

```

[371. 两整数之和](https://leetcode.cn/problems/sum-of-two-integers/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def getSum(self, a: int, b: int) -> int:
        """异或是无进位的加法，结合与运算来进位"""
        x = 0xffffffff
        a, b = a & x, b & x  # 舍去a的32位以上的位变为0
        # 循环，当进位为 0 时跳出
        while b != 0:
            # a, b = 非进位和, 进位
            a, b = (a ^ b), (a & b) << 1 & x
        return a if a <= 0x7fffffff else ~(a ^ x)  # ~（a ^ x)将32位以上的位取反，1至32位不变


```

[136. 只出现一次的数字](https://leetcode.cn/problems/single-number/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for n in nums:
            res ^= n
        return res

```

[137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ones, twos = 0, 0
        for num in nums:
            ones = ones ^ num & ~twos
            twos = twos ^ num & ~ones
        return ones


```

# 数学

[238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """计算前缀积，再计算后缀积，最后相乘。
        
        时间：O(N)
        空间：O(N)，可以通过只使用res来存储前后缀积来优化到O(1)
        """
        n = len(nums)

        prefix_product_l, prefix_product_r = [1] * n, [1] * n

        for i in range(1, n):
            prefix_product_l[i] = prefix_product_l[i-1] * nums[i-1]
        
        for i in range(n-2, -1, -1):
            prefix_product_r[i] = prefix_product_r[i+1] * nums[i+1]
        
        res = []

        for i in range(n):
            res.append(prefix_product_l[i] * prefix_product_r[i])
        
        return res

```

```python

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        空间：O(1)
        """
        n = len(nums)
        res = [1] * n
        tmp = 1

        for i in range(1, n):
            res[i] = res[i-1] * nums[i-1]
        
        for i in range(n - 2, -1, -1):
            tmp *= nums[i+1]
            res[i] *= tmp
        
        return res
        
```


[169. 多数元素](https://leetcode.cn/problems/majority-element/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        """
        遍历数组，如果votes==0，假设当前数字是众数

        时间：O(N)
        空间：O(1)
        """
        votes = 0
        for num in nums:
            if votes == 0:  # 每当votes==0，假设当前数字是众数，重新来过
                x = num
            if num == x:  # 摩尔投票
                votes += 1
            else:
                votes -= 1
        
        return x

```

如果不能确保有众数，再遍历一次做判断即可

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        """
        遍历数组，如果votes==0，假设当前数字是众数

        时间：O(N)
        空间：O(1)
        """
        votes = 0
        for num in nums:
            if votes == 0:  # 每当votes==0，假设当前数字是众数，重新来过
                x = num
            if num == x:  # 摩尔投票
                votes += 1
            else:
                votes -= 1
        
        count = 0
        for num in nums:
            if num == x:
                count += 1
        return x if count > len(nums) // 2 else 0
```


[343. 整数拆分](https://leetcode.cn/problems/integer-break/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        """
        尽可能拆成3，求导的极大值为e，3是离自然对数e最近的那个数。
        """
        if n <= 3:
            return n - 1
        
        a, b = n // 3, n % 3
        if b == 0:  # 被3整除
            return 3 ** a
        if b == 1:  # 余1，将一个1 + 3转化为2 + 2
            return 3 ** (a - 1) * 2 * 2
        return 3 ** a * 2
```


[89. 格雷编码](https://leetcode.cn/problems/gray-code/description/?envType=study-plan-v2&envId=selected-coding-interview)


```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        """
        R(n) = G(n)倒序，并给每个元素二进制前面添加1
        G(n+1) = G(n) 拼接 R(n)
        """
        res, head = [0], 1
        for i in range(n):
            for j in range(len(res) - 1, -1, -1):
                res.append(head + res[j])
            head <<= 1
        return res

```


[1823. 找出游戏的获胜者](https://leetcode.cn/problems/find-the-winner-of-the-circular-game/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        dp = [0] * (n + 1)  # dp[i]是(i, k)的解
        # base case (1, k)为 dp[1] = 0
        for i in range(2, n + 1):  # 从2开始，计算dp[i]
            dp[i] = (dp[i-1] + k) % i  # 状态转移方程
        return dp[n] + 1 # 最终解为dp[n]，返回下一个数
```


[400. 第 N 位数字](https://leetcode.cn/problems/nth-digit/description/?envType=study-plan-v2&envId=selected-coding-interview)


```python

class Solution:
    def findNthDigit(self, n: int) -> int:
        """
        时间：O(logN)
        空间：O(logN)
        """
        digit = 1  # 几位数
        start = 1  # 整十整百开始的那个数字
        count = 9  # 整十整百从开始到结束的数字个数
        
        # 找到目标是在几位数
        while n > count: 
            n -= count
            start *= 10
            digit += 1
            count = 9 * start * digit 

        # 找到目标所在的数字
        num = start + (n - 1) // digit  # 在从start开始的第（n - 1) // digit个数字里

        # 找到目标在该数字里的哪个位置
        s = str(num)
        res = int(s[(n - 1) % digit])  # 在num的第(n - 1) % digit个位置

        return res

```

[65. 有效数字](https://leetcode.cn/problems/valid-number/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        """
        1. sign: only first, or right after "eE"
        2. expo: before and after seenDigit, only appear once
        3. dot: no expo before, only appear once
        """
        
        seen_digit = seen_expo = seen_dot = False
        
        for i, ch in enumerate(s):
            if ch.isdigit():
                seen_digit = True
            elif ch in "+-":
                if i > 0 and s[i - 1] != "e" and s[i - 1] != "E":
                    return False
            elif ch in "eE":
                if seen_expo or not seen_digit:
                    return False
                seen_expo = True
                seen_digit = False
            elif ch == ".":
                if seen_dot or seen_expo:
                    return False
                seen_dot = True
            else:
                return False
        
        return seen_digit
```

[233. 数字 1 的个数](https://leetcode.cn/problems/number-of-digit-one/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def countDigitOne(self, n: int) -> int:
        digit, res = 1, 0
        high = n // 10
        cur = n % 10  # 位数，从后往前
        low = 0
        # 给每一个位置放入1，计算其可以形成的次数。累加即可得到总次数
        while high != 0 or cur != 0:
            if cur == 0:  # 如果该位是0，那么乘法原理，那么1在这一位出现的次数=前面的数*该位（前）
                res += high * digit
            elif cur == 1:  # 如果该位是1，那么1在这一位出现的次数=前面的数*该位（前）+后+1（自己）
                res += high * digit + low + 1
            else:  # 所有其他情况，只由高位决定，1在这一位出现的次数=（前面的数（前）+ 1（自己）） * 该位
                res += (high + 1) * digit
            low += cur * digit
            cur = high % 10
            high //= 10
            digit *= 10
        
        return res

```



