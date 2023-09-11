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
