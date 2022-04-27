[20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)


```py
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        closeToOpen = { ")":"(", "]":"[", "}":"{" }

        for c in s:
            if c in "({[":
                stack.append(c)
            elif stack and stack[-1] == closeToOpen[c]:
                stack.pop()
            else:
                return False
        
        return stack == []
```
        
[155. Min Stack](https://leetcode.com/problems/min-stack/)
用两个stack就可以，重点是min_stack的插入数据方式：先找到最小值，然后放入

```py
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

[716. Max Stack](https://leetcode.com/problems/max-stack/)
和min stack的区别是，用一个buffer来存删除过的数字，最后再加回来

```py
class MaxStack:

    def __init__(self):
        self.stack = []
        self.max_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        val = max(x, self.max_stack[-1] if self.max_stack else x)
        self.max_stack.append(val)

    def pop(self) -> int:
        self.max_stack.pop()
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def peekMax(self) -> int:
        return self.max_stack[-1]

    def popMax(self) -> int:
        val = self.max_stack[-1]
        buffer = []

        while self.stack[-1] != val:
            self.max_stack.pop()
            buffer.append(self.stack.pop())
        self.stack.pop()
        self.max_stack.pop()

        while buffer:
            self.push(buffer.pop())
        return val


# Your MaxStack object will be instantiated and called as such:
# obj = MaxStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.peekMax()
# param_5 = obj.popMax()
```

[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)
用stack存所有的数字即可

```py
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        
        for c in tokens:
            if c == "+":
                stack.append(stack.pop() + stack.pop())
            elif c == "-":
                a, b = stack.pop(), stack.pop()
                stack.append(b - a)
            elif c == "*":
                stack.append(stack.pop() * stack.pop())
            elif c == "/":
                a, b = stack.pop(), stack.pop()
                stack.append(int(b / a))
            else:
                stack.append(int(c))
        
        return stack[0]
```

[22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
添加close的条件：close<open

```py
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        """
        1. add ( if open < n
        2. add ) if close < open
        3. valid if open == close == n 
        """

        stack = []
        res = []
        
        def backtrack(openN, closedN):
            if openN == closedN == n: # base case
                res.append("".join(stack))
                return
            
            if openN < n:
                stack.append("(")
                backtrack(openN + 1, closedN)
                stack.pop()
            
            if closedN < openN:
                stack.append(")")
                backtrack(openN, closedN + 1)
                stack.pop()
        
        backtrack(0, 0)
        return res

```

[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

单调递减栈（非增）monotonic decreasing stack；如果下一个数更大，就一直弹栈，直到找到能把这个数放进去；弹栈的时候就可以idx的差值就是被删除栈的output；如果下一个数更大，就压栈

时间：O(N)
空间：O(N)

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = [] # pair:[temp, index]

        for idx, t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackTemp, stackIdx = stack.pop()
                res[stackIdx] = (i - stackIdx)
            stack.append([t, idx])
        
        return res    
```

[853. Car Fleet](https://leetcode.com/problems/car-fleet/)
计算出每辆车到达所需要的时间，按照position从后往前来比较，如果下一辆车所需时间更短，就说明会相撞并形成一个车队，然后一直和那个position最后的比较。因为总是和上一个比较，所以使用stack。最后的解说就是stack的大小

```py
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        pair = [[p, s] for p, s in zip(position, speed)]

        stack = []
        for p, s in sorted(pair)[::-1]: # reverse sorted order
            stack.append((target - p) / s)
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()
        
        return len(stack)
```


[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
stack存idx和height

时间：O(N)
空间：O(N)
```py
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        max_area = 0
        stack = [] # pair: (idx, height)

        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                max_area = max(max_area, height * (i - index))
                start = index
            stack.append((start, h))
        
        for i, h in stack:
            max_area = max(max_area, h * (len(heights) - i))

        return max_area

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

[141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

while里面用到了fast.next.next所以条件是fast and fast.next

时间：O(N)
空间：O(1)
```py
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        
        return False
```


[287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
Floyd's algo：先找到相遇的节点，再找cycle起点

时间：O(N)
空间：O(1)

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

[25. Reverse Nodes in k-Group](https://www.youtube.com/watch?v=1UOPsfP85V4)


```py


```