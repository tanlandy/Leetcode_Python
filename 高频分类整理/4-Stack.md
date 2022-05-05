[225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)

```py
class MyStack:
    """
    用两个queue，每次push就正常push到q1，pop的话就把q1其余的东西存到q2，然后再交换q1,q2即可
    """

    def __init__(self):
        self.q1 = collections.deque()
        self.q2 = collections.deque()
        

    def push(self, x: int) -> None: # O(1)
        self.q1.append(x)

    def pop(self) -> int: # O(N)
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        x = self.q1.popleft()
        self.q1, self.q2 = self.q2, self.q1
        return x

    def top(self) -> int:
        return self.q1[-1]

    def empty(self) -> bool:
        return len(self.q1) < 1


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```


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

