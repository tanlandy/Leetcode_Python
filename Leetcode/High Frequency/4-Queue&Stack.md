# Queue&Stack

# 基础知识，技巧与思路

# 高频题

## Krahets精选题

20, 155, 232, 394, 295

# Comparation

| Aspects | Arrays | Linked Lists |
| --- | --- | --- |
| Insertion/Deletion at the beginning | O(N) | O(1) |
| Access Element | O(1) | O(N) |
| Contiguous Memory | Yes | No |

# Queue

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

[346. Moving Average from Data Stream](https://leetcode.com/problems/moving-average-from-data-stream/)

```python
from collections import deque

class MovingAverage:
    """
    用queue，每次记录当前的windowSum，如果size满足的话，来一个新数字就弹出末尾的，同时加进来新的，最后返回windowSum/len即可
    queue删除头部: queue.popleft()；queue加数字: queue.append(val)

    时间：O(1)
    空间：O(N)N is the size of window
    """

    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        self.windowSum = 0

    def next(self, val: int) -> float:
        if len(self.queue) == self.size:
            self.windowSum -= self.queue.popleft()
        self.queue.append(val)
        self.windowSum += val
        return self.windowSum / len(self.queue)

obj = MovingAverage(2)
param_1 = obj.next(3)
param_2 = obj.next(5)
param_3 = obj.next(9)
param_4 = obj.next(1)
param_5 = obj.next(2)
param_6 = obj.next(3)

print(param_3)

# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)
```

[281. Zigzag Iterator](https://leetcode.com/problems/zigzag-iterator/)

```py
class ZigzagIterator:
    def __init__(self, v1, v2):
        """
        keep a queue of pointers to the input vectors
        v1 = [1,2], v2 = [3,5,6]
        queue: [[1,2], [3,5,6]]
        """
        self.queue = collections.deque([_ for _ in (v1,v2) if _])

    def next(self):
        v = self.queue.popleft() # pop out a pointer from the queue
        res = v.pop(0) # the result is the element from the chosen vector
        if v: # if the vector is not empty, add it to the end, make it zigzag 
            self.queue.append(v)
        return res

    def hasNext(self):
        if self.queue: 
            return True
        return False
        

# Your ZigzagIterator object will be instantiated and called as such:
# i, v = ZigzagIterator(v1, v2), []
# while i.hasNext(): v.append(i.next())
```

[1429. First Unique Number](https://leetcode.com/problems/first-unique-number/)

```py
class FirstUnique:

    def __init__(self, nums: List[int]):
        self.queue = collections.deque() # first num if the unique number
        self.is_unique = {} # {one_num: unique_or_not}
        for n in nums:
            self.add(n)

    def showFirstUnique(self) -> int:
        # queue的第一个总是first unique，如果不是，就一直删除
        while self.queue and not self.is_unique[self.queue[0]]:
            self.queue.popleft()
        if self.queue:
            return self.queue[0]
        return -1
        
    def add(self, value: int) -> None:
        # new number
        if value not in self.is_unique:
            self.is_unique[value] = True
            self.queue.append(value)
        # seen before
        else:
            self.is_unique[value] = False
            

# Your FirstUnique object will be instantiated and called as such:
# obj = FirstUnique(nums)
# param_1 = obj.showFirstUnique()
# obj.add(value)
```

## Monotonic Deque

[239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

```py
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = collections.deque()  # 单调队列，存储最大值对应的index
        l = r = 0
        res = []

        while r < len(nums):
            while queue and nums[queue[-1]] < nums[r]:  # queue只会存储比nums[r]大的值
                queue.pop()

            queue.append(r)  # 把当前windows最大的值nums[r]对应的index存到queue中
            
            if l > queue[0]:  # queue存index的原因：便于比较当前是否范围大于了sliding window
                queue.popleft()
            
            if r - l + 1 == k:  # 更新res的时机，此时sliding window的范围==k
                res.append(nums[queue[0]])  # 最大值总是queue[0]
                l += 1
            
            r += 1
        
        return res

```

## Spiral Matrix

[48. Rotate Image](https://leetcode.com/problems/rotate-image/)

```py
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        for each row, store the top left first
        技巧是先把四个角交换的写出来，然后再把i加进去

        Time: O(N^2)
        Space: O(1)
        """
        l, r = 0, len(matrix) - 1
        while l < r:
            for i in range(r - l): # each time, solve the outer boundary, 一轮只rotate了col-row次
                top, bottom = l, r # two more pointers

                top_left = matrix[top][l + i] # save top left从后往前得处理，避免要存好几个tmp

                # 先不管+-i，写出来四个角。然后通过变化确定+-i
                matrix[top][l + i] = matrix[bottom - i][l] # 先让左上角等于左下角，然后左下角等于右下角， etc
                matrix[bottom - i][l] = matrix[bottom][r - i]
                matrix[bottom][r - i] = matrix[top + i][r]
                matrix[top + i][r] = top_left
            
            r -= 1
            l += 1
```

[54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

```py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        maintain the boundary using four pointers; each time finish one way, shift that boundary

        Time: O(M*N)
        Space: O(1), if doesn't consider output as extra memory
        """
        res = []
        l, r = 0, len(matrix[0])  # r out of the bound, l and r col
        top, bottom = 0, len(matrix)  # bottom out of the bound

        while l < r and top < bottom:
            # top row
            for i in range(l, r):
                res.append(matrix[top][i])  # we are in the top row
            top += 1  # shift top row down by 1

            # right most col
            for i in range(top, bottom):
                res.append(matrix[i][r - 1])
            r -= 1

            # has to check in the middle, as just updated top and r
            if not (l < r and top < bottom): 
                break

            # bottom row
            for i in range(r - 1, l - 1, -1):  # l is not inclusive, so one step forward by l-1
                res.append(matrix[bottom - 1][i])
            bottom -= 1  # shift upwards

            # left most col
            for i in range(bottom - 1, top - 1, -1):
                res.append(matrix[i][l])
            l += 1
        
        return res
```

[59. Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)

```py
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        """
        Same as LC54
        A = [0] * 2 => [0,0]
        A = [[0] * 2 for _ in range(2)] => [[0,0], [0,0]]
        """
        A = [[0] * n for _ in range(n)] # initiate matirx A filling with 0
        
        l, r = 0, n
        top, bottom = 0, n
        
        num = 1 # start with 1
        
        while l < r and top < bottom:
            for i in range(l, r):
                A[top][i] = num
                num += 1
            top += 1
            
            for i in range(top, bottom):
                A[i][r - 1] = num
                num += 1
            r -= 1
            
            if not (l < r and top < bottom):
                return A
            
            for i in range(r - 1, l - 1, -1):
                A[bottom - 1][i] = num
                num += 1
            bottom -= 1
            
            for i in range(bottom - 1, top - 1, -1):
                A[i][l] = num
                num += 1
            l += 1
        
        return A
        
```

# Stack

## Educative

Convert Decimal Integer to Binary

```py
def convert_int_to_bin(dec_num):
    
    if dec_num == 0:
        return 0
    
    s = Stack()

    while dec_num > 0:
        remainder = dec_num % 2
        s.push(remainder)
        dec_num = dec_num // 2

    bin_num = ""
    while not s.is_empty():
        bin_num += str(s.pop())

    return bin_num

print(convert_int_to_bin(56))
print(convert_int_to_bin(2))
print(convert_int_to_bin(32))
print(convert_int_to_bin(10))

print(int(convert_int_to_bin(56),2)==56)
```

## Zhihu

[155. Min Stack](https://leetcode.com/problems/min-stack/)

```py
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

另一种方式，min_stack只放最小值，删除时候如果相等才删min_stack

```py
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]

    def getMin(self) -> int:
        if self.min_stack:
            return self.min_stack[-1]
```

[716. Max Stack](https://leetcode.com/problems/max-stack/)

```py
class MaxStack:
    """
    和min stack的区别是，用一个buffer来存删除过的数字，最后再加回来
    删除时候一起删，buffer是存stack的值，最后再一起push回两个stack中
    """

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

[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)

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

[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

```py
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        """用stack存所有的数字即可"""
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
                stack.append(int(b / a)) # a, b = 3, 11, -11: b/a = 3.666,  int(b/a)=3, int(c/a) = -3
            else:
                stack.append(int(c)) # convert data type to int，每次都可以加，因为就是个数字
        
        return stack[0]
```

[227. Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/)

用Stack存数字，每次如果是+-就直接压进去，如果是*/就压进去相对应的数，最后弹栈相加；-3//2地板除会得到-2而不是想要的-1，所以用int(-3/2)这样就可以得到-1;检查是否是数字: s[i].isdigit()；把长串string转成对应的数字num=num*10+int(s[i]);如果是"+-*/": if s[i] in "+-*/"；sign的条件：如果是sign或者走到最后一位；相加stack的所有数字：sum(stack)；每次检查完sign之后要更新num和sign；最后还有把最后的数放进stack里
时间：O(N)
空间：O(N)

```python
class Solution:
    def calculate(self, s):
        def update(sign, num):
            if sign == "+":
                stack.append(num)
            elif sign == "-":
                stack.append(-num)
            elif sign == "*":
                stack.append(stack.pop()*num)
            else:
                stack.append(int(stack.pop()/num))

        idx, num, stack, sign = 0, 0, [], "+"
        while idx < len(s):
            if s[idx].isdigit(): 
                num = num * 10 + int(s[idx]) # 不是每次都可以加，可能是多位数字
            elif s[idx] in "+-*/":
                update(sign, num)
                num = 0
                sign = s[idx]
            idx += 1
        update(sign, num)
        return sum(stack)
```

[224. Basic Calculator](https://leetcode.com/problems/basic-calculator/)

当看到"("就从下一位call自己，看到")"就返回"()"之间的值

```python
class Solution:
    def calculate(self, s):
        def update(sign, num):
            if sign == "+":
                stack.append(num)
            elif sign == "-":
                stack.append(-num)
        
        idx, num, stack, sign = 0, 0, [], "+"
        while idx < len(s):
            if s[idx].isdigit():
                num = num * 10 + int(s[idx])
            elif s[idx] in "+-":
                update(sign, num)
                num = 0
                sign = s[idx]
            elif s[idx] == "(":
                num, j = self.calculate(s[idx + 1:]) # 需要返回（）内部的值和当前的idx位置
                idx = idx + j
            elif s[idx] == ")":
                update(sign, num)
                return sum(stack), idx + 1 # 看到了“）”，返回（）里面的值和idx的位置
            idx += 1                       # 别忘了idx += 1
        update(sign, num)                  # 别忘了最后一个数的处理
        return sum(stack)
```

[772. Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/)

```python
class Solution:
    def calculate(self, s: str) -> int:
        def update(sign, num):
            if sign == "+":
                stack.append(num)
            elif sign == "-":
                stack.append(-num)
            elif sign == "*": # BC II,III
                stack.append(stack.pop() * num)
            elif sign == "/": # BC II,III
                stack.append(int(stack.pop() / num))
        
        idx, num, stack, sign = 0, 0, [], "+"
        
        while idx < len(s):
            if s[idx].isdigit():
                num = num * 10 + int(s[idx])
            elif s[idx] in "+-*/":
                update(sign, num)
                sign = s[idx]
                num = 0
            elif s[idx] == "(": # BC I,III
                num, j = self.calculate(s[idx+1:])
                idx += j
            elif s[idx] == ")": # BC I,III
                update(sign, num)
                return sum(stack), idx + 1
            idx += 1
        update(sign, num)
        return sum(stack)
```

[20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

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

[1472. Design Browser History](https://leetcode.com/problems/design-browser-history/)

```py
class BrowserHistory:
    """
    two stacks:
    when visit: append to history, and clear future
    when back: append the history to future, while making sure always one in history
    when forward: append the future to history
    """

    def __init__(self, homepage: str):
        self.history = []
        self.future = []
        self.history.append(homepage)

    def visit(self, url: str) -> None:
        self.history.append(url)
        self.future = []

    def back(self, steps: int) -> str:
        while steps > 0 and len(self.history) > 1: # when go back, must have one website in the history
            self.future.append(self.history.pop())
            steps -= 1
        return self.history[-1]

    def forward(self, steps: int) -> str:
        while steps > 0 and self.future:
            self.history.append(self.future.pop())
            steps -= 1
        return self.history[-1]
    
```

[1209. Remove All Adjacent Duplicates in String II](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/)

```py
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        """
        用stack同时存这个char和出现的次数，一旦出先次数达到k就pop，最后decode到需要的大小
        """
        stack = [["#", 0]] # 初始值用[[“#”， 0]], is a list of list
        
        for c in s:
            if stack[-1][0] == c:
                stack[-1][1] += 1
                if stack[-1][1] == k:
                    stack.pop()
            else:
                stack.append([c, 1])
        
        return "".join(c * n for c, n in stack)
```

[1249. Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        """
        先把s变成list，用stack存inValid的'(',')'的index; 
        如何判断inValid：每次看到(就压栈，看到)要么弹，空栈就要直接换成""
        最后把s导出成string：list变成str: "".join(s)

        时间：O(N)
        空间：O(N)
        """
        s = list(s)
        stack = []
        for index, ch in enumerate(s):
            if ch == "(":
                stack.append(index)
            elif ch == ")":
                if stack:
                    stack.pop()
                else:
                    s[index] = ""
        while stack:
            s[stack.pop()] = ""
            
        return "".join(s)
```

[735. Asteroid Collision](https://leetcode.com/problems/asteroid-collision/)

```py
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        
        for a in asteroids:
            a_exist = True
            while a_exist and stack and a < 0 < stack[-1]: # 条件是a是负数，同时stack[-1]是正数
                top = stack[-1]  # 因为可能pop掉，所以用top存储
                if top <= -a:  # 小于或等于，都要碰撞
                    stack.pop()
                if top >= -a:  # 大于或等于，a就要被碰撞消除掉
                    a_exist = False
            if a_exist:
                stack.append(a)
        
        return stack

```

## Other

[1544. Make The String Great](https://leetcode.com/problems/make-the-string-great/description/)

```py
class Solution:
    def makeGood(self, s: str) -> str:
        """
        大小写字母的ord码相差32，维护一个stack
        """
        res = []

        for ch in s:
            if res and abs(ord(ch) - ord(res[-1])) == 32:
                res.pop()
            else:
                res.append(ch)

        return "".join(res)
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

[853. Car Fleet](https://leetcode.com/problems/car-fleet/)

```py
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        """
        计算出每辆车到达所需要的时间，按照position从后往前来比较，如果下一辆车所需时间更短，就说明会相撞并形成一个车队，然后一直和那个position最后的比较。因为总是和上一个比较，所以使用stack。最后的结果就是stack的大小
        """

        pair = [[p, s] for p, s in zip(position, speed)]

        stack = []
        for p, s in sorted(pair)[::-1]: # reverse sorted order
            stack.append((target - p) / s) # 每次计算位置靠前的一辆车的到达所需要的时间
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()
        
        return len(stack)
```

[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) 再看看

```py
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        """
        stack存idx和height

        时间：O(N)
        空间：O(N)
        """
        max_area = 0
        stack = [] # pair: (idx, height)

        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h: # 遇到了下降的情况，一直走直到遇到上升；每次area都是当前h*(i-index) ：上次满足的
                index, height = stack.pop()
                max_area = max(max_area, height * (i - index))
                start = index
            stack.append((start, h))
        
        for i, h in stack:
            max_area = max(max_area, h * (len(heights) - i))

        return max_area

```

[316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)

```py
class Solution:
    def removeDuplicateLetters(self, s) -> int:

        stack = []
        seen = set()

        # this will let us know if there are no more instances of s[i] left in s
        last_occurrence = {c: i for i, c in enumerate(s)}

        for i, c in enumerate(s):
            # we can only try to add c if it's not already in our solution
            # this is to maintain only one of each character
            if c not in seen:
                # if the last letter in our solution:
                #    1. exists
                #    2. is greater than c so removing it will make the string smaller
                #    3. it's not the last occurrence
                # we remove it from the solution to keep the solution optimal
                while stack and c < stack[-1] and i < last_occurrence[stack[-1]]:
                    seen.discard(stack.pop())
                seen.add(c)
                stack.append(c)
        return ''.join(stack)
```

## Monotonic stack

[2104. Sum of Subarray Ranges](https://leetcode.com/problems/sum-of-subarray-ranges/description/)

```py
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        res = 0

        for i in range(len(nums)):
            l = r = nums[i]
            for j in range(i, len(nums)):
                l = min(l, nums[j])
                r = max(r, nums[j])
                res += r - l
        
        return res
```

[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        单调递减栈（非增）monotonic decreasing stack；如果下一个数更大，就一直弹栈，直到找到能把这个数放进去；弹栈的时候就可以idx的差值就是被删除栈的output；如果下一个数更大，就压栈

        时间：O(N)
        空间：O(N)
        """
        res = [0] * len(temperatures)
        stack = [] # [(temp, idx)]

        for idx, t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                temp, i = stack.pop()
                res[i] = idx - i
            
            stack.append((t, idx))
        
        return res    
```
