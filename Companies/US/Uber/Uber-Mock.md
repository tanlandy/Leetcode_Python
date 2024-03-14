## 06/09/22

SDE Fulltime

### BQ

Steps:

1. self-introduce -> JAVA
   1. 教育背景
   2. 项目经历培养了自己-> team orientate, result orientened
   3. Apply knowledge to your company
2. resume
   1. 了解不了解技术
   2. tech efficient
3. coding
   1. 是否重复code -> readability
   2. 是否模块化 -> maintainability
   3. 是否光i, j
   4. 缩进
4. Q&A
   1. what kind of ideal candidate you are looking for
      1. smart
      2. communicate
      3. willing to grow, taking chanllenge
   2. team size, and the people
   3. 要问面试官的intro的业务相关，工作routine，有没有plan做什么东西，是否有深度和面试官深度交流
      1. 看看面试官的linkedin，做做prepare

talk more about the job experience. which part you are building? how did you do the data pre-processing, what models you tested? how did you selected features? how many features? group into 500 for each patient? How did you achieve the improvement? for different group of customers?
-> 不清楚你做了什么，而不是你的team在做什么
-> 清楚为什么做这个东西，做了什么，结果是什么 -> 知道做了什么，为什么做

how process is organized? how decision making, distributing work? who is in charge of the process? did you talk this issue with the manager?

### Coding

50 人 过 6、7个

implement a product class has following fuctions:

1. Product(int k)
2. add(int n)
3. get() -> return the product of last k numbers added
   1. 提问题->自己带着思路

需要clarification

1. 让举个例子弄明白->没有例子不答题：我举个例子你看对不对
2. 解释solution再写code，最好用例子来举例，说下用什么数据结构，为什么用这个
3. 写完code大概说一下
4. 分析时间空间复杂度，寻求最优解
5. 写完code要自己检查，一个是clean，另一个是

```py
import collections
class Product:
    def __init__(self, k):
        self.k = k
        self.queue = collections.deque()
    
    def add(self, n):
        if len(self.queue) >= self.k:
            self.queue.popleft()
        self.queue.append(n)
    
    def get(self):
        product = 1
        for n in self.queue:
            product *= n
        return product

if __name__ == "__main__":
    obj = Product(1)
    obj.add(3)
    obj.add(3)
    obj.add(3)
    obj.add(3)
    obj.add(4)
    print(obj.get())
```

```py
import collections
class Product:
    def __init__(self, k):
        self.queue = collections.deque([1])
        self.k = k

    def add(self, n):    
        if n == 0:
            for i in range(len(self.queue)):
                self.queue[i] = 1
            return
        else:
            if len(self.queue) > self.k:
                self.queue.popleft()
            self.queue.append(n * self.queue[-1])

    def get(self):
        if self.k > len(self.queue):
            return self.queue[-1]
        return self.queue[-1] // self.queue[-1 - self.k]

if __name__ == "__main__":
    obj = Product(4)
    obj.add(3)
    obj.add(0)
    obj.add(5)
    obj.add(2)
    obj.add(4)
    print(obj.get())
```
