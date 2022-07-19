# 基础知识

# 例题

[362. Design Hit Counter](https://leetcode.com/problems/design-hit-counter/)

```py
class HitCounter:

    def __init__(self):
        """
        因为getHits只增，所以可以用queue来存所有数，每次getHits之后删掉距离大于300的数
        """
        self.counter = collections.deque()

    def hit(self, timestamp: int) -> None:
        self.counter.append(timestamp)        

    def getHits(self, timestamp: int) -> int:
        while self.counter and timestamp - self.counter[0] >= 300:
            self.counter.popleft()
        return len(self.counter)


# Your HitCounter object will be instantiated and called as such:
# obj = HitCounter()
# obj.hit(timestamp)
# param_2 = obj.getHits(timestamp)
```

时间复杂度空间复杂度更好的方法
```py
class HitCounter:

    def __init__(self):
        """
        更好的方法是只存最多300个数，分别用time[]和hit[]来记录数字，每次时间超过300，对应位置的hit就重置为1
        这个方法也是multithread safe的，因为只用了tuple和referencing variable
        """
        self.pair = [(0, 0)] * 300

    def hit(self, timestamp: int) -> None:
        idx = timestamp % 300
        time, hit = self.pair[idx]
        if time == timestamp:
            self.pair[idx] = (timestamp, hit + 1)
        else:
            self.pair[idx] = (timestamp, 1)

    def getHits(self, timestamp: int) -> int:
        res = 0
        for i in range(len(self.pair)):
            time, hit = self.pair[i]
            if timestamp - time < 300:
                res += hit
        
        return res


# Your HitCounter object will be instantiated and called as such:
# obj = HitCounter()
# obj.hit(timestamp)
# param_2 = obj.getHits(timestamp)
```

[1166. Design File System](https://leetcode.com/problems/design-file-system/)

```py
class FileSystem:

    def __init__(self):
        self.paths = {}

    def createPath(self, path: str, value: int) -> bool:
        if path == "" or path == "/" or path in self.paths:
            return False
        
        # 找到parent的位置 rfind返回最后一个找到的位置
        part = path.rfind("/")
        parent = path[:part]
        if len(parent) > 1 and parent not in self.paths:
            return False
        
        self.paths[path] = value
        return True

    def get(self, path: str) -> int:
        return self.paths.get(path, -1)


# Your FileSystem object will be instantiated and called as such:
# obj = FileSystem()
# param_1 = obj.createPath(path,value)
# param_2 = obj.get(path)
```

## Iterator

[284. Peeking Iterator](https://leetcode.com/problems/peeking-iterator/)

```py
class PeekingIterator:
    def __init__(self, iterator):
        self._next = iterator.next()
        self._iterator = iterator

    def peek(self):
        return self._next

    def next(self):
        if self._next is None:
            raise StopIteration()
        to_return = self._next
        self._next = None
        if self._iterator.hasNext():
            self._next = self._iterator.next()
        return to_return

    def hasNext(self):
        return self._next is not None
```

[341. Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/)


[173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)

[281. Zigzag Iterator](https://leetcode.com/problems/zigzag-iterator/)

[251. Flatten 2D Vector](https://leetcode.com/problems/flatten-2d-vector/)

[900. RLE Iterator](https://leetcode.com/problems/rle-iterator/)

[1286. Iterator for Combination](https://leetcode.com/problems/iterator-for-combination/)



