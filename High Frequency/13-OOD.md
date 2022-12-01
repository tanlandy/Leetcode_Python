# 基础知识

# 例题

[981. Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)

```py
class TimeMap:
    """
    添加进去之后，用二分查找
    """

    def __init__(self):
        self.dic = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.dic[key].append([timestamp, value])

    def get(self, key: str, timestamp: int) -> str:
        arr = self.dic[key]
        l, r = 0, len(arr) - 1
        while l <= r:
            mid = (l + r) // 2
            if arr[mid][0] <= timestamp:
                l = mid + 1
            else:
                r = mid - 1
        
        return "" if r == -1 else arr[r][1]
```

[2034. Stock Price Fluctuation](https://leetcode.com/problems/stock-price-fluctuation/)



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


[348. Design Tic-Tac-Toe](https://leetcode.com/problems/design-tic-tac-toe/description/)

```py
class TicTacToe:
    def __init__(self, n: int):
        self.n = n
        self.hori = [0] * n
        self.ver = [0] * n
        self.diag1 = 0
        self.diag2 = 0

    def move(self, row: int, col: int, player: int) -> int:
        n = self.n
        move = 1
        if player == 2:
            move = -1
        
        self.hori[col] += move
        self.ver[row] += move
        
        if col == row:
            self.diag1 += move
        if row + col == (n-1): # for points in the same diag, they have the same (r + c)
            self.diag2 += move

        if abs(self.hori[col]) == n or abs(self.ver[row]) == n or abs(self.diag1) == n or abs(self.diag2) == n:
            return player
        
        return 0
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

[729. My Calendar I](https://leetcode.com/problems/my-calendar-i/)


[173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)

[281. Zigzag Iterator](https://leetcode.com/problems/zigzag-iterator/)

[251. Flatten 2D Vector](https://leetcode.com/problems/flatten-2d-vector/)

```py
class Vector2D:

    def __init__(self, vec: List[List[int]]):
        """
        用两个指针，分别指向内部和外部的list
        每次先移动内部的j
        如果j走到了最后，就要移动i并将j重置为0
        """
        self.arr = vec
        self.i = 0 # idx of vec(outer)
        self.j = 0 # idx of element(inner)
        
    def next(self) -> int:
        res = self.arr[self.i][self.j]
        if self.j < len(self.arr[self.i]):
            self.j += 1
        while self.i < len(self.arr) and self.j == len(self.arr[self.i]): # 注意不要越界
            self.i += 1
            self.j = 0
        return res

    def hasNext(self) -> bool:
        while self.i < len(self.arr) and self.j == len(self.arr[self.i]): # 移到满足条件的点
            self.i += 1
            self.j = 0
        return self.i < len(self.arr)

# Your Vector2D object will be instantiated and called as such:
# obj = Vector2D(vec)
# param_1 = obj.next()
# param_2 = obj.hasNext()
```



[900. RLE Iterator](https://leetcode.com/problems/rle-iterator/)

[1286. Iterator for Combination](https://leetcode.com/problems/iterator-for-combination/)



