# 基础知识

# 例题

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




