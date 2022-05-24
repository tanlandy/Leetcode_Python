# Hashtable

[1. Two Sum](https://leetcode.com/problems/two-sum/)

```py
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """map: {num: idx}
        each time check if the complement is in the map, if it is, return
        """
        hashmap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap:
                return [i, hashmap[complement]]
            hashmap[nums[i]] = i
```


[146. LRU Cache](https://leetcode.com/problems/lru-cache/)

```py
"""
OrderedDict: keeps track of the order of the keys as they are added 
OrderDicts have two new methods in Python 3: popitem() and move_to_end() 
popitem() will return and remove a (key, item) pair. 
move_to_end() will move an existing key to either end of the OrderedDict. The item will be moved right end if the last argument for OrderedDict is set to True (which is the default), or to the beginning if it is False.

"""
from collections import OrderedDict
class LRUCache:
    def __init__(self, Capacity):
        self.size = Capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache: 
            return -1
        val = self.cache[key]
        self.cache.move_to_end(key) # move to end
        return val

    def put(self, key, val):
        if key in self.cache: 
            del self.cache[key]
        self.cache[key] = val
        if len(self.cache) > self.size:
            self.cache.popitem(last = False) # FIFO order like a queue
```


```py
"""
需要记录的：capacity;Node: key, val, pre, next; LRU class: cap, left, right, cache还要把left, right连起来 ;如果要get在O(1)：HashMap：{val: pointer to the node}；用left, right pointer来记录LRU和Most freqently used：double linkedlist;当第三个node来了：更新hashMap， 更新left, right pointer，更新第二使用的node和这个node的双向链接；每次get: 删除，添加操作；每次put：如果存在要删除，总要添加操作，如果大小不够，就找到lru(最左），然后删除
"""
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = self.next = None


class LRUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {} # map key to node

        self.left, self.right = Node(0, 0), Node(0, 0)
        self.left.next, self.right.prev = self.right, self.left # left = LRU, right = most recent

    # remove from the linkedlist
    def remove(self, node):
        prev, next = node.prev, node.next
        prev.next, next.prev = next, prev

    # insert node at right
    def insert(self, node):
        # insert at the right most position, before the right pointer
        prev, next = self.right.prev, self.right
        prev.next = next.prev = node
        node.next, node.prev = next, prev

    def get(self, key: int) -> int:
        if key in self.cache:
            # remove + insert为了更新顺序
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key]
        node = Node(key, value)
        self.cache[key] = node
        self.insert(node)

        if len(self.cache) > self.cap:
            # remove from the list and delete the LRU from the hashmap
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]



# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```




# Heap



[535. Encode and Decode TinyURL](https://leetcode.com/problems/encode-and-decode-tinyurl/)
两个字典，分别存{long:short}和{short:long}

```python

class Codec:
    codeDB, urlDB = defaultdict(), defaultdict()
    chars = string.ascii_letters + string.digits

    def getCode(self) -> str:
        code = ''.join(random.choice(self.chars) for i in range(6))
        return "http://tinyurl.com/" + code
 
    def encode(self, longUrl: str) -> str:
        if longUrl in self.urlDB: 
            return self.urlDB[longUrl]

        code = self.getCode()
        while code in self.codeDB:
             code = getCode()
             
        self.codeDB[code] = longUrl
        self.urlDB[longUrl] = code
        return code

    def decode(self, shortUrl: str) -> str:
        return self.codeDB[shortUrl]

```


[49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

用一个count记录每个string的字母出现次数，然后res的values存最后的结果

时间：O(M*N), M is len(strs), N is average len(one string)
空间：O(M*N)

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # {charCount: oneRes}
        res = collections.defaultdict(list) 
        
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord("a")] += 1
            res[tuple(count)].append(s)
        
        return res.values()
```

时间复杂度更高：把每个string排序，按照这个排序来加到map中
```py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = collections.defaultdict(list)
        for entry in strs:
            anagram_id = "".join(sorted(entry))
            res[anagram_id].append(entry)
        sorted(res.items(), key = lambda item:item[1])
        return res.values()


```


[238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

走左右两遍

时间：O(N)
空间：O(1)
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * (len(nums))
        
        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
            
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
            
        return res
```

[36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)
用index标记每个九宫格，index=(r//3, c//3)；或者index=3 * (r//3) + c//3

时间：O(9^2)
空间：O(9^2)
```py
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = collections.defaultdict(set)
        rows = collections.defaultdict(set)
        squares = collections.defaultdict(set) # key = (r//3, c//3)

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (board[r][c] in rows[r] or 
                    board[r][c] in cols[c] or
                    board[r][c] in squares[(r // 3, c // 3)]):
                    return False
                rows[r].add(board[r][c])
                cols[c].add(board[r][c])
                squares[(r // 3, c // 3)].add(board[r][c])
        
        return True

```

[128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)
存成set，对每个数，看是否是start of array:查看是否有左边neighbor。如果没有左边的数就开始看右边neighbor来计算length

时间：O(N)
空间：O(N)

```py
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        numSet = set(nums)
        num_set = 0

        for n in nums:
            if (n - 1) not in num_set:
                length = 1
                while (n + length) in num_set:
                    length += 1
                longest = max(length, longest)
        return longest

```

[205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/)
用map记录映射关系，用set保证只有映射是唯一的

时间：O(N)
空间：O(1) 最多就是26个字母
```py
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        s_map = {}
        used = set()
        
        for c1, c2 in zip(s, t):
            if c1 in s_map:
                if s_map[c1] != c2:
                    return False
            else:
                if c2 in used:
                    return False
                s_map[c1] = c2
                used.add(c2)
        
        return True
```