# 基础知识

# 例题

[1. Two Sum](https://leetcode.com/problems/two-sum/)

```py
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        因为最后要返回idx，所以用dict()来存一下值对应的坐标
        """
        seen = dict() # {val: idx}
        
        for idx, n in enumerate(nums):
            remain = target - n
            if remain in seen:
                return [idx, seen[remain]]
            seen[n] = idx
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
需要记录的：capacity;Node: key, val, pre, next; LRU class: cap, left, right, cache还要把left, right连起来 ;
如果要get在O(1)：HashMap：{val: pointer to the node}；用left, right pointer来记录LRU和Most freqently used：double linkedlist;当第三个node来了：更新hashMap， 更新left, right pointer，更新第二使用的node和这个node的双向链接；每次get: 删除，添加操作；每次put：如果存在要删除，总要添加操作，如果大小不够，就找到lru(最左），然后删除
"""
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val # value is stored in Node
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {} # map key to node

        self.left, self.right = Node(0, 0), Node(0, 0) # left, right是两个node，并且开始时候相互连起来
        self.left.next, self.right.prev = self.right, self.left # left = LRU, right = most recent

    # remove "node" from the linkedlist
    def remove(self, node):
        prev, next = node.prev, node.next
        prev.next, next.prev = next, prev

    # insert "node" at right
    def insert(self, node):
        # insert at the right most position, before the right pointer
        prev, next = self.right.prev, self.right
        prev.next = next.prev = node
        node.next, node.prev = next, prev

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key]) # 先删除掉，这样就可以保证待会儿插入在最右边
        node = Node(key, value)
        self.cache[key] = node
        self.insert(node)

        if len(self.cache) > self.cap:
            # remove from the list and delete the LRU from the hashmap
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
    
    def get(self, key: int) -> int:
        if key in self.cache:
            # remove + insert为了更新顺序
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

[460. LFU Cache](https://leetcode.com/problems/lfu-cache/)



[128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

```py
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        """
        Use a hashset to find element in O(1)
        for each num, 看是否是start of array: check if (num-1) in set or not: if not in, reset and count
        
        Time: O(N) tricky: for[1,2,3,4,5,6] only 1 is valid for the loop
        Space: O(N)
        """
        
        longest = 0
        nums_set = set(nums)
        
        for n in nums:
            if (n - 1) not in nums_set:
                cur = 1
                while (n + cur) in nums_set:
                    cur += 1
                longest = max(longest, cur)
        
        return longest

```

[73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)
```py
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        O(M+N) memory: use two sets, one for rows and one for cols to keep track of those who have zeroes 
        O(1) memory: use the first row and col to keep track of those who have zeroes. if see a zero in the middle of matrix, set the corresponding first row and col to 0, to avoid making first row and col be 0, we need to use two bool to store whether should the first row and col to be zero
        
        Time: O(M*N)
        Space: O(1)
        """
        
        rows, cols = len(matrix), len(matrix[0])
        first_row_zero = first_col_zero = False
        
        for r in range(rows):
            for c in range(cols):
                # mark place that has zero
                if matrix[r][c] == 0:
                    if r == 0:
                        first_row_zero = True
                    if c == 0:
                        first_col_zero = True
                    matrix[r][0] = 0
                    matrix[0][c] = 0
        
        # set other place
        for r in range(1, rows):
            for c in range(1, cols):
                if matrix[r][0] == 0 or matrix[0][c] == 0:
                    matrix[r][c] = 0
        
        # set first row
        if first_row_zero:
            for c in range(cols):
                matrix[0][c] = 0
        
        # set first col
        if first_col_zero:
            for r in range(rows):
                matrix[r][0] = 0
```

[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)

```py
class RandomizedSet:

    def __init__(self):
        self.list = []
        self.map = {}
        

    def insert(self, val: int) -> bool:
        if val in self.map:
            return False
        self.map[val] = len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.map:
            return False
        
        last_elem = self.list[-1]
        idx = self.map[val]
        # move the last element to the place of the one to be removed
        self.list[idx], self.map[last_elem] = self.list[-1], idx
        self.list.pop()
        del self.map[val]
        return True

    def getRandom(self) -> int:
        # use random.randint(), to generate random number
        rand_idx = random.randint(0, len(self.list) - 1)
        return self.list[rand_idx]

```

[49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        用一个count记录每个string中的字母出现次数，然后res的values存最后的结果
        res = {
            (2, 1, 0, 0, ..., 0): ["aab", "aba", "baa"]
            (1, 2, 3, 0, 0, ..., 0): ["abbccc"]
        }

        时间：O(M*N), M is len(strs), N is average len(one string)
        空间：O(M*N)
        """
        # {charCount: oneRes}
        res = collections.defaultdict(list) 
        
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord("a")] += 1
            res[tuple(count)].append(s)  # have to put as tuple
        
        return list(res.values()) 
```

[350. Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/)

```py
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        use a counter to track the count for number in nums1
        iterate along the second array, if the count of the number is positive, then add it to result and decrease its count in the counter
        
        Time: O(M+N)
        Space: O(min(M, N))
        """
        if len(nums1) > len(nums2):
            return self.intersect(nums2, nums1)
        
        counter = collections.Counter(nums1)
        res = []
        
        for n in nums2:
            if counter[n] > 0:
                res.append(n)
                counter[n] -= 1
        
        return res
```

If the two arrays are sorted:

```py
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        if the input arrays are sorted: use two pointers
        
        Time: O(min(M, N)), without counting the sorting time
        Space: O(1), without counting the sorting time
        """
        nums1.sort()
        nums2.sort()
        i, j = 0, 0 # used to compare nums1 and nums2
        k = 0 # used to update result
        
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                nums1[k] = nums1[i]:
                i += 1
                j += 1
                k += 1
        
        return nums1[:k]
```

[299. Bulls and Cows](https://leetcode.com/problems/bulls-and-cows/)

```py
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        """
        first pass to find the bulls and build two maps: {num: counter}
        iterate the secret map, if also in the guess map, update the cows with the smaller counter
        
        Time: O(N)
        Space: O(1), the map contains at most 10 elements
        """
        secret_dict = collections.defaultdict(int)
        guess_dict = collections.defaultdict(int)
        
        cows = bulls = 0
        
        for i in range(len(secret)):
            if secret[i] == guess[i]:
                bulls += 1
            else:
                secret_dict[secret[i]] += 1
                guess_dict[guess[i]] += 1
    
        for n in secret_dict:
            if n in guess_dict:
                cows += min(secret_dict[n], guess_dict[n])
        
        return "{0}A{1}B".format(bulls, cows)
```


```py
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        h = defaultdict(int)
        bulls = cows = 0

        for idx, s in enumerate(secret):
            g = guess[idx]
            if s == g: 
                bulls += 1
            else:
                # secret gives a postition contribution, guess a negative contribution
                cows += int(h[s] < 0) + int(h[g] > 0)
                h[s] += 1
                h[g] -= 1
                
        return "{}A{}B".format(bulls, cows)
```


## Hashset

[217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)

```py
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        
        for n in nums:
            if n in seen:
                return True
            seen.add(n)
            
        return False
```

[219. Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/)

```py
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        seen = set()
        
        for i in range(len(nums)):
            if nums[i] in seen:
                return True
            seen.add(nums[i])
            if len(seen) > k:
                seen.remove(nums[i - k])
        
        return False
```

[220. Contains Duplicate III](https://leetcode.com/problems/contains-duplicate-iii/)



[1207. Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences/)

```py
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        """
        this can exit earlier 
        不是把所有的放到set之后再一一比较，而是边往里放边比较
        """
        seen = set()
        
        for freq in collections.Counter(arr).values():
            if freq in seen:
                return False
            seen.add(freq)
        
        return True
```

[2133. Check if Every Row and Column Contains All Numbers](https://leetcode.com/problems/check-if-every-row-and-column-contains-all-numbers/)

```py
class Solution:
    def checkValid(self, matrix: List[List[int]]) -> bool:
        unique_nums = set()
        rows, cols = len(matrix), len(matrix[0])
        
        for r in range(rows):
            unique_nums.clear()
            for c in range(cols):
                unique_nums.add(matrix[r][c])
            if len(unique_nums) != rows:
                return False
        
        for c in range(rows):
            unique_nums.clear()
            for r in range(cols):
                unique_nums.add(matrix[r][c])
            if len(unique_nums) != cols:
                return False
        
        return True
```

## Tic-Tac-Toe

[348. Design Tic-Tac-Toe](https://leetcode.com/problems/design-tic-tac-toe/)

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

[1275. Find Winner on a Tic Tac Toe Game](https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/)

```py
class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        """
        check for each row/col/diag if all three are the same
        """
        n = 3
        ver = [0] * n
        hori = [0] * n
        diag1 = 0
        diag2 = 0
        
        player = 1
        
        for row, col in moves:
            ver[row] += player
            hori[col] += player
            if row == col:
                diag1 += player
            if (row + col) == n - 1:
                diag2 += player
            
            if abs(ver[row]) == n or abs(hori[col]) == n or abs(diag1) == n or abs(diag2) == n:
                return "A" if player == 1 else "B"
            
            # use this to take turns
            player *= -1
        
        return "Draw" if len(moves) == n * n else "Pending"
```

[794. Valid Tic-Tac-Toe State](https://leetcode.com/problems/valid-tic-tac-toe-state/)

```py
class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        """
        Since 'X' always play first, we need to guarantee below conditions:
        1. the number of 'X' must be larger than one or equal to the number of 'O'
        2. if winner is 'X', the number of 'X' must be equal to the number of 'O' +1
           if winner is 'O', the number of 'O' must be equal to the number of 'X'
        """
        count_X, count_O = 0, 0
        
        for s in board:
            for ch in s:
                if ch == "X":
                    count_X += 1
                elif ch == "O":
                    count_O += 1
        
        # Condition 1 提前判断
        if count_O not in {count_X, count_X - 1}:
            return False
        
        # Condition 2 看行列情况
        def isWinner(player):
            for i in range(3):
                if board[i][0] == board[i][1] == board[i][2] == player:
                    return True
                if board[0][i] == board[1][i] == board[2][i] == player:
                    return True
                
            if board[0][0] == board[1][1] == board[2][2] == player:
                return True
            if board[0][2] == board[1][1] == board[2][0] == player:
                return True
            
            return False
        
        if isWinner("X") and (count_X - count_O != 1):
            return False
        if isWinner("O") and (count_X != count_O):
            return False
        
        # 包括了有人赢，或者没有人赢但是合理的情况
        return True
```

# Others

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

```py
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        """
        用index标记每个九宫格，index=(r//3, c//3)；或者index=3 * (r//3) + c//3

        时间：O(9^2)
        空间：O(9^2)
        """
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

[205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/)

```py
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        """
        用map记录映射关系，用set保证只有映射是唯一的

        时间：O(N)
        空间：O(1) 最多就是26个字母
        """
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

[242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)

```py
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        """
        count the frequency of each letter in s, iterate through t while decreament the counter. 
        keep in mind of cases: 
        s = "a", t = "ab"
        s = "ab", t = "a"
        """
        counter = collections.Counter(s)

        for ch in t:
            counter[ch] -= 1
            if counter[ch] < 0:
                return False
        
        return len(s) == len(t)
```
