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