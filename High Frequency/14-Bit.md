[136. Single Number](https://leetcode.com/problems/single-number/)

```py
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
        XOR: 1^1 == 0, 0^0 == 0, n^0 == n

        Time: O(N)
        Space: O(1)
        """
        res = 0 # x ^ 0 = n
        for n in nums:
            res = n ^ res
        return res        

```










[338. Counting Bits](https://leetcode.com/problems/counting-bits/)

```py
class Solution:
    def countBits(self, n: int) -> List[int]:
        """
        1 + dp[n-i], i is offset, the most significant bit reached so far 
        """
        dp = [0] * (n + 1)
        offset = 1

        for i in range(1, n + 1):
            if offset * 2 == i:
                offset = i
            dp[i] = 1 + dp[i - offset]

        return dp
```

[190. Reverse Bits](https://leetcode.com/problems/reverse-bits/)
```py
class Solution:
    def reverseBits(self, n: int) -> int:
        """
        01 << 1: shift left by one: 010 -> 10
        Time: O(1) only 32 bits
        Space: O(1)
        """
        res = 0
        for i in range(32):
            bit = (n >> i) & 1 # >> to get spot one by one 
            res = res | (bit << (31 - i))

        return res

```

