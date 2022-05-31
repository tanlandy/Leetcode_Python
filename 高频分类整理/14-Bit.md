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
        


```

