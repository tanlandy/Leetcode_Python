[263. Ugly Number]([Loading...](https://leetcode.com/problems/ugly-number/))

```py
class Solution:
    def isUgly(self, n: int) -> bool:
        """
        keep dividing n by 2,3,5 until it becomes 0
        """
        
        for d in 2, 3, 5:
            while n % d == 0 and 0 < n:
                n /= d
        
        return n == 1
```

[326. Power of Three](https://leetcode.com/problems/power-of-three/)

```py
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        """
        naive: n = 3 * 3 * 3 * 3... * 3
        """
        if n < 1:
            return False
        
        while n % 3 == 0:
            n /= 3
        
        return n == 1
```

[169. Majority Element](https://leetcode.com/problems/majority-element/)

```py
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        """
        we maintain a count, which is incremented whenever we see an instance of our current candidate for majority element and decremented whenever we see anything else. 
        Whenever count equals 0, we effectively forget about everything in nums up to the current index and consider the current number as the candidate for majority element.
        """
        count = 0
        res = None
        
        for n in nums:
            if count == 0:
                res = n
            count += (1 if n == res else -1)
        
        return res
```