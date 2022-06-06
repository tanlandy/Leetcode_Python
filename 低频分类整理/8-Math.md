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
