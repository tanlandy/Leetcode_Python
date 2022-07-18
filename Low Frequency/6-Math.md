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

[149. Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/)

```py
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        """
        Given a point p, we compute the slopes of all lines connecting p and other points. Points corresponding to the same slope will fall on the same line.
        In this way, we can figure out the maximum number of points on lines containing p

        Time: O(N^2)
        Space: O(N)
        """
        res = 0
        
        for i, (x1, y1) in enumerate(points):
            slopes = collections.defaultdict(int)
            for j, (x2, y2) in enumerate(points[i + 1:]):
                slope = (y2 - y1) / (x2 - x1) if x1 != x2 else float("inf")
                slopes[slope] += 1
                res = max(res, slopes[slope])
        
        # 最后还要加上自己的这个点
        return res + 1
```

[166. Fraction to Recurring Decimal](https://leetcode.com/problems/fraction-to-recurring-decimal/)


[29. Divide Two Integers](https://leetcode.com/problems/divide-two-integers/)

