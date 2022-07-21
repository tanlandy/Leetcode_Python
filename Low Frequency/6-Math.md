# 基础知识

## 除法相关

20 / 3 = 6 ... 2
20: dividend
3: divisor
6: quotient
2: remainder

range of primitive number in Java and C++: (32-bit number)
[-2 ** 31, 2 ** 31 - 1]
abs(-2**31) = 2 ** 31, which is out of bound

# 例题

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

```py
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        """
        Idea is to put every remainder into the hash table as a key, and the current length of the result string as the value. When the same remainder shows again, it's circulating from the index of the value in the table.
        """
        if numerator % denominator == 0:
            return str(numerator//denominator)
        sign = "" if numerator * denominator >= 0 else "-"
        numerator, denominator = abs(numerator), abs(denominator)
        res = sign + str(numerator//denominator) + '.' # finished before "." part
        # the decimal part
        numerator %= denominator
        i, part = 0, ''
        rem_size = {numerator: i}
        while numerator % denominator != 0:
            numerator *= 10
            i += 1
            rem = numerator % denominator
            part += str(numerator // denominator)
            if rem in rem_size: # 如果重复出现过，就可以返回结果
                part = part[:rem_size[rem]]+"("+part[rem_size[rem]:]+")"
                return res + part
            rem_size[rem] = i
            numerator = rem
        return res + part
```

[29. Divide Two Integers](https://leetcode.com/problems/divide-two-integers/)

```py
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        """
        Linear scan: slow, but basic
        """
        MAX_INT = 2 ** 31 - 1
        MIN_INT = -2 ** 31
        
        # overflow: 32 bit integer: [[-2 ** 31, 2 ** 31 - 1]]
        if dividend == MIN_INT and divisor == -1:
            return MAX_INT
        
        # convert both numbers to negative, as it has a larger scope
        negatives = 2
        if dividend > 0:
            negatives -= 1
            dividend = -dividend
        if divisor > 0:
            negatives -= 1
            divisor = -divisor
        
        # count how many times the divisor has to be added
        quotient = 0
        while dividend - divisor <= 0:
            quotient -= 1
            dividend -= divisor
        
        return -quotient if negatives != 1 else quotient
```

```py
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        """
        try and subtract multiple copies of the divisor each time: double the divisor each time
        
        (10, 3)             while   while
        negatives = 0                           
        dividend = -10          -4      -1
        divisor = -3
        quotient = 0            -2      -3  return 3
        power_of_two = -1   -2      -1
        value = -3          -6      -3  
        Time: O(logN)
        Space: O(1)
        """
        MAX_INT = 2 ** 31 - 1
        MIN_INT = -2 ** 31
        HALF_MIN_INT = MIN_INT // 2
        
        if dividend == MIN_INT and divisor == -1:
            return MAX_INT
        
        negatives = 2
        if dividend > 0:
            negatives -= 1
            dividend = -dividend
        if divisor > 0:
            negatives -= 1
            divisor = -divisor
        
        quotient = 0
        while divisor >= dividend:
            power_of_two = -1
            value = divisor
            
            # while away from divisor, move nearer
            while value >= HALF_MIN_INT and value + value >= dividend:
                value += value
                power_of_two += power_of_two
            
            # subtract divisor another power_of_two times
            quotient += power_of_two
            # remove value so far, and continue to deal with remainder
            dividend -= value
        
        return -quotient if negatives != 1 else quotient
```

[172. Factorial Trailing Zeroes](https://leetcode.com/problems/factorial-trailing-zeroes/)

```py
class Solution:
    def trailingZeroes(self, n: int) -> int:
        """
        末尾的0肯定来源于因子2*5，因为因子2比5多得多，所以只考虑n最后能分解到多少个因子5
        25!中，5可以提供1个，10可以1个，15可以1个，20可以1个，25可以2个，总共有6个因子5，所以结果末尾有6个0
        对于125!，除了每5可以提供一个，每25还可以提供多一个5，每125又可以提供多一个5
        """
        res = 0
        divisor = 5
        
        while divisor <= n:
            res += (n // divisor)
            divisor *= 5
        
        return res
```

## Geometry

[939. Minimum Area Rectangle](https://leetcode.com/problems/minimum-area-rectangle/)

```py
class Solution:
    def minAreaRect(self, points: List[List[int]]) -> int:
        point_set = set()
        
        for x, y in points:
            point_set.add((x, y))
        
        res = float("inf")
        for x1, y1 in points:
            for x2, y2 in points:
                if x1 > x2 and y1 > y2 and (x1, y2) in point_set and (x2, y1) in point_set:
                    area = abs(x1 - x2) * abs(y1 - y2)
                    res = min(res, area)
        
        return res if res != float("inf") else 0
```

[593. Valid Square](https://leetcode.com/problems/valid-square/)

```py
class Solution:
    def validSquare(self, p1, p2, p3, p4):
        """
        先算出来所有边的大小，然后看最小的4条是否相等，相等的话就看对角线是否相等
        """
        if p1==p2==p3==p4:return False
        def dist(x,y):
            return (x[0]-y[0])**2+(x[1]-y[1])**2
        ls=[dist(p1,p2),dist(p1,p3),dist(p1,p4),dist(p2,p3),dist(p2,p4),dist(p3,p4)]
        ls.sort()
        if ls[0]==ls[1]==ls[2]==ls[3]:
            if ls[4]==ls[5]:
                return True
        return False
```

