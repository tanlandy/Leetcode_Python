[70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

```py
class Solution:
    def climbStairs(self, n: int) -> int:
        """
        bottom-up DP，从最后往最前面
        """
        if n == 1:
            return 1
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i-1] + dp[i-2]            
        
        return dp[n]
```

```py
class Solution:
    def climbStairs(self, n: int) -> int:
        """
        可以不把所有的中间结果存下来，只用两个变量来记录

        时间：O(N)
        空间：O(1)
        """
        one = two = 1
        
        for i in range(n - 1):
            tmp = one
            one = one + two
            two = tmp
        
        return one
```        
