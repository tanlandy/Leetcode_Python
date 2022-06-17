# 算法笔记
## 框架
动态规划的一般形式就是求最值，求最值的核心就是穷举
-> 列出正确的**状态转移方程**，从而正确地穷举
-> 利用**最优子结构**，通过子问题的最值得到原问题的最值
-> 利用DP Table，优化**重叠子问题**的穷举过程

框架：
```py
# 自顶向下递归的动态规划
def dp(状态1, 状态2, ...):
    for 选择 in 所有可能的选择:
        # 此时的状态已经因为做了选择而改变
        result = 求最值(result, dp(状态1, 状态2, ...))
    return result

# 自底向上迭代的动态规划
# 初始化 base case
dp[0][0][...] = base case
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)
```

[509. Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)

```py
class Solution:
    def fib(self, n: int) -> int:
        """
        暴力解法存在大量的重叠子问题
        dp[i] = dp[i-1] + dp[i-2]
        
        Time: O(N)
        Space: O(N)
        """
        if n == 0:
            return 0
        if n == 1:
            return 1
        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 1
        
        for i in range(2, n+1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
```

```py
class Solution:
    def fib(self, n: int) -> int:
        """
        只用三个数来代替整个table
        
        时间：O(N)
        空间：O(1)
        """
        if n <= 1:
            return n
        
        cur = 0
        prev1 = 1
        prev2 = 0
        
        for i in range(2, n + 1):
            cur = prev1 + prev2
            prev2 = prev1
            prev1 = cur
        
        return cur
```

[322. Coin Change](https://leetcode.com/problems/coin-change/)

```py
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        具有最优子结构，子问题相互独立 -> 确定是动态规划问题

        1. 确定base case：amount是0
        2. 确定状态：原问题和子问题的变化的量：目标金额amount
        3. 确定选择：导致状态发生变化的行为
        4. DP数组的定义

        cannot be greedy: choose from the largest to the smallest as the total count is not guaranteed to be smallest
        BF: backtracking using desicion tree
        DP bottom-up: DP[i] = min number of coins it takes to count to i
        DP[i] = min( one_coin_used + DP[i - value_one_coin_used])
        
        [1,3,4,5]
        DP[7] = min(1 + DP[6], 1 + DP[4], 1 + DP[3], 1 + DP[2])

        Time: O(amount * len(coins))
        Space: O(amount)
        """
        dp = [float("inf")] * (amount + 1) # go from 0 to amount

        dp[0] = 0 # base case

        for i in range(1, amount + 1): # need to calc dp[amnout], so range(1, amount + 1)
            for c in coins:
                if i - c >= 0:
                    dp[i] = min(dp[i], 1 + dp[i - c])
        
        return dp[amount] if dp[amount] != float("inf") else -1

```


[300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

```py
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        BF: generate all subsequences: for each value: 2 choices: include or not include -> 2^N all sequencies
        -> draw a decision tree and memorize the
        DP: right to left: dp[i] stores the LIS starting at index i
        dp[i] = max(1, 1+dp[i+1], 1+dp[i+2], ...), depends on whether dp[i] < dp[i+1}

        Time: O(N^2)
        Space: O(N)
        """
        LIS = [1] * len(nums) # base case

        for i in range(len(nums) - 1, -1, -1): 
            for j in range(i + 1, len(nums)):
                if nums[i] < nums[j]: 
                    LIS[i] = max(LIS[i], 1 + LIS[j]) # transition of the given state
        
        return max(LIS)

```





















# Explore

## What is DP

1. The problem can be broken down into `"overlapping subproblems"` - smaller versions of the original problem that are re-used multiple times. -> subproblems are dependent
2. The problem has an `"optimal substructure"` - an optimal solution can be formed from optimal solutions to the overlapping subproblems of the original problem.

## Top-down and Bottom-up
### Bottom-up -> runtime is faster
implemented with iteration and starts at the base case

### Top-down -> easier to write 想象决策树
implemented with recursion and made efficient with memoization -> recursion tree
> memoizing a result means to store the result of a function call, usually in a hashmap or an array, so that when the same function call is made again, we can simply return the memoized result instead of recalculating the result. 

## When to use
1. Ask for the optimum value (maximum or minimum) of something, or the number of ways there are to do somethings
2. Future decisions depend on earlier decision
   - House Robber
   - LIS

## Framework for DP problems
State. In a DP problem, a state is a set of variables that can `sufficiently` describe a scenario. These variables are called state variables

Climbing Stairs, there is `only` 1 relevant state variable, the current step we are on. We can denote this with an integer \text{i}i. If \text{i = 6}i = 6, that means that we are describing the state of being on the 6th step. Every unique value of \text{i}i represents a unique state.

### Framework
1. a function or data structure that will compute/contain the answer to the problem for every given state
for Climbing Stairs, we have a function dp where dp[i] returns the number of ways to climb to the ith step. Solving the original problem would be to return dp[n] - literally the original problem, but generalized for a given state.
> Typically, top-down is implemented with a recursive function and hash map, whereas bottom-up is implemented with nested for loops and an array. When designing this function or array, we also need to decide on state variables to pass as arguments. 

2. A recurrence relation to transition between states
`finding the recurrence relation is the most difficult part of solving a DP problem`

3. Bases cases
What state(s) can I find the answer to without using dynamic programming? 

memoization means caching results from function calls and then referring to those results in the future instead of recalculating them. This is usually done with a hashmap or an array.


```py
class Solution:
    def climbStairs(self, n: int) -> int:
        def dp(i):
            if i <= 2: 
                return i
            if i not in memo:
                # Instead of just returning dp(i - 1) + dp(i - 2), calculate it once and then
                # store the result inside a hashmap to refer to in the future.
                memo[i] = dp(i - 1) + dp(i - 2)
            
            return memo[i]
        
        memo = {}
        return dp(n)
```

# 题目

## Neetcode.io
### 1D DP
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
        dp[1] = 1 # base case
        dp[2] = 2 # base case
        for i in range(3, n + 1):
            dp[i] = dp[i-1] + dp[i-2] # recurrence relation          
        
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

[746. Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)

```py
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """
        Draw a desicion tree for dp problems, the depth of the tree is len(cost), time: O(2^N) -> O(N) with memorization
        Solve problem from right to left, each time add the minimum of a one jump or a two step jump to the dp
        dp[i] += min(dp[i + 1], dp[i + 2])

        Time: O(N)
        Space: O(1) as use only two variables
        """

        # add a 0 to the end
        cost.append(0)

        # [10, 15, 20], 0
        for i in range(len(cost) - 3, -1, -1): # have to start at 15 instead of 20, as there would be out of bound for 20
            cost[i] += min(cost[i + 1], cost[i + 2])

        return min(cost[0], cost[1])

```

[198. House Robber](https://leetcode.com/problems/house-robber/)

```py
class Solution:
    def rob(self, nums: List[int]) -> int:
        """
        dp[i] = max(dp[i-2]+nums[i], dp[i-1]): the current dp[i] is determined by whether add this nums[i] or not

        Time: O(N)
        Space: O(N)
        """
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        
        return dp[-1]
```

```py
class Solution:
    def rob(self, nums: List[int]) -> int:
        """
        BF: list all combinations using decision tree
        Identify subproblems: if choose nums[0], then find the max from nums[2] to the end; else, find the max from nums[1:]
        rob = max(nums[0] + rob[2:], rob[1:])
        dp from the beginning to the end, each time only consider the larger one using two variables

        Time: O(N)
        Space: O(1)
        """
        rob1, rob2 = 0, 0

        # [rob1, rob2, n, n + 1, ...]
        for n in nums:
            # calc the max up until n
            tmp = max(n + rob1, rob2)
            rob1 = rob2
            rob2 = tmp
        
        return rob2

```

[213. House Robber II](https://leetcode.com/problems/house-robber-ii/)

```py
class Solution:
    def rob(self, nums: List[int]) -> int:
        """
        only restriction is cannot use nums[0] and nums[-1] at the same time: 
        run the LC198 two times  on nums[0:-2] and nums[1:-1], return the max of these 

        Time: O(N)
        Space: O(1)
        """
        if len(nums) == 1:
            return nums[0]

        res1 = self.helper(nums[0: -1])
        res2 = self.helper(nums[1:])
        return max(res1, res2)

    def helper(self, nums):
        rob1, rob2 = 0, 0

        for n in nums:
            tmp = max(rob1 + n, rob2)
            rob1 = rob2
            rob2 = tmp
        
        return rob2
```

[5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

```py
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        find through middle to the beginning and end
        
        Time: O(n^2) : nested loop, each time need O(N), and findPalindrome O(N) times
        Space: O(1)
        """
        res = ""
        
        def findPalindrome(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l+1: r] # 多走了一步，所以上一步是满足要求的大小
        
        for i in range(len(s)):
            # odd length
            s1 = findPalindrome(i, i)
            
            # even length
            s2 = findPalindrome(i, i + 1)
            
            if len(s1) > len(res):
                res = s1
            if len(s2) > len(res):
                res = s2
                
        return res
```

[647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

```py
class Solution:
    def countSubstrings(self, s: str) -> int:
        """
        Same as LC5, use a helper() for calc and return the count of Palidrome given the current idx

        Time: O(N^2)
        Space: O(1)
        """
        def countPali(l, r):
            one_res = 0
            while l >= 0 and r < len(s) and s[l] == s[r]:
                one_res += 1
                l -= 1
                r += 1
            return one_res
        
        res = 0
        for i in range(len(s)):
            res += countPali(i, i)
            res += countPali(i, i + 1)
        
        return res       
            
```

[91. Decode Ways](https://leetcode.com/problems/decode-ways/)

```py
class Solution:
    def numDecodings(self, s: str) -> int:
        """
        BF: when str has more than two digits: draw a desicion tree
        Example: "121" can only branch to 1-26 -> O(2^N)
                 121
             /          \
            1            12
          /   \         /
         2    21       1
        /
        1

        subproblem: once solve 21, the subproblem is 1, solve from right to left
        dp[i] = dp[i + 1] + dp[i + 2]

        Time: O(N)
        Space: O(N), O(1) if only use two variables
        """
        dp = [1] * (len(s) + 1)

        for i in range(len(s) - 1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]

            if ((i + 1) < len(s)) and ((s[i] == "1") or s[i] == "2" and s[i + 1] in "0123456"): # double digit
            # if 10 <= int(s[i:i+2]) <= 26:
                dp[i] += dp[i + 2]
        
        return dp[0]
```

[322. Coin Change](https://leetcode.com/problems/coin-change/) 前有


[152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

```py
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        """
        BF: find all the products-> O(N^2)
        DP: 
        all positive: product increasing
        all negative: odd products smaller; even products larger: the sign is alternating
        subproblem is to find the max and min of the previous several elements
        when see 0, reset max and min to 1 to ignore 0

        Time: O(N)
        Space: O(1)
        """
        res = max(nums) # cannot as 0, as the input can be  [-1]
        cur_min, cur_max = 1, 1

        for n in nums:
            if n == 0: # 这个判断也可以不要，因为这样的话下次默认就是cur_max = cur_min = n
                cur_min, cur_max = 1, 1
                continue
            tmp = cur_max * n
            cur_max = max(n * cur_max, n * cur_min, n) # three senarios: [1,2,3], [1,-2,-3], [-1, -2, 3]
            cur_min = min(tmp, n * cur_min, n)
            res = max(cur_max, cur_min, res)
        
        return res
```

index of the maxProduct
```py
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        """
        first find the end index, then go from right to left to find the beginning

        Time: O(N)
        Space: O(1)
        """
        cur_min, cur_max = nums[0], nums[0]
        res = nums[0] # cannot be max(nums), as the max(nums) can be result and in this case the index is missed
        end_idx = 0

        # get the end_idx
        for i in range(1, len(nums)): # use i to get the index
            cur = nums[i]
            tmp = cur_max * cur
            cur_max = max(cur_max * cur, cur_min * cur, cur)
            cur_min = max(tmp, cur_min * cur, cur)
            if cur_max > res:
                res = cur_max
                end_idx = i 
        
        # get the begin_idx
        prod = nums[end_idx]
        begin_idx = end_idx - 1
        while prod != res and begin_idx >= 0:
            prod *= nums[begin_idx]
            begin_idx -= 1
        
        print(nums[begin_idx + 1: end_idx + 1])
        return res

```

[139. Word Break](https://leetcode.com/problems/word-break/)

```py
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        dp[] stores given index is true or false

        Time: O(N*M*N), N is len(s), M is len(wordDict)
        Space: O(N+M)
        """
        
        dp = [False] * (len(s) + 1)
        dp[len(s)] = True
        
        for i in range(len(s) - 1, -1, -1):
            for w in wordDict:
                if (i + len(w) <= len(s)) and s[i:i + len(w)] == w:
                    dp[i] = dp[i + len(w)] # at idx i, dp[i] determines at dp[i+len(w)] if s[i:i+len(w)] == w
                if dp[i]:
                    break
        return dp[0]
```

[300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) 前面出现


[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)

```py
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        """
        for every element, include or not include -> decision tree -> 2^n
        Time: O(N*Sum(nums))
        Space: 
        """

        if sum(nums) % 2 == 1:
            return False
        
        dp = set()
        dp.add(0)
        target = sum(nums) // 2
        for i in range(len(nums) - 1, -1, -1):
            newDP = set()
            for t in dp:
                newDP.add(t + nums[i])
                newDP.add(t)
            dp = newDP
        return True if target in dp else False

```

### 2D DP

[62. Unique Paths](https://leetcode.com/problems/unique-paths/)
```py
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """
        dp[r][c] = right + down, each one stores the number of unique paths

        Time: O(M*N)
        Space: O(N)
        """
        row = [1] * n
        
        for i in range(m - 1): # wait until first row
            newRow = [1] * n 
            # out of range
            for j in range(n - 2, -1, -1):
                newRow[j] = newRow[j + 1] + row[j]
            row = newRow
        
        return row[0]
        
```


[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

```py
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]
        
        for i in range(len(text1) - 1, -1, -1):
            for j in range(len(text2) - 1, -1, -1):
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i][j + 1], dp[i + 1][j])
        
        return dp[0][0]   

```

