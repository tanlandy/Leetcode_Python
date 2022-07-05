# 基础知识

## Intro
Dynamic programming is an algorithmic optimization technique that breaks down a complicated problem into smaller overlapping subproblems in a recursive manner and use solutions to the subproblems to construct solution to the original problem.

-> It is a simple concept of solving bigger problems using smaller problems while saving results to avoid repeated calculations. 

1. The problem can be broken down into `"overlapping subproblems"` - smaller versions of the original problem that are re-used multiple times. -> subproblems are dependent
2. The problem has an `"optimal substructure"` - an optimal solution can be formed from optimal solutions to the overlapping subproblems of the original problem.


### Characteristics of DP
1. Optimal substructure: the problem can be divided into subproblems. And its optimal solution can be constructed from optimal solutions of the subproblems
2. The subproblems overlap




### Greedy vs DP
In greedy, we always want to choose the best answer
DP: is not always necessarily the best answer for every state

## When to use
DP is an **optimization** method on one or more **sequences**.
1. The problem asks for the maximum/longest, minimal/shortest value/cost/profit you can get from doing operations on a sequence.
2. You've tried greedy but it sometimes gives the wrong solution. This often means you have to consider subproblems for an optimal solution.
3. The problem asks for how many ways there are to do something. This can often be solved by DFS + memoization, i.e. top-down dynamic programming.
4. Partition a string/array into sub-sequences so that a certain condition is met. This is often well-suited for top-down dynamic programming.
5. The problem is about the optimal way to play a game.
6. Future decisions depend on earlier decision
   - House Robber
   - LIS

## How to use
1. Top-down: DFS + Memorization: split large problems and recursively solve smaller subproblems
   - draw the tree
   - identify states
    1. 站在节点：需要什么来解决问题，如何解决
    2. 站在节点：需要什么信息来确定如何往下走
   - DFS + Memoization
     - memoizing a result means to store the result of a function call, usually in a hashmap or an array,
2. Bottom-up: solve subproblmes first, and then use their solution to find the solutions to bigger subproblems -> normally done in a tabular form -> start at the base case
    - 找到recurrence relation，例如dp[i] = dp[i-1] + dp[i-2]

不论top-down还是bottom-up，都要思考
1. A function or data structure that will compute/contain the answer to the problem for every given state

2. A recurrence relation to transition between states: A state is a set of variables that can `sufficiently` describe a scenario. These variables are called state variables
`finding the recurrence relation is the most difficult part of solving a DP problem`

3. Bases cases
What state(s) can I find the answer to without using dynamic programming? 


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

# Problems

## Sequence
dp[i] normally means max/min/best value of the sequnce ending at index i

### Algo
[198. House Robber](https://leetcode.com/problems/house-robber/)

```py
class Solution:
    def rob(self, nums: List[int]) -> int:
        """
        dp[i] means the max value we can get using elements from idx 0 up to i
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
        DP bottom-up:
        DP[i] = min number of coins needed to count to i
        dp[i] = min(1 + dp[i - each_coin])
        dp[amount] is return value

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

## Grid
This is 2D version of the sequence DP. dp[i][j] means max/min/best value for matrix cell ending at index i, j
### Algo
[62. Unique Paths](https://leetcode.com/problems/unique-paths/)
```py
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """
        number of path to a cell = number of path to its left + to its tops
        dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        base case: dp[r][c]的第一行和第一列都是1
        最后返回dp[-1][-1]
        """

        dp = [[0 for _ in range(n)] for _ in range(m)]

        for c in range(n):
            dp[0][c] = 1
        for r in rnage(m):
            dp[r][0] = 1
        
        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        
        return dp[-1][-1]

```

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


[64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
```py
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        dp[r][c] is the minumum path sum to (r, c)
        dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1])
        
        Time: O(M*N)
        Space: O(M*N)
        """
        
        rows, cols = len(grid), len(grid[0])
        
        dp = [[0 for _ in range(cols)] for _ in range(rows)]
        
        dp[0][0] = grid[0][0]
        for r in range(1, rows):
            dp[r][0] = grid[r][0] + dp[r-1][0]
        for c in range(1, cols):
            dp[0][c] = grid[0][c] + dp[0][c-1]
            
        for r in range(1, rows):
            for c in range(1, cols):
                dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1])
        
        return dp[-1][-1]
```

[221. Maximal Square](https://leetcode.com/problems/maximal-square/)



## Dynamic number of subproblems
Similar to Sequence DP, except dp[i] depends on a dynamic number of subproblems: dp[i] = max(d[j]) from 0 to i
### Algo

## Partition
This is a continuation of DFS + Memoization problems. The key is to draw the state-space tree and then traverse it
### Algo

## Interval
Find subproblem defined on an interval dp[i][j]
### Algo

## Two sequences
dp[i][j] represents the max/min/best value for the first sequence ending in index i and second sequence ending in index j
可能类似Grid，之后确认
### Algo

## Game theory
This asks for whether a player can win a decision game. Key is to identify winning state, and formulate a winning state as a state that returns a losing state to the opponent
### Algo

## 0-1 Knapsack
### Algo

## Bitmask
Use bitmasks to reduce factorial compelxity to 2^n by encoding the dp state in bitmasks
### Algo







## Other待分类














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



### Neetcode.io
#### 1D DP
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

```py
class Solution:
    def numDecodings(self, s: str) -> int:
        """
        Time: O(N)
        Space: O(N)
        """
        memo = {}
        
        def dfs(idx):
            if idx in memo:
                return memo[idx]
            
            # 走到头了
            if idx == len(s):
                return 1
            
            # 这个string以0开头
            if s[idx] == "0":
                return 0
            
            # 走到前一位：只有1种方式了
            if idx == len(s) - 1:
                return 1
            
            res = dfs(idx + 1)
            if int(s[idx: idx + 2]) <= 26:
                res += dfs(idx + 2)
            
            memo[idx] = res       
                 
            return res
        
        return dfs(0)        
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
        dp[i] means s[i:] whether can be formed by words in wordDict or not

        From right to left
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

```py
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        DFS + Memo
        
        Time: O(N^3)
        Space: O(N)
        """
        if not s:
            return False
        words = set(wordDict)
        memo = {}
        
        def dfs(s):
            if s in memo:
                return memo[s]
            if not s:
                return True
            for word in words:
                # 前面不同就跳过
                if s[:len(word)] != word:
                    continue
                # 前面相同就可以往后看
                remain = dfs(s[len(word):])
                if remain:
                    memo[s] = True # 保存remain的结果
                    return True
            memo[s] = False
            return False
        
        return dfs(s)
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

[121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        two pointers: always try to pick the smallest starting point, while calculating the potentional profit: when find a smaller one, change the pointer l to that one.
        """
        l, r = 0, 1
        res = 0
        cur_profit = 0
        
        while r < len(prices):
            if prices[l] > prices[r]:
                l = r
            else:
                cur_profit = prices[r] - prices[l]
                res = max(res, cur_profit)
            r += 1
        
        return res
```


#### 2D DP




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

