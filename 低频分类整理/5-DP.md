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

[322. Coin Change](https://leetcode.com/problems/coin-change/)

```py
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        cannot be greedy: choose from the largest to the smallest as the total count is not guaranteed to be smallest
        BF: backtracking using desicion tree
        DP bottom-up: DP[i] = min number of coins it takes to count to i
        DP[i] = min(one_coin + DP[i-one_coin])
        
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