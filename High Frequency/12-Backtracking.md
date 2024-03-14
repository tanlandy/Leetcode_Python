# 回溯

# Template

```shell
function dfs(node, state):
    if state is a solution:
        report(state) # e.g. add state to final result list
        return

    for child in children:
        if child is a part of a potential solution:
            state.add(child) # make move
            dfs(child, state)
            state.remove(child) # backtrack
```

## 回溯整理总结

### 形式一、元素无重不可复选，即nums中的元素都是唯一的，每个元素最多只能被使用一次

```py
"""子集问题 LC77, 78, 90""" 
for i in range(start, len(nums)):
    one_res.append(nums[i])
    backtrack(i + 1, one_res)
    one_res.pop()

"""排列问题 LC46"""
for i in range(len(nums)):
    if used[i]:
        continue
    
    used[i] = True
    one_res.append(nums[i])
    backtrack(one_res)
    one_res.pop()
    used[i] = False
```

### 形式二、元素可重不可复选，即nums中的元素可以存在重复，每个元素最多只能被使用一次，其关键在于排序和剪枝

```py
"""子集问题 LC40"""
nums.sort()
for i in range(start, len(nums)):
    if i > start and nums[i] == nums[i - 1]:
        continue
    
    one_res.append(nums[i])
    backtrack(i + 1, one_res)
    one_res.pop()

"""排列问题 LC47"""
nums.sort()
for i in range(len(nums)):
    if used[i]:
        continue
    
    if i > start and nums[i] == nums[i - 1] and not used[i - 1]: # not used[i-1]保证了元素的相对位置的统一，永远都是2->2'->2"
        continue

    used[i] = True
    one_res.append(nums[i]
    backtrack(one_res)
    one_res.pop()
    used[i] = False
```

### 形式三、元素无重可复选，即nums中的元素都是唯一的，每个元素可以被使用若干次，只要删掉去重逻辑即可

```py
"""子集问题 LC39"""
for i in range(start, len(nums)):
    one_res.append(nums[i])
    backtrack(i, one_res) # 注意这里是i
    one_res.pop()

"""排列问题"""
for i in range(len(nums)):
    one_res.append(nums[i])
    backtrack(one_res)
    one_res.pop()

```

# 例题

[22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
添加close的条件：close<open

```py
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        """
        1. add ( if open < n
        2. add ) if close < open
        3. valid if open == close == n 
        """

        stack = []
        res = []
        
        def backtrack(openN, closedN):
            if openN == closedN == n: # base case
                res.append("".join(stack))
                return
            
            if openN < n:
                stack.append("(")
                backtrack(openN + 1, closedN)
                stack.pop()
            
            if closedN < openN:
                stack.append(")")
                backtrack(openN, closedN + 1)
                stack.pop()
        
        backtrack(0, 0)
        return res
```

[78. Subsets](https://leetcode.com/problems/subsets/)（子集 元素无重不可复选）
Given an integer array nums of unique elements, return all possible subsets (the power set).
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

```py
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        通过保证元素之间的相对顺序不变来防⽌出现重复的⼦集
        并不是满树，所以利用start来控制树的遍历

        时间：O(N*2^N) generate all subsets and copy them
        空间：O(N) use O(N) for one_res
        """
        res = []
        
        def backtrack(start, one_res):
            # 添加条件：每个中间结果都是最终结果
            res.append(one_res.copy())
            
            # 子集问题：i从start开始
            # 通过 start 参数控制树枝的遍历，避免产生重复的子集            
            for i in range(start, len(nums)):
                one_res.append(nums[i]) # 做选择
                backtrack(i+1, one_res)
                one_res.pop() # 撤销选择
        
        backtrack(0, [])
        return res
```

[77. Combinations](https://leetcode.com/problems/combinations/) （组合 元素无重不可复选）
Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].

You may return the answer in any order.

Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

```py
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        不是满树，使用start来控制树的遍历
        backtrack从1开始，到n+1为止（左闭右开）
        """
        res = []

        def backtrack(start, one_res):
            # 添加条件：长度是k
            if len(one_res) == k: 
                res.append(one_res.copy())
                return

            # 子集问题：i从start开始
            # 通过 start 参数控制树枝的遍历，避免产生重复的子集
            for i in range(start, n + 1): # 左闭右开区间，所以要n+1
                one_res.append(i)
                backtrack(i+1, one_res)
                one_res.pop()
            
        backtrack(1, []) # 从1开始
        return res
```

[46. Permutations](https://leetcode.com/problems/permutations/) 排列（元素无重不可复选）

Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        满树问题，剪枝条件是跳过重复使用的值：可以用used[]来记录使用过的值，也可以每次判断nums[i] in one_res
        """
        res = []
        used = [False] * len(nums)

        def backtrack(one_res):
            # 添加条件： 长度
            if len(one_res) == len(nums):
                res.append(one_res.copy())
                return
            
            # 满树问题：i从0开始
            for i in range(len(nums)):
                # 跳过不合法的选择，否则结果有[1,1,1],[1,1,2]...
                if used[i]:
                    continue
                
                used[i] = True
                one_res.append(nums[i])
                backtrack(one_res)
                one_res.pop()
                used[i] = False
        
        backtrack([])
        return res
```

[90. Subsets II](https://leetcode.com/problems/subsets-ii/)
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

```py
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """需要先进行排序，让相同的元素靠在一起，如果发现nums[i] == nums[i-1]，则跳过"""
        nums.sort()
        res = []
        
        def backtrack(start, one_res):
            # 添加条件：每个中间结果都是最终结果
            res.append(one_res.copy())
            
            # 子集问题：i从start开始
            for i in range(start, len(nums)):
                # 跳过不合法选择，否则有最终结果有两个[1,2]
                if i > start and nums[i] == nums[i-1]:
                    continue
                one_res.append(nums[i])
                backtrack(i+1, one_res)
                one_res.pop()
        
        backtrack(0, [])
        return res
```

[40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output:
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

```py
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """类似LC90，只是base case不同"""
        candidates.sort() # 元素可以重复，所以要排序
        res = []
        
        def backtrack(one_res, start, target):
            if target < 0:
                return
            
            if target == 0:
                res.append(one_res.copy())
            
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                
                one_res.append(candidates[i])
                target -= candidates[i]
                backtrack(one_res, i + 1, target)  # 从i+1开始，因为每个元素只能用一次
                one_res.pop()
                target += candidates[i]
        
        backtrack([], 0, target)
        return res
```

[39. Combination Sum](https://leetcode.com/problems/combination-sum/)

Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]

```py
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort() # 元素可重复，所以要排序
        
        def backtrack(one_res, start, target):
            if target < 0:
                return
            
            if target == 0:
                res.append(one_res.copy())
                return
            
            for i in range(start, len(candidates)):
                # 与LC40不同点：不怕重复，所以不需要选择条件
                one_res.append(candidates[i])
                target -= candidates[i]
                backtrack(i, target) # 与LC40不同点，可以重复，所以从i再用一次
                one_res.pop()
                target += candidates[i]
        
        backtrack([], 0, target)
        return res
```

[47. Permutations II](https://leetcode.com/problems/permutations-ii/)

Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]

```py
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        used = [False] * len(nums)
        
        def backtrack(one_res):
            if len(one_res) == len(nums):
                res.append(one_res.copy())
                return
        
            for i in range(len(nums)):
                if used[i]:
                    continue
                    
                # 新添加的剪枝逻辑，固定相同的元素在排列中的相对位置
                # not used[i-1]保证相同元素在排列中的相对位置保持不变。
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                    
                used[i] = True
                one_res.append(nums[i])
                backtrack()
                one_res.pop()
                used[i] = False
        
        backtrack([])
        return res
```

[79. Word Search](https://leetcode.com/problems/word-search/)

```py
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        dfs(r, c, i)同时传入一个idx

        时间：O(M*N*4 ^ N)
        空间：O(L) L is len(words)
        """
        rows, cols = len(board), len(board[0])
        visited = set()
        
        def dfs(r, c, i):
            # Base case
            if i == len(word):
                return True
            
            # 排除的条件
            if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in visited or board[r][c] != word[i]:
                return False
            
            # 做选择
            visited.add((r, c))
            # Backtrack
            res =  (dfs(r + 1, c, i + 1) or
                    dfs(r - 1, c, i + 1) or         
                    dfs(r, c + 1, i + 1) or         
                    dfs(r, c - 1, i + 1)
                  )
            # 回溯
            visited.remove((r, c))            
            return res
        
        for r in range(rows):
            for c in range(cols):
                if dfs(r, c, 0):
                    return True
        
        return False
```

[131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)

```py
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def isPalindrome(a, l, r):
            
            while l < r:
                if a[l] != a[r]:
                    return False
                l += 1
                r -= 1
            
            return True
        
        res = []
        one_res = []
        def backtrack(i):
            if i >= len(s):
                res.append(one_res.copy())
                return
            
            for j in range(i, len(s)):
                if isPalindrome(s, i, j):
                    one_res.append(s[i: j+1])
                    backtrack(j + 1)
                    one_res.pop()
        
        backtrack(0)
        return res
```

[17. Letter Combinations of a Phone Number](https://www.youtube.com/watch?v=0snEunUacZY)

时间：N*4^N, N is len(input):共4^N种组合，每组的长度为N
空间：

```py
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        d_ch = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "qprs",
            "8": "tuv",
            "9": "wxyz" 
        }
        
        res = []
        def backtrack(one_res, idx):
            if len(one_res) == len(digits):
                res.append("".join(one_res))
                return
            
            for ch in d_ch[digits[idx]]:
                one_res.append(ch)
                backtrack(one_res, idx + 1)
                one_res.pop()
        
        backtrack([], 0)
        return res
```

[51. N-Queens](https://leetcode.com/problems/n-queens/)

```py
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        """
        第一行，把第一个Queen放在每一个位置，这样一共就走N次
        Need to keep track: Queen的cols, pos_diag, neg_diag
        for the same neg_diag: (r - c) stays the same
        for the same pos_diag: (r + c) stays the same

                r
         /    /   \    \
        0,0  0,1  0,2  0,3
 /    /   \  \
1,0 1,1  1,2 1,3
        """
        col = set()
        pos_diag = set()
        neg_diag = set()

        res = []
        board = [["."] * n for i in range(n)]

        def backtrack(r): # go by row by row
            # base case
            if r == n:
                copy = ["".join(row) for row in board]
                res.append(copy)
                return 

            for c in range(n):
                if c in col or (r + c) in pos_diag or (r - c) in neg_diag:
                    continue
                
                col.add(c)
                pos_diag.add(r + c)
                neg_diag.add(r - c)
                board[r][c] = "Q"

                backtrack(r + 1)

                col.remove(c)
                pos_diag.remove(r + c)
                neg_diag.remove(r - c)
                board[r][c] = "."     

        backtrack(0)
        return res          

```

[698. Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/)

```py
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        """
        可能是TLE，但是好理解：
        每次都找数往bucket里放，直到k==0就返回
        时间：O(k * 2^N) 一个数的时间是2^N，找到一个数之后在这树下会还有k-1颗数
        """
        if sum(nums) % k:
            return False

        nums.sort(reverse=True)
        target = sum(nums) / k
        used = [False] * len(nums)

        def backtrack(i, k, cur_sum): # how many k are left
            if k == 0: # ultimate base case
                return True
            
            if cur_sum == target: # base case
                return backtrack(0, k - 1, 0) # 从idx0开始找
            
            for j in range(i, len(nums)):
                if used[j] or cur_sum + nums[j] > target:
                    continue
                    
                used[j] = True
                cur_sum += nums[j]
                if backtrack(j + 1, k, cur_sum):
                    return True
                cur_sum -= nums[j]
                used[j] = False
            return False
        
        return backtrack(0, k, 0)
```

[93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)

```py
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        self.res = []
        self.backtrack(s, [], 0)
        return self.res
    
    def backtrack(self, s, current, start):
        if len(current) == 4:
            if start == len(s):
                self.res.append(".".join(current))
            return
        for i in range(start, min(start+3, len(s))):
            if s[start] == '0' and i > start:
                continue
            if 0 <= int(s[start:i+1]) <= 255:
                self.backtrack(s, current + [s[start:i+1]], i + 1)
```

```py
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        self.backtrack(res, "", 0, s)
        return res
    
    def backtrack(self, res, path, idx, s): # s每次都往后截取
        if idx > 4:
            return 
        if idx == 4 and not s: # 4位数同时s走完了
            res.append(path[:-1])
        
        for i in range(1, len(s) + 1):
            if s[:i] == "0" or (s[0] != "0" and 0 < int(s[:i]) <= 255):
                self.backtrack(res, path + s[:i] + ".", idx + 1, s[i:])
```

# DFS + Memorization

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

[97. Interleaving String](https://leetcode.com/problems/interleaving-string/)

```py
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        """
        双指针遍历，遇到相同情况下走哪一个？->走每一个
        DFS + Memo
        
        时间：O(MN)
        空间：O(MN)
        """
        if len(s3) != len(s1) + len(s2):
            return False
        
        memo = {}
        
        def dfs(i, j, k):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == len(s1):
                return s2[j:] == s3[k:]
            if j == len(s2):
                return s1[i:] == s3[k:]
            
            if s1[i] == s3[k]:
                if dfs(i + 1, j, k + 1):
                    memo[(i, j)] = True
                    return True
            
            if s2[j] == s3[k]:
                if dfs(i, j + 1, k + 1):
                    memo[(i, j)] = True
                    return True
            
            memo[(i, j)] = False
            return False
        
        return dfs(0, 0, 0)
```

[526. Beautiful Arrangement](https://leetcode.com/problems/beautiful-arrangement/)

```py
class Solution:
    def countArrangement(self, N):
        """
        :type N: int
        :rtype: int
        """
        cache = {}
        def helper(X):
            if len(X) == 1:
                # Any integer can be divide by 1
                return 1
            
            if X in cache:
                return cache[X]
            total = 0
            for j in range(len(X)):
                if X[j] % len(X) == 0 or len(X) % X[j] == 0:
                    total += helper(X[:j] + X[j+1:])
                    
            cache[X] = total 
            return total 
        
        return helper(tuple(range(1, N+1)))
```

[[698. Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/) 好题
]

```py
class Solution:
    def canPartitionKSubsets(self, arr: List[int], k: int) -> bool:
        n = len(arr)
    
        total_array_sum = sum(arr)
        
        # If the total sum is not divisible by k, we can't make subsets.
        if total_array_sum % k != 0:
            return False

        target_sum = total_array_sum // k

        # Sort in decreasing order.
        arr.sort(reverse=True)

        taken = ['0'] * n
        
        memo = {}
        
        def backtrack(index, count, curr_sum):
            n = len(arr)
            
            taken_str = ''.join(taken)
      
            # We made k - 1 subsets with target sum and the last subset will also have target sum.
            if count == k - 1:
                return True
            
            # No need to proceed further.
            if curr_sum > target_sum:
                return False
            
            # If we have already computed the current combination.
            if taken_str in memo:
                return memo[taken_str]
            
            # When curr sum reaches target then one subset is made.
            # Increment count and reset current sum.
            if curr_sum == target_sum:
                memo[taken_str] = backtrack(0, count + 1, 0)
                return memo[taken_str]
            
            # Try not picked elements to make some combinations.
            for j in range(index, n):
                if taken[j] == '0':
                    # Include this element in current subset.
                    taken[j] = '1'
                    # If using current jth element in this subset leads to make all valid subsets.
                    if backtrack(j + 1, count, curr_sum + arr[j]):
                        return True
                    # Backtrack step.
                    taken[j] = '0'
                    
            # We were not able to make a valid combination after picking 
            # each element from the array, hence we can't make k subsets.
            memo[taken_str] = False
            return memo[taken_str] 
        
        return backtrack(0, 0, 0)
```

<https://leetcode.com/problems/longest-increasing-path-in-a-matrix/discuss/2052360/Python%3A-Beginner-Friendly-%22Recursion-to-DP%22-Intuition-Explained>

<https://leetcode.com/problems/out-of-boundary-paths/discuss/1293697/python-easy-to-understand-explanation-recursion-and-memoization-with-time-and-space-complexity>

<https://leetcode.com/problems/number-of-matching-subsequences/discuss/1289549/python-explained-all-possible-solutions-with-time-and-space-complexity>
