# 前缀和

# 基础知识

前缀和本质上是在一个list当中，用O（N）的时间提前算好从第0个数字到第i个数字之和，在后续使用中可以在O（1）时间内计算出第i到第j个数字之和

Find a number of continuous subarrays/submatrices/tree paths that sum to target

# 高频题

## 知乎

## Krahets精选题

## AlgoMonster

## Youtube

560, 238, 523, 370, 974

# 题目

[14. Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)

```py
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        
        shortest = min(strs, key = len) # find the possible one
        
        for i, ch in enumerate(shortest): # check idx one by one 
            for other in strs: # compare this idx with other strs one by one
                if other[i] != ch:
                    return shortest[:i] # as long as find an invalid, return
        
        return shortest
```

[2483. Minimum Penalty for a Shop](https://leetcode.com/problems/minimum-penalty-for-a-shop/description/)

```python
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        """
        假设从idx=0开始闭店，一个一个往右，如果当前ch=='Y'，那么相对的penalty应该减少1。、
        如果需要真实penalty，需要首先计算idx=0闭店的初始penalty值

        时间：O(n)
        空间：O(1)
        """
        cur_p = min_p = 0
        earliest_hour = 0

        for i, ch in enumerate(customers):
            if ch == 'Y':
                cur_p -= 1
            else:
                cur_p += 1
            
            if cur_p < min_p:
                earliest_hour = i + 1
                min_p = cur_p
        
        return earliest_hour
```

[560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

```py
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        """
        BF
        Time: O(N^2)
        Space: O(1)
        """
        res = 0
        
        for i in range(len(nums)):
            cur_sum = 0
            
            for j in range(i, len(nums)):
                cur_sum += nums[j]
                if cur_sum == k:
                    res += 1
        
        return res
```

```py
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        """
        Prefix Sum
        Time: O(N)
        Space: O(N)
        """
        cur_sum = 0
        count = 0
        prefix_freq = collections.defaultdict(int)
        
        for n in nums:
            cur_sum += n
            if cur_sum == k: # 第一种情况：从第一个数开始前缀和等于k
                count += 1
            count += prefix_freq[cur_sum - k] # 第二种情况：之前某个数i的从零开始前缀和是cur_sum - k，那么从i开始到这个数的和就是cur_sum - (cur_sum - k) = k
            prefix_freq[cur_sum] += 1 # 存下来当前前缀和出现的次数
        
        return count
```

[437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)

```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
        remove the last cur_sum from dict before processing the parallel subtree
        """
        prefix_freq = collections.defaultdict(int)
        res = [0]
        def dfs(node, cur_sum): # cur_sum is the prefix_sum from the previous node
            if not node:
                return
            
            cur_sum += node.val
            
            if cur_sum == targetSum:
                res[0] += 1
            res[0] += prefix_freq[cur_sum - targetSum]
            prefix_freq[cur_sum] += 1
            
            dfs(node.left, cur_sum)
            dfs(node.right, cur_sum)
            prefix_freq[cur_sum] -= 1
        
        dfs(root, 0)
        return res[0]
```

[53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

```py
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        """
        Clarify: will there be all negative nums? Null? Need to return the index?
        curA = max(nums[i], precurA + sum)
        """
        cur_sum = 0
        r = 0
        res = float("-inf")
        
        while r < len(nums):
            cur_sum += nums[r]
            res = max(res, cur_sum)
            r += 1
            
            if cur_sum < 0:
                cur_sum = 0
        
        return res
```

[523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)

```py
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        """
        clarify: k = 0? single element? array is null?
        """
        remainder = {0: -1}
        
        total = 0
        
        for i, n in enumerate(nums):
            total += n
            r = total % k
            
            if r not in remainder:
                remainder[r] = i
            else:
                if i - remainder[r] > 1:
                    return True
        
        return False
```

[528. Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/) (前缀和，可以先做一下LC53、523)

用list存所有的前缀和。概率是w[i]/total_sum，可以用找到第一个preSum来代替；用random.random()来获得[0,1);w:[1,3]-> pre_sums:[1, 4] -> target in randomly in [0, 4); find the first index in pre_sums s.t. target < pre_sums[idx]
时间：构造O(N)，找数O(N)
空间：构造O(N)，找数O(1)

```python
class Solution:

    def __init__(self, w: List[int]):
        self.prefix_sums = []
        pre_sum = 0
        for weight in w:
            pre_sum += weight
            self.prefix_sums.append(pre_sum)
        self.total_sum = pre_sum

    def pickIndex(self) -> int:
        target = self.total_sum * random.random()
        for i, pre_sum in enumerate(self.prefix_sums):
            if target < pre_sum:
                return i
```

用list存所有的前缀和。概率是w[i]/total_sum，可以用二分查找找到第一个preSum来代替；用random.random()来获得[0,1); 当右边左右的数都满足的时候，找最左满足的数，最后返回的是l
时间：构造O(N)，找数O(logN)
空间：构造O(N)，找数O(1)

```python
class Solution:
    def __init__(self, w: List[int]):
        self.prefix_sums = []
        pre_sum = 0
        for weight in w:
            pre_sum += weight
            self.prefix_sums.append(pre_sum)
        self.total_sum = pre_sum

    def pickIndex(self) -> int:
        target = self.total_sum * random.random()
        l, r = 0, len(self.prefix_sums) - 1
        while l <= r:
            mid = l + (r - l) // 2 # 要地板除
            if (target > self.prefix_sums[mid]):
                l = mid + 1
            else: 
                r = mid - 1
        return l
```

[303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/)

```py
class NumArray:

    def __init__(self, nums: List[int]):
        self.prefix = [0]
        cur = 0
        for n in nums:
            cur += n
            self.prefix.append(cur)

    def sumRange(self, left: int, right: int) -> int:
        # 如果一开始self.prefix = []
        # if left == 0:
        #     return self.prefix[right] 
        # return self.prefix[right] - self.prefix[left - 1]
        return self.prefix[right + 1] - self.prefix[left]
```

[304. Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/)

```py
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        """
        构建前缀和矩阵
        """
        rows, cols = len(matrix), len(matrix[0])
        
        self.prefix = [[0] * (cols + 1) for x in range(rows + 1)]
        
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                self.prefix[r][c] = self.prefix[r - 1][c] + self.prefix[r][c - 1] - self.prefix[r-  1][c - 1] + matrix[r - 1][c - 1]
        
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        画图知道方位
                 c1   c2        
              1   2   3
        r1    4   5   6
        r2    7   8   9
        sum(2, 2) = prefix(r2, c2) + prefix(r1-1, c2-1) - prefix(r2, c1-1) - prefix(r1-1, c2) 
        """
        row1 += 1
        row2 += 1
        col1 += 1
        col2 += 1
        return self.prefix[row2][col2] + self.prefix[row1 - 1][col1 - 1] - self.prefix[row1 - 1][col2] - self.prefix[row2][col1 - 1]

```

[930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)

```py
class Solution(object):
    def numSubarraysWithSum(self, A, S):
        P = [0]
        for x in A: P.append(P[-1] + x)
        count = collections.Counter()

        ans = 0
        for x in P:
            ans += count[x]
            count[x + S] += 1

        return ans
```

[1352. Product of the Last K Numbers](https://leetcode.com/problems/product-of-the-last-k-numbers/)

```py
class ProductOfNumbers:

    def __init__(self):
        """
        初始化数组为[1]
        关键是处理0
        -> 当是0的时候，直接重置
        """
        self.queue = [1]
        

    def add(self, num: int) -> None:
        if num == 0:
            self.queue = [1]
        else:
            self.queue.append(num * self.queue[-1])

    def getProduct(self, k: int) -> int:

        if k >= len(self.queue):
            return 0
            
        return self.queue[-1] // self.queue[-1 - k]

```

[1074. Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)

[1423. Maximum Points You Can Obtain from Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/)

[1031. Maximum Sum of Two Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)

[325. Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)
