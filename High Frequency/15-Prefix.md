# 基础知识
前缀和本质上是在一个list当中，用O（N）的时间提前算好从第0个数字到第i个数字之和，在后续使用中可以在O（1）时间内计算出第i到第j个数字之和

# 题目

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

[560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)


[1074. Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)


[1423. Maximum Points You Can Obtain from Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/)


[1031. Maximum Sum of Two Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)
