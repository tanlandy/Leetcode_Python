前缀和本质上是在一个list当中，用O（N）的时间提前算好从第0个数字到第i个数字之和，在后续使用中可以在O（1）时间内计算出第i到第j个数字之和

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
