[1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit]([Loading...](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/))

```py
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        """
        use sliding window, and two monotonic queues to keep track of min and max in the window. store index in deques instead of num
        
        Time: O(N)
        Space: O(N)
        """
		min_deque, max_deque = deque(), deque()
        l = r = 0
        res = 0
        
        while r < len(nums):
            while min_deque and nums[r] <= nums[min_deque[-1]]:
                min_deque.pop()
            while max_deque and nums[r] >= nums[max_deque[-1]]:
                max_deque.pop()
            min_deque.append(r)
            max_deque.append(r)
            
            while nums[max_deque[0]] - nums[min_deque[0]] > limit:
                l += 1
                if l > min_deque[0]:
                    min_deque.popleft()
                if l > max_deque[0]:
                    max_deque.popleft()
            
            res = max(res, r - l + 1)
            r += 1
                
        return res
        
```
