[496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)

[503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)

[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

[907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/)


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

[239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

```py
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        monotonic decreasing queue

        时间：O(N)
        空间：O(N)
        """
        res = []
        l = r = 0
        queue = collections.deque() # store index

        while r < len(nums):
            while queue and nums[queue[-1]] < nums[r]:
                queue.pop()
            queue.append(r)

            if l > queue[0]:
                queue.popleft()

            if r + 1 - l == k:
                res.append(nums[queue[0]])
                l += 1
            r += 1

        return res
```

[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)


[85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)


[901. Online Stock Span](https://leetcode.com/problems/online-stock-span/)
