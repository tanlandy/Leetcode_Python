[34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)


```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        二分查找，用一个boolean来判断是查左边还是右边，如果是左边就r=mid-1；另外用一个idx来记录最终值
        """
        left = self.binSearch(nums, target, True)
        right = self.binSearch(nums, target, False)
        return [left, right]

    def binSearch(self, nums, target, firstPos):
        l, r = 0, len(nums) - 1
        i = -1
        while l <= r:
            mid = l + (r - l) // 2
            if target > nums[mid]:
                l = mid + 1
            elif target < nums[mid]:
                r = mid - 1
            else:
                i = mid
                if firstPos:
                    r = mid - 1
                else:
                    l = mid + 1
        return i

```

[33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

```py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        """
        先判断是否找到，找不到的话，比较nums[l]和nums[mid]，从而找到接下来的二分查找区间：内部嵌套2个binary search

        时间：O(logN)
        空间：O(1)
        """
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if nums[mid] == target:
                return mid

            if nums[l] <= nums[mid]: # 说明mid左边是sorted
                if nums[l] <= target and target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1            
            else: # 说明mid右边是Sorted
                if nums[mid] < target and target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1

        return -1
```

[1095. Find in Mountain Array](https://leetcode.com/problems/find-in-mountain-array/)

```py

class Solution:
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        """
        find peak idx, then binary search left part and right part, respectively
    
        Time: O(logN)
        Space: O(1)
        """
        length = mountain_arr.length()
        
        # find peak
        l, r = 0, length - 1
        while l <= r:
            mid = (l + r) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                l = mid + 1
            else:
                r = mid - 1
        
        peak = l
        
        if mountain_arr.get(peak) == target:
            return peak
        
        # search left
        l, r = 0, peak - 1
        while l <= r:
            mid = (l + r) // 2
            mid_val = mountain_arr.get(mid)
            if mid_val < target:
                l = mid + 1
            elif mid_val > target:
                r = mid - 1
            else:
                return mid
        
        # search right
        l, r = peak + 1, length - 1
        while l <= r:
            mid = (l + r) // 2
            mid_val = mountain_arr.get(mid)
            if mid_val > target:
                l = mid + 1
            elif mid_val < target:
                r = mid - 1
            else:
                return mid
        
        return -1
```

[162. Find Peak Element](https://leetcode.com/problems/find-peak-element/)

```py
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        """
        tricky part is to check valid

        Time: O(logN)
        Space: O(1)
        """
        n = len(nums)
        l, r = 0, n - 1
        while l <= r:
            mid = (l + r) // 2
            if (mid == 0 or nums[mid-1] < nums[mid]) and (mid == n-1 or nums[mid] > nums[mid+1]):  # Found peak, also consider the edge case: beginning and end
                return mid
            elif mid == 0 or nums[mid-1] < nums[mid]:  # Find peak on the right
                l = mid + 1
            else:  # Find peak on the left
                r = mid - 1
        return -1

```

[278. First Bad Version](https://leetcode.com/problems/first-bad-version/)

```py
class Solution:
    def firstBadVersion(self, n: int) -> int:
        """
        Binary search for the most left valid one
        """
        l, r = 1, n
        
        while l <= r:
            mid = (l + r) // 2
            if isBadVersion(mid):
                r = mid - 1
            else:
                l = mid + 1
    
        return l
```

[74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)

```py
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        做两轮binary search；第一轮找到target所在的行，第二轮找具体的位置

        时间：O(logM + logN)
        空间：O(1)
        """
        rows = len(matrix)
        cols = len(matrix[0])

        # 先找到所在行
        top, bot = 0, rows - 1
        while top <= bot:
            cur_row = (top + bot) // 2
            if matrix[cur_row][-1] < target:
                top = cur_row + 1
            elif matrix[cur_row][0] > target:
                bot = cur_row - 1
            else:
                break

        if top > bot:
            return False

        # 在所在行里面找target
        l, r = 0, cols - 1
        while l <= r:
            mid = (l + r) // 2
            if matrix[cur_row][mid] < target:
                l = mid + 1
            elif matrix[cur_row][mid] > target:
                r = mid - 1
            else:
                return True

        return False
```

[240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)
```py
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        start from top-right to bottom-left
        
        Time: O(M+N)
        Space: O(1)
        """
        rows, cols = len(matrix), len(matrix[0])
        
        r, c = 0, cols - 1
        
        while r < rows and c >= 0:
            if matrix[r][c] == target:
                return True
            elif matrix[r][c] < target:
                r += 1
            else:
                c -= 1
        
        return False
```

[69. Sqrt(x)](https://leetcode.com/problems/sqrtx/)
```py
class Solution:
    def mySqrt(self, x: int) -> int:
        
        l, r = 0, x // 2
        while l <= r:
            mid = (l + r) // 2
            num = mid * mid
            if num > x:
                r = mid - 1
            elif num < x:
                l = mid + 1
            else:
                return mid
        
        return r
```

[540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/)








































[875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)
向上取整: ceil(a/b)

时间：O(N * logM): time for canFinish is O(N)
空间：O(1)

```py
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        """
        [False, False, ..., True, True, True, ...]
        Binary search to find the left most num of True
        """
        def canFinish(k):
            hour_need = 0
            for c in piles:
                hour_need += ceil(c / k)
            return hour_need <= h

        l, r = 1, max(piles)

        while l <= r:
            mid = (l + r) // 2

            if canFinish(mid):
                r = mid - 1
            else:
                l = mid + 1

        return l
```

[153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
也是找最左满足的数

时间：O(logN)
空间：O(1)

```py
class Solution:
    def findMin(self, nums: List[int]) -> int:
        """
        find the left most valid num
        """

        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if nums[mid] <= nums[-1]:
                r = mid - 1
            else:
                l = mid + 1

        return nums[l]
```

[4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)

时间：O(log(min(m, n)))看到log就要想到binary search
空间：

```py
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        A, B = nums1, nums2
        total = len(nums1) + len(nums2)
        half = total // 2

        if len(B) < len(A):
            A, B = B, A

        l, r = 0, len(A) - 1
        while True:
            i = (l + r) // 2 # A
            j = half - i - 2 # B, -2 因为index start at 0

            Aleft = A[i] if i >= 0 else float("-inf")
            Aright = A[i + 1] if (i + 1) < len(A) else float("inf")
            Bleft = B[j] if j >= 0 else float("-inf")
            Bright = B[j + 1] if (j + 1) < len(B) else float("inf")

            if Aleft <= Bright and Bleft <= Aright: # partition is correct
                # odd
                if total % 2:
                    return min(Aright, Bright)
                # even
                return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
            elif Aleft > Bright:
                r = i - 1 # reduce the size of A
            else:
                l = i + 1
```


[852. Peak Index in a Mountain Array](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

```py
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        """
        相当于binary search找最右满足的点
        
        Time: O(logN)
        Space: O(1)
        """
        l, r = 0, len(arr) - 1
        
        while l <= r:
            mid = (l + r) // 2
            if arr[mid] < arr[mid + 1]:
                l = mid + 1
            else:
                r = mid - 1
        
        return l
```