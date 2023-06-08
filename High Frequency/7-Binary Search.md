# 基础知识

## 找一个数
```py
def find_target(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r: # 最后一次搜索的是left == right的情况，是[left, right]: 左闭右闭
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            l = mid + 1 # mid已经搜索过了，所以+1
        else:
            r = mid - 1
        
    return -1
```

## 找最左侧边界
```py
def find_leftmost(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target: # 继续往左边找，所以右边的空间就缩小不要了
            r = mid - 1
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    
    if l >= len(nums) or nums[l] != target: # 一个都没有相等的, l一直往右走
        return -1
    return l
```

## 找最右侧边界
```py
def find_leftmost(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target: # 继续往右边找，所以把左边的空间缩小不要了
            l = mid + 1
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    
    if r < 0 or nums[r] != target: # 一个都没有相等的, r一直往左走
        return -1
    return r
```

## Bisect
```py
import bisect

A = [-14, -10, 2, 108, 108, 243, 285, 285, 285, 401]

# first 108 is at index 3
print(bisect.bisect_left(A, 108))

bisect_right() # 相当于 bisect()

bisect.insort_left(A, 108) # insert at the first index
print(A)
```

# 例题

## 显式二分法
说明了是有序数组/序列，那么大概率是可以使用二分法的
[1351. Count Negative Numbers in a Sorted Matrix](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/description/)

```py
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        """
        找最左侧边界
        时间：O(MlogN)
        空间：O(1)
        """
        neg_count = 0
        for row in range(len(grid)):
            l, r = 0, len(grid[0]) - 1
            while l <= r:  # 右边都满足，找最左侧边界，第一个小于0的数
                pivot = (l + r) // 2
                if grid[row][pivot] < 0:
                    r = pivot - 1
                else:
                    l = pivot + 1
            neg_count += (len(grid[0]) - l)
    
        return neg_count
```


这道题除了明显可以用二分查找以外，还有时间复杂度更低的方法
```py
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        """
        从右上往左下
        时间：O(M+N)
        空间：O(1)
        """
        neg_count = 0
        pivot = len(grid[0]) - 1  # 第一行最右边
        for row in range(len(grid)):
            while pivot >= 0 and grid[row][pivot] < 0:  # 一旦调用了grid[pivot]，那么就要想到越界的情况
                pivot -= 1
            neg_count += (len(grid[0]) - pivot - 1)  # 对于是否+-1的情况，举一个例子试试看就知道了
        return neg_count     
```





## 隐式二分法
Find the Closest Number
Input :arr[] = {2, 5, 6, 7, 8, 8, 9};
Target number = 4
Output : 5
```py
A1 = [1, 2, 4, 5, 6, 6, 8, 9]
A2 = [2, 5, 6, 7, 8, 8, 9]


def find_closest_num(A, target):
    min_diff = min_diff_left = min_diff_right = float("inf")
    low = 0
    high = len(A) - 1
    closest_num = None

    # Edge cases for empty list of list
    # with only one element:
    if len(A) == 0:
        return None
    if len(A) == 1:
        return A[0]

    while low <= high:
        mid = (low + high)//2

        # Calc the min_diff_left and min_diff_right
        if mid + 1 < len(A):
            min_diff_right = abs(A[mid + 1] - target)
        if mid > 0:
            min_diff_left = abs(A[mid - 1] - target)

        # Check if the absolute value between left and right elements 
        # are smaller than any seen prior.
        if min_diff_left < min_diff:
            min_diff = min_diff_left
            closest_num = A[mid - 1]

        if min_diff_right < min_diff:
            min_diff = min_diff_right
            closest_num = A[mid + 1]

        # Move the mid-point appropriately as is done via binary search.
        if A[mid] < target:
            low = mid + 1
        elif A[mid] > target:
            high = mid - 1
        else:
            return A[mid]
    return closest_num


print(find_closest_num(A1, 11))
print(find_closest_num(A2, 8))
```



Find Bitonic Peak
input: [1, 2, 3, 4, 5, 4, 3, 2, 1]
output: 5

```py
def find_highest_number(A):
    low = 0
    high = len(A) - 1

    # Require at least 3 elements for a bitonic sequence.
    if len(A) < 3:
        return None

    while low <= high:
        mid = (low + high)//2

        mid_left = A[mid - 1] if mid - 1 >=0 else float("-inf")
        mid_right = A[mid + 1] if mid + 1 < len(A) else float("inf")

        if mid_left < A[mid] and mid_right > A[mid]:
            low = mid + 1
        elif mid_left > A[mid] and mid_right < A[mid]:
            high = mid - 1
        elif mid_left < A[mid] and mid_right < A[mid]:
            return A[mid]
    return None
```

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

    def binSearch(self, nums, target, find_left):
        l, r = 0, len(nums) - 1
        res = -1
        while l <= r:
            mid = l + (r - l) // 2
            if target > nums[mid]:
                l = mid + 1
            elif target < nums[mid]:
                r = mid - 1
            else:
                res = mid
                if find_left:
                    r = mid - 1
                else:
                    l = mid + 1
        return res
```

[1182. Shortest Distance to Target Color](https://leetcode.com/problems/shortest-distance-to-target-color/)
```py
class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        """
        color_idx:{color: [idx]}
        then use binary search to find the nearest idx

        Time: O(QlogN + N)
        Space: O(N)
        """
        color_idx = collections.defaultdict(list)
        
        for idx, c in enumerate(colors):
            color_idx[c].append(idx)
        
        res = []
        for i, (target, color) in enumerate(queries):
            # for invalid input
            if color not in color_idx:
                res.append(-1)
                continue
            
            # for valid input
            idx_list = color_idx[color]
            
            # binary search for the nearest element
            nearest = self.findNearest(target, idx_list)
            res.append(nearest)
        return res
    
    def findNearest(self, target, nums):
        if len(nums) == 1:
            return abs(target - nums[0])
        
        min_diff = left_diff = right_diff = min(abs(nums[0] - target), abs(nums[-1] - target))
        l, r = 0, len(nums) - 1
        
        while l <= r:
            mid = (l + r) // 2
            if mid + 1 < len(nums):
                right_diff = abs(nums[mid + 1] - target)
            if mid > 0:
                left_diff = abs(nums[mid - 1] - target)
            
            min_diff = min(min_diff, right_diff, left_diff)
            
            if nums[mid] < target:
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
            else:
                return 0
        
        return min_diff
```

pre-compute方法
```py
class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        """
        pre-compute and store the shortest distance btw each idx and color
        ==> find the nearest color on idx's left, and then on idx's right
        """
        n = len(colors)
        rightmost = [0, 0, 0]
        leftmost = [n-1, n-1, n-1]
        dist = [[-1] * n for _ in range(3)]
        
        # look forward
        for i in range(n):
            color = colors[i] - 1
            for j in range(rightmost[color], i + 1):
                dist[color][j] = i - j
            rightmost[color] = i + 1
        
        for i in range(n-1, -1, -1):
            color = colors[i] - 1
            for j in range(leftmost[color], i-1, -1):
                if dist[color][j] == -1 or dist[color][j] > j - i:
                    dist[color][j] = j - i
            leftmost[color] = i - 1
        
        return [dist[color - 1][idx] for idx, color in queries]
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
                if nums[l] <= target and target < nums[mid]: # target is in left part's left
                    r = mid - 1
                else:
                    l = mid + 1            
            else: # 说明mid右边是Sorted
                if nums[mid] < target and target <= nums[r]: # target is in right part's right
                    l = mid + 1
                else:
                    r = mid - 1

        return -1
```

[81. Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)

```py
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        """
        因为会有重复，所以在nums[mid] == nums[r]的时候就不能判断target在哪边，所以这时候直接r-=1。然后通过nums[mid]和nums[r]的大小比较，来确认哪边是sorted

        时间：O(N)
        空间：O(1)
        """
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if nums[mid] == target:
                return True
            elif nums[mid] == nums[r]: # failed to determine which side is sorted, as well as which side the target is located
                r -= 1
            elif nums[mid] > nums[r]: # 说明mid左边是sorted
                if nums[l] <= target and target < nums[mid]: # target is in left part's left
                    r = mid - 1
                else:
                    l = mid + 1            
            else: # 说明mid右边是Sorted
                if nums[mid] < target and target <= nums[r]: # target is in right part's right
                    l = mid + 1
                else:
                    r = mid - 1

        return False
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

[162. Find Peak Element](https://leetcode.com/problems/find-peak-element/) 好题

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

[1901. Find a Peak Element II](https://leetcode.com/problems/find-a-peak-element-ii/)

```py
class Solution:
    def findPeakGrid(self, mat: List[List[int]]) -> List[int]:
        """
        Say we took the largest element in a column, if any of adjacent
        element is greater than the current Largest, we can just greedily go through
        the largest elements till we find a Peak Element
        
        In my Solution I am splitting column wise because in the hint it is given that width is more than the height.
        简而言之：每一列找最大值，然后看是否是该行的peak，不是的话再找另一列的最大值

        Time: O(MlogN) 最多找M次，每次是LogN
        Space: O(1)
        """
        start, end = 0, len(mat[0]) - 1 # leftMost and rightMost col
        while start <= end:
            cmid = (start + end) // 2 # column mid
            
            # Finding the largest element in the middle Column
            ridx, curLargest = 0, float('-inf')
            for i in range(len(mat)):
                if mat[i][cmid] > curLargest:
                    curLargest = mat[i][cmid]
                    ridx= i 
            
            # Checking the adjacent element
            leftisBig = cmid > start and mat[ridx][cmid - 1] > mat[ridx][cmid]
            rightisBig = cmid < end and mat[ridx][cmid + 1] > mat[ridx][cmid]
            
            if not leftisBig and not rightisBig:
                return [ridx, cmid]
            
            # binary search 
            if leftisBig:
                end = cmid - 1
            else:
                start = cmid + 1
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
        rows, cols = len(matrix), len(matrix[0])

        # 先找到所在行
        top, bot = 0, rows - 1
        while top <= bot:
            cur_row = (top + bot) // 2
            if matrix[cur_row][-1] < target:
                top = cur_row + 1
            elif matrix[cur_row][0] > target: # 最小的数都更大
                bot = cur_row - 1
            else:
                break

        if top > bot: # 可有可无：提前返回
            return False

        # 再在所在行里面找target
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

The pairs which are on the left of the single element, will have the first element in an even position and the second element at an odd position. All the pairs which are on the right side of the single element will have the first position at an odd position and the second element at an even position.

```py
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        """
        the nums must always have an odd number of elements
        """
        if len(nums) == 1:
            return nums[0]
        
        
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            before_is_even = (r - mid) % 2 == 0
            if nums[mid] == nums[mid + 1]:
                if before_is_even: # find after: find the odd part
                    l = mid + 1
                else:
                    r = mid - 1
            elif nums[mid] == nums[mid - 1]:
                if before_is_even: # [:mid-1] is odd, find the odd part
                    r = mid - 2
                else:
                    l = mid + 1
            else:
                return nums[mid]
        
        return nums[l]
```


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

[154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

```py
class Solution():
    def findMin(self, nums):
        """
        Time: O(N) in the worst case where the array contains identical elements, when the algorigthm would iterate each element
        Space: O(1)
        
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < nums[r]: # the mid resides in the same half as the upper bound element -> the target reside to its left-hand side
                r = mid
            elif nums[mid] > nums[r]: # the target reside to its right-hand side
                l = mid + 1
            else: # not sure which side of the mid that the target would reside
                r -= 1
        
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

[744. Find Smallest Letter Greater Than Target](https://leetcode.com/problems/find-smallest-letter-greater-than-target/)

```py
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        """
        binary search: 右边都满足，找最左侧的边界
        
        Time: O(logN)
        Space: O(1)
        """
        # if the number is out of bound
        if target >= letters[-1] or target < letters[0]:
            return letters[0]
        
        l, r = 0, len(letters) - 1
        
        while l <= r:
            mid = (l + r) // 2
            
            if  target > letters[mid]:
                l = mid + 1
            elif target < letters[mid]:
                r = mid - 1
            else:
                l = mid + 1
                
        return letters[l]
```

[1062. Longest Repeating Substring](https://leetcode.com/problems/longest-repeating-substring/)
二分法和移动窗口的结合

```py
class Solution:
    def search(self, L: int, n: int, S: str) -> str:
        """
        Search a substring of given length
        that occurs at least 2 times.
        @return start position if the substring exits and -1 otherwise.
        """
        seen = set()
        for start in range(0, n - L + 1):
            tmp = S[start:start + L]
            if tmp in seen:
                return start
            seen.add(tmp)
        return -1
        
    def longestRepeatingSubstring(self, S: str) -> str:
        n = len(S)
        
        # binary search, L = repeating string length
        left, right = 1, n
        while left <= right:
            L = left + (right - left) // 2
            if self.search(L, n, S) != -1:
                left = L + 1
            else:
                right = L - 1
               
        return left - 1
```

[1060. Missing Element in Sorted Array](https://leetcode.com/problems/missing-element-in-sorted-array/) 好题

```python
class Solution:
    def missingElement(self, nums: List[int], k: int) -> int:
        """
        nums[i]之前的missing个数是nums[i] - nums[0] - i（如果不missing，idx=i的位置数字应该是nums[0] + i）; 那么要找第k个数，那么就是找位置i，满足nums[i] - nums[0] - i < k < nums[i+1] - nums[0] - (i+1)
        返回nums[i] + k - (nums[i] - nums[0] - i) = k + nums[0] + i

        时间：O(logN)
        空间：O(1)
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] - nums[0] - mid < k:
                l = mid + 1
            else:
                r = mid - 1
        return k + nums[0] + r

```


[1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

```python
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        """
        l=max(weights), r=sum(weights),最后返回的是最左侧边界l；isValid: r = mid - 1；计算isValid：用一个cur，每次cur+=w，先检查是否cur+w>cap，是的话就days_need+=1, cur = 0

        时间：O(logN)
        空间：O(1)
        """
        l, r = max(weights), sum(weights)
        
        def isvalid(cap): # 注意如何计算满足
            day_need = 1
            cur = 0
            for w in weights:
                if cur + w > cap: # 每次更新现有的，直到大于cap
                    day_need += 1
                    cur = 0
                cur += w
            
            return day_need <= days
        
        while l <= r:
            mid = l + (r - l) // 2
            if isvalid(mid):
                r = mid - 1
            else:
                l = mid + 1
        
        return l
        
```

[4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)

```py
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """
        Binary search大小更小的那组数A，把这组数partition到2部分
        看A的左边和对应B的左边是否刚好就是把AB合并之后左边的那部分：Aleft<=Bright, Bleft<=Aright
            - 如果是奇数个数，那么就是Aright和Bright中较小的数
            - 如果是偶数个数，那么就是(max(Aleft, Bleft) + min(Aright, Bright)) / 2
        否则就重新搜索A的大小：
            - Aleft>Bright的话，就要缩小A：r = mid - 1
            - Bleft>Aright的话，就要扩大A：l = mid + 1
        
        Time: O(min(A, B))
        Space: O(1)
        """
        A, B = nums1, nums2
        if len(A) > len(B):
            A, B = B, A
        total = len(A) + len(B)
        half = total // 2
        
        l, r = 0 , len(A) - 1
        while True:
            i = (l + r) // 2
            j = half - i - 2
            
            Aleft = A[i] if i >= 0 else float("-inf")
            Aright = A[i + 1] if (i + 1) < len(A) else float("inf")
            Bleft = B[j] if j >= 0 else float("-inf")
            Bright = B[j + 1] if (j + 1) < len(B) else float("inf")
            
            if Aleft <= Bright and Bleft <= Aright:
                if total % 2:
                    return min(Aright, Bright) # 左边刚好就满了，下一个数就是median
                else:
                    return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
            elif Aleft > Bright:
                r = i - 1
            else:
                l = i + 1
            
```

[719. Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/)

[410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)

[981. Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)
```py
class TimeMap:

    def __init__(self):
        """
        dic: {key: list of [timestamp, value]}
        """
        self.dic = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.dic[key].append([timestamp, value])

    def get(self, key: str, timestamp: int) -> str:
        arr = self.dic[key]
        l, r = 0, len(arr) - 1
        while l <= r:
            mid = (l + r) // 2
            if arr[mid][0] <= timestamp:
                l = mid + 1
            else:
                r = mid - 1
        
        return "" if r == -1 else arr[r][1]
```



[1891. Cutting Ribbons](https://leetcode.com/problems/cutting-ribbons/)

转化思路，题目要求最多切成n次，那n=1到max(ribbon)，这样满足条件的是n最大的那个时候，相当于每次都看是否满足条件，直到找到最后满足条件的值。可以用二分查找找最右侧边界；count >= k是满足的条件，count表示可以提供的数量；最后return right，因为跳出的时候left = right + 1了
时间：O(Nlog(max(Length))) 
空间：O(1)
```python
class Solution:
    def maxLength(self, ribbons: List[int], k: int) -> int:
        left = 1
        right = max(ribbons)
        
        while left <= right: # 左闭右闭
            mid = left + (right - left) // 2
            if self.isValid(ribbons, k, mid):
                left = mid + 1
            else:
                right = mid - 1
        
        return right # 最后要return right，因为while的终止条件是left += 1
    
    def isValid(self, ribbons, k, mid):
        count = 0
        for num in ribbons:
            count += num // mid
        return count >= k  # 满足的情况，count表示可以提供的数量
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

# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()
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

[441. Arranging Coins](https://leetcode.com/problems/arranging-coins/)

```py
class Solution:
    def arrangeCoins(self, n: int) -> int:
        """
        1+2+3+...+k = k*(k+1)//2
        => find the max k such that k*(k+1)//2 <= N
        """
        l, r = 0, n
        
        while l <= r:
            k = (l + r) // 2
            curr = k * (k + 1) // 2
            if curr == n:
                return k
            elif curr > n:
                r = k - 1
            else:
                l = k + 1
        
        return r
```
