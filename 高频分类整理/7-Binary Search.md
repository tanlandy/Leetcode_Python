[74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
做两轮binary search；第一轮找到target所在的行，第二轮找具体的位置

时间：O(logM + logN)
空间：O(1)

```py
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
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

[33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
先判断是否找到，找不到的话，比较nums[l]和nums[mid]，从而找到接下来的二分查找区间：内部嵌套2个binary search

时间：O(logN)
空间：O(1)

```py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if nums[mid] == target:
                return mid

            if nums[l] <= nums[mid]:
                if nums[l] <= target and target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1            
            else:
                if nums[mid] < target and target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1

        return -1
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