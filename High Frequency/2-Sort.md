# 基础知识
In Java, Arrays.sort() for primitives is implemented using a variant of the Quick Sort algorithm, which has a space complexity of O(\log n)O(logn)
In C++, the sort() function provided by STL uses a hybrid of Quick Sort, Heap Sort and Insertion Sort, with a worst case space complexity of O(\log n)O(logn)
In Python, the sort() function is implemented using the Timsort algorithm, which has a worst-case space complexity of O(n)O(n)

A stable sorting algorithm means that when two elements have the same value, their relative order is maintained. 

## Insertion sort

从idx=0开始形成一个有序数列，每次把新的一个数插入进来，直到这个有序数列和最开始的一样大
It is a stable algorithm because later elements will not swap with earlier elements unless the later element is smaller,

时间：O(N^2)
空间：O(1)

```py
def insertionSort(nums):
    for idx, n in enumerate(nums):
        cur = idx
        while cur > 0 and nums[cur-1] > nums[cur]: # 把一个数插入进来，直到这个有序数列和最开始的一样大，一个while之后nums[:idx]是有序的
            nums[cur-1], nums[cur] = nums[cur], nums[cur - 1]
            cur -= 1
    return nums
```

## Selection sort

每次找到剩余序列里面最小的值，然后放到剩余序列的开头
This algorithm is not stable because an earlier element can jump after an element of the same value during a swap

时间：O(N^2)
空间：O(1)

```py
def selectionSort(nums):
    for i in range(len(nums)): # 一个i之后，nums[:i]是有序的
        min_idx = i
        for j in range(i, len(nums)): # 找到剩余序列里最小的值
            if nums[j] < nums[min_idx]:
                min_idx = j
        nums[min_idx], nums[i] = nums[i], nums[min_idx] # 最小的值放到剩余序列的开头

    return nums
```

## Bubble sort

每次走一遍都是前后两两比较，把大的放到后面，走完一遍之后能保证最后的几个是有序的，直到不再需要swap说明排好序了
It is a stable algorithm because a swap cannot cause an element to move past another one with the same value

时间：O(N^2)
空间：O(1)

```py
def bubbleSort(nums):
    for i in range(len(nums)):
        swapped = False

        for j in range(1, len(nums) - i):
            if nums[j - 1] > nums[j]:
                nums[j - 1], nums[j] = nums[j], nums[j - 1]
                swapped = True

        if not swapped: # 提前退出
            return nums

    return nums
```

## Merge sort

分而治之

时间：O(NlogN)
空间：O(N)

[912. Sort an Array](https://leetcode.com/problems/sort-an-array/)

```py
def mergesort(seq):
    """归并排序"""
    if len(seq) <= 1:
        return seq
    mid = len(seq) // 2  # 将列表分成更小的两个列表
    # 分别对左右两个列表进行处理，分别返回两个排序好的列表
    left = mergesort(seq[:mid])
    right = mergesort(seq[mid:])
    # 对排序好的两个列表合并，产生一个新的排序好的列表
    return merge(left, right)

def merge(left, right):
    """合并两个已排序好的列表，产生一个新的已排序好的列表"""
    result = []  # 新的已排序好的列表
    i = 0  # 下标
    j = 0
    # 对两个列表中的元素 两两对比。
    # 将最小的元素，放到result中，并对当前列表下标加1
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return results
```

## Quick sort

This algorithm is not stable, as each swap skips a lot of values.

时间：O(NlogN)
空间：O(N)

```py
def quickSort(nums, l, r):
    if l > r:
        return
    p = partition(nums, l, r)
    quickSort(nums, l, p - 1)
    quickSort(nums, p + 1, r)

def partition(nums, l, r):    
    pivot = nums[r]
    p = l

    for i in range(l, r):
        if nums[i] <= pivot:
            nums[i], nums[p] = nums[p], nums[i]
            p += 1

    nums[p], nums[r] = nums[r], nums[p]

    return p
```

# 例题

[148. Sort List](https://leetcode.com/problems/sort-list/)

```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Top down Mergesort: divide into subproblems, solve each, then merge together

        Time: O(NlogN)
        Space: O(NlogN)
        """
        def getMid(head):
            slow, fast = head, head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            mid = slow.next #  move one more
            slow.next = None # cut the rest
            return mid

        def merge(l, r):
            """Merge two lists"""
            if not l or not r:
                return l or r
            dummy = p = ListNode(0)
            while l and r:
                if l.val < r.val:
                    p.next = l
                    l = l.next
                else:
                    p.next = r
                    r = r.next
                p = p.next
            p.next = l or r
            return dummy.next

        if not head or not head.next:
            return head

        mid = getMid(head)
        left = self.sortList(head)
        right = self.sortList(mid)
        return merge(left, right)
```

[27. Remove Element](https://leetcode.com/problems/remove-element/)

```py
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        """
        use two pointers: slow for the correct one, fast ptr iterate throught the entire array
        Time: O(N)
        Space: O(1)
        """
        slow = fast = 0
        while fast < len(nums):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
```

[179. Largest Number](https://leetcode.com/problems/largest-number/)

```py
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        """
        use a comparator when sorting the nums

        Time: O(NlogN)
        Space: O(N)
        """

        def cmp_func(x, y):
            """
            Sorted by value of concatenated string increasingly.
            For case [3, 30], will return 330 instead of 303
            """
            if x + y > y + x:
                return 1
            elif x == y:
                return 0
            else:
                return -1

        # Build nums contains all numbers in the String format.
        nums = [str(num) for num in nums]

        # Sort nums by cmp_func decreasingly.
        nums.sort(key=cmp_to_key(cmp_func), reverse=True)

        res = "0" if nums[0] == "0" else "".join(nums)
        return res
```

[75. Sort Colors](https://leetcode.com/problems/sort-colors/)

```py
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Count sort: find out the number of 0, 1, 2, then build the res

        Time: O(N)
        Space: O(1)
        """
        c0 = c1 = c2 = 0
        for n in nums:
            if n == 0:
                c0 += 1
            elif n == 1:
                c1 += 1
            else:
                c2 += 1

        nums[:c0] = [0] * c0
        nums[c0: c0 + c1] = [1] * c1
        nums[c0 + c1:] = [2] * c2
        return nums
```

```py
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Three pointers: One pass
        p0, p2 record the position of "0" and "2" respectively
        for a new number n, if n == 0: swap it with p0, move both pointers forward: make sure p0 always "0"
        if n == 2: swap it with p2, move p2 left

        Time: O(N)
        Space: O(1)
        """
        p0 = cur = 0
        p2 = len(nums) - 1

        while cur <= p2:
            if nums[cur] == 0:
                nums[p0], nums[cur] = nums[cur], nums[p0]
                p0 += 1
                cur += 1
            elif nums[cur] == 2:
                nums[p2], nums[cur] = nums[cur], nums[p2]
                p2 -= 1
            else:
                cur += 1

        return nums
```

[215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        heapify这个array到minHeap：O(N)，然后pop()共n+1-k次=>时间O(N+(n+1-k)logN)

        用minHeap，size总是k，这样堆顶就是kth largest，然后一个一个pop()共n+1-k次，每次pop()时间是logN
        时间 O(N+Nlogk) 每次pop数字需要O(logk)，一共n次
        空间 O(K)
        """
        minHeap = []
        for n in nums:
            minHeap.append(n)

        heapq.heapify(minHeap) # time: O(n)

        # 第2大，一共6个数字，就是第5小
        k = len(nums) + 1 - k

        while k > 1:
            heapq.heappop(minHeap)
            k -= 1

        return minHeap[0]    
```

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        用maxHeap；注意这个时候用heapq._heapify_max(maxHeap)和heapq._heappop_max(maxHeap)来进行对应的操作
        """

        maxHeap = []
        for n in nums:
            maxHeap.append(n)

        heapq._heapify_max(maxHeap) # time: O(n)

        while k > 1:
            heapq._heappop_max(maxHeap)
            k -= 1

        return maxHeap[0]
```

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        partition: cut to two halves，左边的数都比右边的小，pivot就选最右的数，这个数字就是左右两边数的分界: p从最左index开始一直往右走，如果这个数比pivot小，那就放进来，然后p+=1，最后把p和pivot互换，效果就是pivot左边的数都比pivot小

        Quickselect
        时间 O(N)；如果每次的pivot都刚好是最大值，那每次都需要走一遍，所以那就是O(N^2)
        空间 O(1)
        """
        k = len(nums) - k # 把k变成sorted情况下的第k位

        # return p, p is pth smallest in nums     
        def partition(l, r):
            pivot, p = nums[r], l

            # nums before < nums[p] < nums after, based on pivot
            for i in range(l, r):
                if nums[i] <= pivot: # 如果当前这个数<=pivot，就放到左边
                    nums[p], nums[i] = nums[i], nums[p] # python不用一个swap()
                    p += 1

            # nums before < nums[r] < nums after
            nums[p], nums[r] = nums[r], nums[p]
            return p

        def select(l, r): # l, r告诉跑quickSelect的范围
            if l > r:
                return
            p = partition(l, r)

            if k < p: 
                return select(l, p - 1)
            elif k > p:
                return select(p + 1, r)
            else:
                return nums[p]

        return select(0, len(nums) - 1)
```


[1996. The Number of Weak Characters in the Game](https://leetcode.com/problems/the-number-of-weak-characters-in-the-game/)
```py
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        """
        按第一位降序排列，第一位相同就升序排列第二位
        从左往右比较第二位，同时记录见过的最大值，如果比见过的最大值小，那就双小了
        """
        properties.sort(key=lambda x: (-x[0],x[1]))

        res = 0
        curr_max = 0
        
        for _, d in properties:
            if d < curr_max:
                res += 1
            else:
                curr_max = d
        return res
```

## Intervals
### 模版

Interval的关键是找到两个区间的重合部分：overlap of two intervals

重叠的部分：[max(x1, y1), min(x2, y2)]
因此，终点是确保max(x1, y1) <= min(x2, y2)

常见的技巧是首先sort by start time

两个相邻区间的三种相对位置：
1. 完全重叠
2. 部分重叠
3. 不重叠

### 例题
[56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)

```py
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda i : i[0])  # i代表interval，：后面表示按照排序i[0]
        # intervals.sort()
        res = [intervals[0]] # 一开始把第一个放进去，这样好直接进行第一次比较

        for start, end in intervals[1:]:
            lastEnd = res[-1][1]
            if start <= lastEnd:
                res[-1][1] = max(lastEnd, end)
            else:
                res.append([start, end])

        return res 
```


[57. Insert Interval](https://leetcode.com/problems/insert-interval/)
```py
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """
        普通方法是把newInterval加到最后，然后用LC56的方式: sort之后一一比较。时间O(NlogN)
        本方法是Greedy，找到要插入区间的起始点，然后找到终止点。插入区间的前后保持原样。时间O(N)
        """

        res = []
        i = 0
        n = len(intervals)

        # find the beginning of newInterval
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i]) # before remains the same
            i += 1

        # find the end of newInterval
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i += 1

        res.append(newInterval) # add the interval

        while i < n: 
            res.append(intervals[i]) # after remains the same
            i += 1

        return res
```

[1288. Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/)
```py
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        """
        sort the intervals by start, when start are the same, sort base on end in decreasing order
        """
        intervals.sort(key = lambda x: (x[0], -x[1]))
        count = 0
        pre_end = 0
        
        # count the remaining intervals
        for start, end in intervals:
            if end > pre_end: # condition for a valid remaining interval
                count += 1
                pre_end = end
        
        return count
```

[986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)
```py
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        """
        可能重叠的地方就是[max(x1, y1), min(x2, y2)]
        """
        res = []
        i = j = 0
        
        while i < len(firstList) and j < len(secondList):
            start_1, end_1 = firstList[i][0], firstList[i][1]
            start_2, end_2 = secondList[j][0], secondList[j][1]
 
            start = max(start_1, start_2)
            end = min(end_1, end_2)
            
            if start <= end:
                res.append([start, end])
            
            if end_1 < end_2:
                i += 1
            else:
                j += 1
        
        return res
```

[252. Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)

```py
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        # intervals.sort()
        intervals.sort(key = lambda i : i[0])
        
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                return False
        
        return True
```


[253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) 高频好题
```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        """
        2个[]，分别存所有的start和end；双指针比较，如果start<end就count+=1同时start往后走；如果start>=end，就移动end的指针同时count -= 1；双指针走到头的情况就是start到头了；[]存start的方法：sorted([i[0] for i in intervals])

        时间：O(NlogN)
        空间：O(N)
        """
        start = sorted([i[0] for i in intervals])
        end = sorted([i[1] for i in intervals])

        res, count = 0, 0

        s, e = 0, 0

        while s < len(intervals):
            if start[s] < end[e]:
                count += 1
                s += 1
                res = max(res, count)
            else:
                e += 1
                count -= 1
        
        return res
```

```py
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        """
        判断是否需要一件新的room：check the min(end) with cur_start
        effectively check the minimum number of all rooms allocated: min_heap with the key is the end time of the current room
        for a new interval: if add then add, if not then update  
        the return value is the final size of the min_heap
        """
        intervals.sort()
        min_heap = []
        
        heapq.heappush(min_heap, intervals[0][1])
        
        for interval in intervals[1:]:
            if min_heap[0] <= interval[0]: # min_end <= cur_start: if no new room needed, free up the space(no overlap)
                heapq.heappop(min_heap)
            # push to room all the time
            heapq.heappush(min_heap, interval[1])
        
        return len(min_heap)
```

[1229. Meeting Scheduler](https://leetcode.com/problems/meeting-scheduler/) 高频好题

```py
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        """
        LC986的变形 加了一个判断条件
        sort input arrays and apply two pointers
        always move the pointer that ends earlier

        Time: O(MlogM + NlogN)
        Space: O(M + N) as sort in python takes O(N) for the worst case
        """
        
        slots1.sort()
        slots2.sort()
        
        i = j = 0
        
        while i < len(slots1) and j < len(slots2):
            s1, e1 = slots1[i][0], slots1[i][1]
            s2, e2 = slots2[j][0], slots2[j][1]
            
            start = max(s1, s2)
            end = min(e1, e2)
            
            if start + duration <= end: # <>的比较一定要注意，总是容易弄反
                return [start, start + duration]
            
            if e1 < e2:
                i += 1
            else:
                j += 1
        
        return []
```

```py
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        # build up a heap containing time slots last longer than duration
        timeslots = list(filter(lambda x: x[1] - x[0] >= duration, slots1 + slots2))
        heapq.heapify(timeslots)

        # 同一个的时间不可能重叠：有重叠的话就是2个人的重叠部分，就是可能的结果区间

        while len(timeslots) > 1:
            start1, end1 = heapq.heappop(timeslots) # 其中一个人
            start2, end2 = timeslots[0] # 另一个人
            if end1 >= start2 + duration: # 因为在构建timeslots时候，已经确保了end2 >= start2 + duration
                return [start2, start2 + duration]
        return []
```

[280. Wiggle Sort](https://leetcode.com/problems/wiggle-sort/)
```py
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Approach1: sort and then swap one pair by one pair, with a step of two
        Approach2: one pass: compare the odd index, swap two times each step if needed
        
        """
        if not nums:
            return
        n = len(nums)
        for i in range(1, n, 2):
            if nums[i] < nums[i-1]:
                nums[i], nums[i-1] = nums[i-1], nums[i]
            
            if i + 1 < n and nums[i] < nums[i+1]:
                nums[i], nums[i+1] = nums[i+1], nums[i]
```

[324. Wiggle Sort II](https://leetcode.com/problems/wiggle-sort-ii/)

```py
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        要求严格nums[0] < nums[1] > nums[2]
        """
        arr = sorted(nums)
        for i in range(1, len(nums), 2): nums[i] = arr.pop() 
        for i in range(0, len(nums), 2): nums[i] = arr.pop() 
```

## Merge Sort

https://leetcode.com/problems/reverse-pairs/discuss/97268/General-principles-behind-problems-similar-to-%22Reverse-Pairs%22 

[315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)

```py
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        n = len(nums)
        arr = [[v, i] for i, v in enumerate(nums)]  # record value and index
        result = [0] * n

        def merge_sort(arr, left, right):
            # merge sort [left, right) from small to large, in place
            if right - left <= 1:
                return
            mid = (left + right) // 2
            merge_sort(arr, left, mid)
            merge_sort(arr, mid, right)
            merge(arr, left, right, mid)

        def merge(arr, left, right, mid):
            # merge [left, mid) and [mid, right)
            i = left  # current index for the left array
            j = mid  # current index for the right array
            # use temp to temporarily store sorted array
            temp = []
            while i < mid and j < right:
                if arr[i][0] <= arr[j][0]: # 比较数字大小
                    # j - mid numbers jump to the left side of arr[i] ==> 精华
                    result[arr[i][1]] += j - mid # 直接更新坐标结果
                    temp.append(arr[i])
                    i += 1
                else:
                    temp.append(arr[j])
                    j += 1
            # when one of the subarrays is empty
            while i < mid:
                # j - mid numbers jump to the left side of arr[i] ==> 精华
                result[arr[i][1]] += j - mid
                temp.append(arr[i])
                i += 1
            while j < right:
                temp.append(arr[j])
                j += 1
            # restore from temp
            for i in range(left, right):
                arr[i] = temp[i - left]

        merge_sort(arr, 0, n)

        return result
```

[327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/)

```py
class Solution:

    # prefix-sum + merge-sort | time complexity: O(nlogn)
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        cumsum = [0]
        for n in nums:
            cumsum.append(cumsum[-1]+n)
            
		# inclusive
        def mergesort(l,r):
            if l == r:
                return 0
            mid = (l+r)//2
            cnt = mergesort(l,mid) + mergesort(mid+1,r)
			
            i = j = mid+1
            # O(n)
            for left in cumsum[l:mid+1]:
                while i <= r and cumsum[i] - left < lower:
                    i+=1
                while j <= r and cumsum[j] - left <= upper:
                    j+=1
                cnt += j-i
                
            cumsum[l:r+1] = sorted(cumsum[l:r+1])
            return cnt
			
        return mergesort(0,len(cumsum)-1)
```

[493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/)
In each round, we divide our array into two parts and sort them. So after "int cnt = mergeSort(nums, s, mid) + mergeSort(nums, mid+1, e); ", the left part and the right part are sorted and now our only job is to count how many pairs of number (leftPart[i], rightPart[j]) satisfies leftPart[i] <= 2*rightPart[j].
For example,
left: 4 6 8 right: 1 2 3
so we use two pointers to travel left and right parts. For each leftPart[i], if j<=e && nums[i]/2.0 > nums[j], we just continue to move j to the end, to increase rightPart[j], until it is valid. Like in our example, left's 4 can match 1 and 2; left's 6 can match 1, 2, 3, and left's 8 can match 1, 2, 3. So in this particular round, there are 8 pairs found, so we increases our total by 8.
```py
class Solution:
    def reversePairs(self, nums):
        return self.mergeSort(nums, 0, len(nums)-1)

    def mergeSort(self, nums, start, end):
        if start >= end:
            return 0
        mid = (start+end)//2 + 1
        count = self.mergeSort(nums, start, mid-1) + self.mergeSort(nums, mid, end)
        j = mid 
        for i in range(start, mid):
            while j<=end and nums[j]*2 < nums[i]:
                j += 1
            count += (j-mid)
        nums[start:end+1] = sorted(nums[start:end+1])
        return count
```
