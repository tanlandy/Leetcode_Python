# 排序基础知识
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

```py
def mergesort(seq):
    """归并排序"""
    if len(seq) <= 1:
        return seq
    mid = len(seq) / 2  # 将列表分成更小的两个列表
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

[56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)

```py
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda i : i[0])  # i代表interval，：后面表示按照排序i[0]
        # intervals.sort()
        res = [intervals[0]] # 一开始把第一个放进去

        for start, end in intervals[1:] :
            lastEnd = res[-1][1]
            if start <= lastEnd:
                res[-1][1] = max(lastEnd, end)
            else:
                res.append([start, end])
        
        return res 
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
        nums.sort(key = cmp_to_key(cmp_func), reverse = True)
        
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

[4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)

