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