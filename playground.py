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

nums = [1,4,2 ,1231242,124,212,1,435,1,23,3,4]

quickSort(nums, 0, len(nums) - 1)
print(nums)