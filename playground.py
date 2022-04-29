def bubbleSort(nums):
    for i in range(len(nums)):
        swapped = False

        for j in range(1, len(nums) - i):
            if nums[j - 1] > nums[j]:
                nums[j - 1], nums[j] = nums[j], nums[j - 1]
                swapped = True
            
        if not swapped:
            return nums
    
    return nums

nums=[1, 4, 2, 4, 5, 2, 5, 22]

bubbleSort(nums)

print(nums)
