def movingAverage(nums, window_size):
    if window_size <= 0:
        print("Error: the window size should be larger than 0!")
        return []
    
    left = right = 0
    res = []
    cur_sum = 0

    while right < len(nums):
        cur_sum += nums[right]
        right += 1
        if right - left == window_size:
            res.append(cur_sum / window_size)
            cur_sum -= nums[left]
            left += 1
    
    return res

nums = [1,2,3,4,5]
print(movingAverage(nums, 3))
