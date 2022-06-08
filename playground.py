def movingAverage(nums, window):
    res = []
    cur_sum = sum(nums[: window])
    res.append(cur_sum / window)

    for i in range(window, len(nums)):
        cur_sum += nums[i] - nums[i - window]
        res.append(cur_sum / window)

    return res
