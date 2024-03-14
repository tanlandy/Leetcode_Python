# 剑指offer2题目

[剑指offer2精选75题](https://leetcode.cn/problem-list/xb9nqhhg/)

# 1 - 数组

[03. 数组中重复的数字](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

```python

class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        """直接在原数组上操作，把数字i放到nums[i]上"""
        for i in range(len(nums)):
            while i != nums[i]:
                if nums[i] == nums[nums[i]]:
                    return nums[i]
                tmp = nums[i]
                nums[i], nums[tmp] = nums[tmp], nums[i]
        

```

[04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

```python

class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        """右上往左下走"""
        r, c = len(matrix) - 1, 0

        while r >= 0 and c < len(matrix[0]):
            if matrix[r][c] > target:
                r -= 1
            elif matrix[r][c] < target:
                c += 1
            else:
                return True
        
        return False

```

[11. 旋转数组的最小数字](https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

```python

class Solution:
    def minArray(self, numbers: List[int]) -> int:
        l, r = 0, len(numbers) - 1

        while l <= r:
            mid = (l + r) // 2
            # mid不与l比较，是因为无法准确判定mid在左/右排序数组，因为l动来动去不确定会在左/右排序数组，但r总是在右排序数组
            if numbers[mid] < numbers[r]:  # mid一定在右排序中，旋转点x一定在[l, m]闭区间内
                r = mid
            elif numbers[mid] > numbers[r]:  # mid一定在左排序中，旋转点x一定在[m+1, r]闭区间内
                l = mid + 1
            else:  # 无法判定mid在左/右排序数组，自然无法通过二分法缩小区间
                r -= 1
        
        return numbers[l]

```
