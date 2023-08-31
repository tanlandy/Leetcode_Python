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

