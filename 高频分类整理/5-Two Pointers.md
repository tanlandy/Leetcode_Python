# 背向双指针



# 相向双指针
[125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
用相向two pointer，当不是char时候就比较; s[i].isalnum() 看是否是string或者num; s[i].lower() 返回一个小写

时间：O(N)
空间：O(1)

```python
def isPalindrome(s: str) -> bool:
    i, j = 0, len(s) - 1
    while i < j:
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1
        if s[i].lower() != s[j].lower():
            return False
        i += 1
        j -= 1
    return True

```


[167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
相向双指针，分别从开头和最后往中间走。如果cur_sum太小就移动左边，如果太大就移动右边。

时间：O(N), N is len(numbers)
空间：O(1)
```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1
        while l < r:
            cur_sum = numbers[l] + numbers[r]
            if cur_sum > target:
                r -= 1
            elif cur_sum < target:
                l += 1
            else:
                return [l + 1, r + 1] 
```        

[15. 3Sum](https://leetcode.com/problems/3sum/)
先排序，然后针对每个数，对另两个数来TwoSum2，避免重复使用的方法是if i > 0 and a == nums[i-1]: continue;针对这个a找到一个组合之后，只用移动左指针，同时如果左指针相同就一直移动

时间：ON^2), 排序用O(NlogN)，比较和用O(N)*O(N)
空间：O(1) 取决于排序所用的空间
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        
        for i, a in enumerate(nums):
            # 同一个数不重复使用
            if i > 0 and a == nums[i - 1]:
                continue
            
            # LC167一样的套路
            l, r = i + 1, len(nums) - 1
            while l < r:
                cur_sum = a + nums[l] + nums[r]
                if cur_sum > 0:
                    r -= 1
                elif cur_sum < 0:
                    l += 1
                else:
                    res.append([a, nums[l], nums[r]])
                    # update pointer
                    l += 1
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1
        return res
```


[11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
相向双指针，每次移动nums[l], nums[r]中更小的那一个，计算面积

时间：O(N)
空间：O(1)
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        area = 0
        
        while l < r:
            cur_area = (r - l) * min(height[l], height[r])
            area = max(area, cur_area)
            
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        
        return area
```

[42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

对于位置i能存储的水: min(maxL, maxR) - h[i]；相向双指针，加上两个变量maxL, maxR来时刻保存左右两边的最大值；每次移动maxL, maxR之间较小那个数的指针，然后新位置i能存储的水：被移动指针之前的值-h[i]：不用考虑另外一个值，因为那个值肯定比较大；移动指针之后计算这个指针所在位置能存储的水

时间：O(N)
空间：O(1) -> two pointers
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        res = 0
        if not height: # input is empty
            return res
        
        l, r = 0, len(height) - 1
        max_l, max_r = height[l], height[r]
        
        while l < r:
            if max_l < max_r:
                l += 1
                max_l = max(max_l, height[l])
                res += max_l - height[l]
            else:
                r -= 1
                max_r = max(max_r, height[r])
                res += max_r - height[r]
        
        return res
```

# 同向双指针-Sliding Window

[121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)


```python

```