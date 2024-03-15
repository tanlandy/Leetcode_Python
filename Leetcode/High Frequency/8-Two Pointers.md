# Two Pointers

# 基础知识，技巧与思路

## 同向双指针(Sliding Window)

该类问题的关键点：扩大的条件和结果；缩小的条件和结果；更新res的条件和结果
1、当移动right扩大窗口，即加入字符时，应该更新哪些数据？

2、什么条件下，窗口应该暂停扩大，开始移动left缩小窗口？

3、当移动left缩小窗口，即移出字符时，应该更新哪些数据？

4、我们要的结果应该在扩大窗口时还是缩小窗口时进行更新？

s = [2,2,1,3,2], d = 4, m = 2
find how many ways to make it sum to d with a size of m

```py
def birthday(s, d, m):
    l = r = 0
    cur_sum = 0
    res = 0
    while r < len(s):
        cur_sum += s[r]
        if r - l + 1 == m:  # 当window size==m
            if cur_sum == d:
                res += 1
            cur_sum -= s[l]
            l += 1
        r += 1
    return res
```

Given an array (list) nums consisted of only non-negative integers, find the largest sum among all subarrays of length k in nums.
nums = [1, 2, 3, 7, 4, 1], k = 3
output would be 14 as the largest length 3 subarray sum is given by [3, 7, 4]

```py
def subarray_sum_fixed(nums: List[int], k: int) -> int:
    """最标准的sliding window，注意移动r, l的时机"""
    l = r = 0
    res = 0
    cur_sum = 0
    while r < len(nums):
        cur_sum += nums[r]        
        if r - l + 1 == k:
            res = max(cur_sum, res)
            cur_sum -= nums[l]
            l += 1
        r += 1
        
    return res
```

Given a string original and a string check, find the starting index of all substrings of original that is an anagram of check.
Input: original = "cbaebabacd", check = "abc"
Output: [0, 6]

```py
def find_all_anagrams(original: str, check: str) -> List[int]:
    """标准的sliding window需要记录个数的情况，可以用list来微微简化代码，但是用dict更加直观"""
    res = []
    counter = collections.defaultdict(int)
    l = r = 0
    for ch in check:
        counter[ch] += 1
    seen = collections.defaultdict(int)
    while r < len(original):
        seen[original[r]] += 1
        if r - l + 1 == len(check):  # 每次在大小满足之后进行处理                       
            if seen == counter:
                res.append(l)                
            seen[original[l]] -= 1
            if seen[original[l]] == 0:  # 会多余一个e:0的情况，需要将其删掉
                del seen[original[l]]            
            l += 1
        r += 1            
    
    return res
```

# 高频题

## 知乎

11, 42, 240

## Krahets精选题

3, 15, 142（链表）, 151, 160, 167, 239, 392, 876

# 以题型分类

## 背向双指针

[409. Longest Palindrome](https://leetcode.cn/problems/longest-palindrome/)

```py
class Solution:
    def longestPalindrome(self, s: str) -> int:
        """
        the char appears odd times must be removed 1 time
        
        Time: O(N)
        Space: O(1), as the size of s is fixed
        """
        counts = collections.Counter(s)
        odd = 0
        
        for count in counts.values():
            if count % 2:
                odd += 1
        
        return len(s) if odd <= 1 else len(s) - odd + 1
```

[125. Valid Palindrome](https://leetcode.cn/problems/valid-palindrome/)

```python
def isPalindrome(s: str) -> bool:
    """
    用相向two pointer，当不是char时候就比较; s[i].isalnum() 看是否是string或者num; s[i].lower() 返回一个小写
    isalnum(): check if alphanumeric: (a-z) and (0-9)

    时间：O(N)
    空间：O(1)
    """
    l, r = 0, len(s) - 1
    
    while l < r:
        while l < r and not s[l].isalnum(): # l < r的条件不能少，否则s=".,"会out of index
            l += 1
        while l < r and not s[r].isalnum():
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l += 1
        r -= 1
    
    return True
```

[5. Longest Palindromic Substring](https://leetcode.cn/problems/longest-palindromic-substring/)

```py
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        从中间往两边扩散，同时要看以自己为中心和以自己为左半边出发的情况。最后要注意成功时候的index
        时间：O(N^2)
        空间：O(1)
        """
        res = ""
        
        def findPalindrome(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            
            return s[l + 1: r]
    
        for i in range(len(s)):
            s1 = findPalindrome(i, i)
            s2 = findPalindrome(i, i + 1)            
            
            if len(s1) > len(res):
                res = s1
            if len(s2) > len(res):
                res = s2
        
        return res
```

[647. Palindromic Substrings](https://leetcode.cn/problems/palindromic-substrings/)

```py
class Solution:
    def countSubstrings(self, s: str) -> int:
        """
        Expand Around Possible Centers: use a helper function to find the palindromic substrings
        check one by one
        
        Time: O(N^2)
        Space: O(1)
        """
        
        def countPalind(l, r):
            one_res = 0
            while l >= 0 and r < len(s) and s[l] == s[r]:
                one_res += 1
                l -= 1
                r += 1
            return one_res
        
        res = 0
        for i in range(len(s)):
            res += countPalind(i, i)
            res += countPalind(i, i + 1)      
        
        return res
```

## 相向双指针

### 缩减空间思想

[167. Two Sum II - Input Array Is Sorted](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)

```py
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1
        
        while l < r:
            cur_sum = numbers[l] + numbers[r]
            if cur_sum == target:
                l += 1
                r += 1
                return [l, r]
            elif cur_sum < target:
                l += 1
            else:
                r -= 1
```

[240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/?envType=study-plan-v2&envId=selected-coding-interview)

```py
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        start from top-right to bottom-left
        
        Time: O(M+N)
        Space: O(1)
        """
        rows, cols = len(matrix), len(matrix[0])
        r, c = 0, cols - 1
        
        while r < rows and c >= 0:
            if matrix[r][c] == target:
                return True
            elif matrix[r][c] < target:
                r += 1
            else:
                c -= 1
        
        return False
```

[11. Container With Most Water](https://leetcode.cn/problems/container-with-most-water/description/)

```py
class Solution:
    def maxArea(self, height: List[int]) -> int:
        """
        Time: O(N)
        Space: O(1)
        """
        l, r = 0, len(height) - 1
        area = 0
        
        while l < r:
            cur_area = (r - l) * min(height[l], height[r])
            area = max(area, cur_area)
            
            # 移动短边，消去的面积组合肯定比当前的小
            if height[l] < height[r]: # move the smaller edge, to make the area larger
                l += 1
            else:
                r -= 1
        
        return area
```

[15. 3Sum](https://leetcode.cn/problems/3sum/)

```py
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        排序之后固定一个指针，用双指针
        Time: O(N^2)
        Space: O(logN), depending on how to sort
        """
        nums.sort()
        res = []

        for i, n in enumerate(nums):
            if i > 0 and nums[i - 1] == nums[i]:  # avoid duplicate
                continue

            if nums[i] > 0:  # 如果最小的数都>0，那肯定没有能够满足的结果
                break
            
            l, r = i + 1, len(nums) - 1
            while l < r:
                cur_sum = nums[i] + nums[l] + nums[r]
                if cur_sum < 0:
                    l += 1
                elif cur_sum > 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    # keep moving to find other possibilities
                    l += 1
                    r -= 1
                    while l < r and nums[l - 1] == nums[l]:  # avoid duplicate
                        l += 1
            
        return res
```

### Others

[42. Trapping Rain Water](https://leetcode.cn/problems/trapping-rain-water/description/)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        """
        对于位置i能存储的水: min(maxL, maxR) - h[i]；
        相向双指针，加上两个变量maxL, maxR来时刻保存左右两边的最大值:
        每次移动maxL, maxR之间较小那个数的指针，然后新位置i能存储的水：被移动指针之前的值-h[i]：不用考虑另外一个值，因为那个值肯定比较大；移动指针之后计算这个指针所在位置能存储的水

        时间：O(N)
        空间：O(1) -> two pointers
        """
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

[16. 3Sum Closest](https://leetcode.cn/problems/3sum-closest/)

```py
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        diff = float('inf')
        nums.sort()  # 总是要排序之后才能用双指针
        for i in range(len(nums)):
            l, r = i + 1, len(nums) - 1  # 不用再往前看了，只需要往后看，因为前面看过了
            while l < r:
                total = nums[i] + nums[l] + nums[r]
                if abs(target - total) < abs(diff):
                    diff = target - total
                if total < target:
                    l += 1
                else:
                    r -= 1
                if diff == 0:
                    break
        return target - diff
                
```

[18. 4Sum](https://leetcode.cn/problems/4sum/)

```py
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        先确定前两个数，然后跑后两个数
        
        Time: O(N^3)
        Space: O(1)
        """
        nums.sort()
        n = len(nums)
        res = set()
        for i in range(n):
            for j in range(i + 1, n):
                l, r = j + 1, n - 1
                remain = target - nums[i] - nums[j]
                while l < r:
                    if nums[l] + nums[r] == remain:
                        res.add((nums[i], nums[j], nums[l], nums[r]))
                        l += 1
                        r -= 1
                    elif nums[l] + nums[r] > remain:
                        r -= 1
                    else:
                        l += 1
        return res
```

[454. 4Sum II](https://leetcode.cn/problems/4sum-ii/)

```py
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        """
        counts: {(a + b): freq}
        then enumerate c, d, update the res if (c + d) = -(a + b)
        
        Time: O(N^2)
        Space: O(N^2) for the counter
        """
        
        res = 0
        counter = collections.defaultdict(int)
        for a in nums1:
            for b in nums2:
                counter[a + b] += 1
        
        for c in nums3:
            for d in nums4:
                res += counter[-(c + d)]
        
        return res
        
```

[277. Find the Celebrity](https://leetcode.cn/problems/find-the-celebrity/)

```py
class Solution:
    def findCelebrity(self, n: int) -> int:
        """
        Step1: two pointers to find the potiential celebrity
        Step2: check whether that one is the true celebrity
        """
        
        # Step1
        may_cele = 0
        for nxt_per in range(1, n):
            if knows(may_cele, nxt_per):
                may_cele = nxt_per
        
        # Step2
        for i in range(n):
            if may_cele == i:
                continue
            if knows(may_cele, i) or (not knows(i, may_cele)):
                return -1
        
        return may_cele
```

## 同向双指针

[151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()

        i = j = len(s) - 1  # 从右往左的同向双指针
        res = []
        while i >= 0:
            while i >= 0 and s[i] != ' ':  # 从右往左找空格
                i -= 1
            res.append(s[i + 1: j + 1])  # 左闭右开区间
            while i >= 0 and s[i] == ' ':
                i -= 1
            j = i
        return ' '.join(res)

```

[160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pA, pB = headA, headB
        
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        
        return pA

```

[876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        return slow

```

[392. 判断子序列](https://leetcode.cn/problems/is-subsequence/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        p1 = p2 = 0
        while p1 < len(s) and p2 < len(t):
            if s[p1] == t[p2]:
                p1 += 1
                p2 += 1
            else:
                p2 += 1
        
        return p1 == len(s)
```

[643. Maximum Average Subarray I](https://leetcode.cn/problems/maximum-average-subarray-i/)

[159. Longest Substring with At Most Two Distinct Characters](https://leetcode.cn/problems/longest-substring-with-at-most-two-distinct-characters/)

[283. Move Zeroes](https://leetcode.cn/problems/move-zeroes/)

```py
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        use two pointers: the slow one keeps track of valid value
        
        Time: O(N)
        Space: O(1)
        """
        slow = fast = 0
        
        while fast < len(nums):
            if nums[fast] != 0:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
            
        while slow < len(nums):
            nums[slow] = 0
            slow += 1
        return nums
```

[26. Remove Duplicates from Sorted Array](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

```py
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        """
        use two pointers: the slower one keeps track of valid items
        return slow + 1 as 0-indexed
        
        Time: O(N)
        Space: O(1)
        """
        slow = fast = 0
        while fast < len(nums):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        
        return slow + 1
```

[424. Longest Repeating Character Replacement](https://leetcode.cn/problems/longest-repeating-character-replacement/)

```py
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        """
        用一个counter来记录window中字母出现的次数，这个window是什么字母由max(counter.values())决定，一直expand直到window的大小超过了max+k
        windowLen - max(count[letter]) <= k, valid就移动r，直到不valid就停止

        时间：O(26N)或者O(N)
        空间：O(N)
        """
        count = collections.defaultdict(int)
        res = 0

        l = r = 0
        max_f = 0

        while r < len(s):
            count[s[r]] += 1
            max_f = max(max_f, counts[s[r]])

            # while (r - l + 1) - max(count.values()) > k:
            while (r - l + 1) - max_f > k:
                count[s[l]] -= 1
                l += 1

            res = max(res, r - l + 1)
            r += 1

        return res
```

[76. Minimum Window Substring](https://leetcode.cn/problems/minimum-window-substring/)

```py
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        """
        用count_remain来记录需要配对的剩余的量
        当counter[c] > 0，说明找到一个合适的，如果是在expand时候遇到就要count_remain-=1，如果是在shrink时候遇到就要+=1
        """
        countMap = collections.Counter(t)

        l = r = 0
        minStart = 0
        minLen = float("inf")
        count_remain = len(t)

        while r < len(s):
            # 扩大窗口
            ch = s[r]
            if countMap[ch] > 0:
                count_remain -= 1
            countMap[ch] -= 1 # 扩大的结果
            r += 1

            # 缩小窗口
            while count_remain == 0: # 缩小的条件
                if minLen > r - l: # 更新res
                    minStart = l
                    minLen = r - l
                ch2 = s[l] # 缩小的结果
                countMap[ch2] += 1 # 相互对称，先扩大一个再更新窗口
                if countMap[ch2] > 0:
                    count_remain += 1
                l += 1

        return s[minStart: minStart + minLen] if minLen != float("inf") else ""
```

[3. Longest Substring Without Repeating Characters](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        用set记录当前sliding window的数据；如果s[r]在set里，移动窗口直到不在并且在set中删去

        时间：O(N)
        空间：O(N)
        """
        chars = set()
        l = r = 0
        res = 0

        while r < len(s):
            while s[r] in chars:
                chars.remove(s[l])
                l += 1
            chars.add(s[r])
            res = max(res, r - l + 1)
            r += 1

        return res
```

[121. Best Time to Buy and Sell Stock](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)
同向双指针,L=buy, R=sell：当buy>sell，L=R，否则移动R，同时一直更新profit

时间：O(N)
空间：O(1)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l, r = 0, 1
        res = 0
        cur_profit = 0

        while r < len(prices):
            if prices[l] > prices[r]:
                l = r
            else:
                cur_profit = prices[r] - prices[l]
                res = max(res, cur_profit)
            r += 1

        return res
```

[53. Maximum Subarray](https://leetcode.cn/problems/maximum-subarray/)
Sliding window一直计算当前的window_sum，当<0的时候就清零

```py
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cur_sum = 0
        r = 0
        res = float("-inf")

        while r < len(nums):
            cur_sum += nums[r]
            res = max(res, cur_sum)
            r += 1

            if cur_sum < 0:
                cur_sum = 0

        return res
```

[567. Permutation in String](https://leetcode.cn/problems/permutation-in-string/)

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        counter = collections.Counter(s1)
        l = r = 0
        count = len(s1)

        while r < len(s2):
            c1 = s2[r]
            if counter[c1] > 0:
                count -= 1
            counter[c1] -= 1
            r += 1

            if (r - l == len(s1)):
                if count == 0:
                    return True

                c2 = s2[l]
                counter[c2] += 1
                if counter[c2] > 0:
                    count += 1
                l += 1

        return False
```

[438. Find All Anagrams in a String](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

```py
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        """
        Similar to LC567
        """
        counter = collections.Counter(p)
        l = r = 0
        to_match = len(p)
        res = []
        
        while r < len(s):
            if counter[s[r]] > 0:
                to_match -= 1
            counter[s[r]] -= 1
            r += 1
            if (r - l) == len(p):
                if to_match == 0:
                    res.append(l)
                counter[s[l]] += 1
                if counter[s[l]] > 0:
                    to_match += 1
                l += 1
        
        return res
```

[209. Minimum Size Subarray Sum](https://leetcode.cn/problems/minimum-size-subarray-sum/)

```py
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l = r = 0
        res = float("inf")
        cur_sum = 0
        
        while r < len(nums):
            cur_sum += nums[r]
            r += 1
            while cur_sum >= target:
                res = min(res, r - l)
                cur_sum -= nums[l]
                l += 1
        
        return 0 if res == float("inf") else res
```

[395. Longest Substring with At Least K Repeating Characters](https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/)

```py
class Solution:
    def longestSubstring(self, s, k):
        """
        do sliding window unique_letters(at most 26) times, each time i means the unique letter in the sliding window is i
        """
        def longest_h_unique_at_least_k_repeat(s, k, h):
        
            start = end = 0
            hist = collections.Counter()
            unique = no_less_than_k = 0
            res = 0
        
            while end < len(s):
                hist[s[end]] += 1
                if hist[s[end]] == 1:
                    unique += 1
                if hist[s[end]] == k:
                    no_less_than_k += 1
                
                end += 1
            
                while unique > h:
                    hist[s[start]] -= 1
                    if hist[s[start]] == k-1:
                        no_less_than_k -= 1
                    if hist[s[start]] == 0:
                        unique -= 1
                    start += 1
                if no_less_than_k == unique:
                    res = max(res, end - start)
            return res
        
        counter = collections.Counter(s)
        unique_letters = len(counter)
        
        return max(longest_h_unique_at_least_k_repeat(s, k, i) for i in range(1, unique_letters + 1))
```

[340. Longest Substring with At Most K Distinct Characters](https://leetcode.cn/problems/longest-substring-with-at-most-k-distinct-characters/)

```py
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        counter = collections.defaultdict(int)
        l = r = 0
        res = 0
        
        while r < len(s):
            counter[s[r]] += 1
            r += 1
            while len(counter) > k:
                counter[s[l]] -= 1
                if counter[s[l]] == 0:
                    del counter[s[l]]
                l += 1
            res = max(res, r - l)

        return res
```

[1004. Max Consecutive Ones III](https://leetcode.cn/problems/max-consecutive-ones-iii/)

```py
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        """
        expand window until k < 0, when expanding, k -= 1 when flip needed
        shrink window until k == 0, when shrinking, k += 1 when flip no longer needed
        """
        l = r = 0
        res = 0
        
        while r < len(nums):
            if nums[r] == 0:
                k -= 1
            r += 1
            while k < 0:
                if nums[l] == 0:
                    k += 1
                l += 1
            res = max(res, r - l)
        
        return res
```

[487. Max Consecutive Ones II](https://leetcode.cn/problems/max-consecutive-ones-ii/)

```py
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        res = 0
        zeroes = 0
        l = r = 0
        
        while r < len(nums):
            if nums[r] == 0:
                zeroes += 1
            while zeroes == 2:
                if nums[l] == 0:
                    zeroes -= 1
                l += 1
            res = max(res, r - l + 1)
            r += 1
        
        return res
```

[1052. Grumpy Bookstore Owner](https://leetcode.cn/problems/grumpy-bookstore-owner/description/)

```py
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        """
        找到minutes范围内最大的和
        """
        satis, unsatis, trik_satis = 0, 0, 0

        for end, customer in enumerate(customers):
            if grumpy[end] == 0:
                satis += customer
            else:
                unsatis += customer # 相当于用掉了一个minute
            
            if end >= minutes and grumpy[end - minutes] == 1: # 用完了minute
                unsatis -= customers[end - minutes]

            trik_satis = max(trik_satis, unsatis)
        
        return satis + trik_satis
```

## Others

[31. Next Permutation](https://leetcode.cn/problems/next-permutation/)

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        从后往前找，找到下降序列左边的值k；找到k右边最后一个比k小的数，交换；把k右边的数倒序排列
        时间：O(N)
        空间：O(1)
        """
        i = len(nums)-1
        
        # step1: find the last "ascending" position: k
        while i > 0 and nums[i-1] >= nums[i]: # 要加上=号：直到找到不大于的那个数
            i -= 1 
        if i == 0:   # nums are in descending order
            nums.reverse()
            return 
        k = i - 1   
        
        # step2: 找到k右边比k大的最后一个数，和k交换
        j = len(nums) - 1
        while nums[j] <= nums[k]: # 要加=：直到找到不小于的
            j -= 1 
        nums[k], nums[j] = nums[j], nums[k]  
        
        # step3: 把k右边的数倒序排列
        l, r = k + 1, len(nums) - 1  
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
```

[88. Merge Sorted Array]([Loading...](https://leetcode.cn/problems/merge-sorted-array/))

```py
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        edit nums1 in-place using two pointers
        at the end, what left in nums2 should be added to nums1

        Time: O(M+N)
        Space: O(1)
        """
        while m > 0 and n > 0:
            if nums1[m - 1] > nums2[n - 1]:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1

        nums1[:n] = nums2[:n]
```

[1099. Two Sum Less Than K](https://leetcode.cn/problems/two-sum-less-than-k/)

```py
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        """
        sort, then use two pointers technique to find
        
        Time: O(NlogN)
        Space: O(1)
        """
        nums.sort()
        
        l, r = 0, len(nums) - 1
        res = -1
        
        while l < r:
            cur_sum = nums[l] + nums[r]
            if cur_sum < k:
                res = max(res, cur_sum)
                l += 1
            elif cur_sum > k:
                r -= 1
            else:
                r -= 1
        
        return  res
```

[259. 3Sum Smaller](https://leetcode.cn/problems/3sum-smaller/)

```py
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        """
        Time: O(N^2)
        Space: O(1)
        """
        if len(nums) < 3: 
            return 0
        
        nums.sort()
        res = 0
        
        for i in range(len(nums) - 2):
            res += self.twoSmaller(nums, i + 1, target - nums[i]) # 固定nums[i]是最小的数，接下来的从i+1开始找
        
        return res

    def twoSmaller(self, nums, i, target):
        l, r = i, len(nums) - 1
        count = 0
        
        while l < r:
            cur_sum = nums[l] + nums[r]
            if cur_sum < target:
                count += r - l # 两者之间都是满足条件的
                l += 1
            else:
                r -= 1
        return count
                
```

[905. Sort Array By Parity](https://leetcode.cn/problems/sort-array-by-parity/)

Two pass
时间：O(N)
空间：O(N)

```py
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        return [x for x in nums if x % 2 == 0] + [x for x in nums if x % 2 == 1]
```

Two pointers
时间：O(N)
空间：O(1)

```py
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        i = 0
        j = len(nums) - 1

        while i < j:
            if nums[i] % 2 > nums[j] % 2:
                nums[i], nums[j] = nums[j], nums[i]

            if nums[i] % 2 == 0:
                i += 1

            if nums[j] % 2 == 1:
                j -= 1

        return nums
```

[161. One Edit Distance](https://leetcode.cn/problems/one-edit-distance/)

```py
class Solution(object):
    def isOneEditDistance(self, s, t):
        """
        i, j往后走，如果第二次发现不相同的：return False
        如果第一次：根据s, t相对length来往回退i,j一步
        """
        if abs(len(s) - len(t)) > 1 or s == t:
            return False
        
        found_inequality = False
        i = j = 0
        
        while i < len(s) and j < len(t):
            if s[i] != t[j]:
                if found_inequality: 
                    return False
                found_inequality = True
                if len(s) < len(t): 
                    i -= 1
                elif len(s) > len(t): 
                    j -= 1
            i += 1
            j += 1
        
        return True
```
