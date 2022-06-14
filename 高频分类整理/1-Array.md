
# 知乎

[26. Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

```py
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        """
        快慢指针，快指针在前面走，当找到一个不同的数的时候，慢指针走一步并更新
        """
        slow = fast = 0
        while fast < len(nums):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1

        return slow + 1
```

[27. Remove Element](https://leetcode.com/problems/remove-element/)

```py
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        slow = fast = 0
        while fast < len(nums):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
```

[271. Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)
组合大文字的时候用数字+特殊字符来连接，decode时候就需要找到数字大小

时间：O(N), N is num of word in words
空间：O(1)

```py
class Codec:
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        res = ""
        for s in strs:
            res += str(len(s)) + "#" + s
        return res


    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        res = []
        i = 0

        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i : j])
            res.append(s[j + 1: j + 1 + length])
            i = j + 1 + length

        return res

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))
```

# Others
[189. Rotate Array](https://leetcode.com/problems/rotate-array/)
Given an array, rotate the array to the right by k steps, where k is non-negative.
```py
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Place every element of the array at its correct position

        Time: O(N)
        Space: O(N)
        """
        n = len(nums)
        a = [0] * n
        
        for i in range(n):
            a[(i + k) % n] = nums[i]
        
        nums[:] = a
```

```py
class Solution:
    def rotate(self, nums, k) -> None:
        """
        reverse three times: reverse all + reverse first k + reverse last n-k

        n = 7, k = 3
        Original List                   : 1 2 3 4 5 6 7
        After reversing all numbers     : 7 6 5 4 3 2 1
        After reversing first k numbers : 5 6 7 4 3 2 1
        After revering last n-k numbers : 5 6 7 1 2 3 4 --> Result

        Time: O(N)
        Space: O(1)
        """
        k %= len(nums)
        self.reverse(nums, 0, len(nums)-1)
        self.reverse(nums, 0, k-1)
        self.reverse(nums, k, len(nums)-1)

    def reverse(self, nums, start, end) -> None:
        while start < end: 
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
```

```py
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Cyclic Replacement
        
        Time: O(N)
        Space: O(1)
        """
        n = len(nums)
        k %= n
        
        start = count = 0
        while count < n:
            current, prev = start, nums[start]
            while True:
                next_idx = (current + k) % n
                nums[next_idx], prev = prev, nums[next_idx]
                current = next_idx
                count += 1
                
                if start == current:
                    break
            start += 1
```


## Time 相关

[2224. Minimum Number of Operations to Convert Time](https://leetcode.com/problems/minimum-number-of-operations-to-convert-time/)

```py
class Solution:
    def convertTime(self, current: str, correct: str) -> int:
        # current and target time in mins
        current_time = 60 * int(current[0:2]) + int(current[3:5]) 
        target_time = 60 * int(correct[0:2]) + int(correct[3:5])

        # diff in mins
        diff = target_time - current_time

        # Greedy approach
        count = 0 
        for i in [60, 15, 5, 1]:
            count += diff // i # add number of operations needed with i to count
            diff %= i # Diff becomes modulo of diff with i
        return count
```

[12进制时间转换为24进制]
input: "07:05:45PM"
output: "19:05:45"

```py
def timeConversion(s):
    """
    分别根据"AM"和"PM"这两种情况来考虑，注意PM时候转换的技巧
    """
    if s[-2:] == "AM":
        if s[0:2] == "12":
            return "00" + s[2:-2]
        else:
            return str(s[:-2])

    elif s[-2:] == "PM":
        if s[0:2] == "12":
            return str(s[:-2])
        else:
            return str(int(s[0:2]) + 12) + s[2:-2]
```

[14. Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)
先找到最短的字符串，然后依次和其他比较，比较时候发现不相同就返回那个长度，最后返回最短的字符串（只有一个字符串的情况）；本题要点是min(strs, key = len)的使用方法

时间：O(N*S) 
空间：O(min(len(s)))

```py
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""

        shortest = min(strs, key = len)

        for idx, c in enumerate(shortest):
            for other in strs:
                if other[idx] != c:
                    return shortest[:idx]

        return shortest
```

[171. Excel Sheet Column Number](https://leetcode.com/problems/excel-sheet-column-number/)
学会使用ord("A")，以及递归的思路

```py
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        res = 0

        for c in columnTitle:
            res = res * 26
            res += (ord(c) - ord("A") + 1)

        return res
```

[1047. Remove All Adjacent Duplicates In String](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)

```py
class Solution:
    def removeDuplicates(self, s: str) -> str:
        """
        用stack，每次遇到新的就比较一下是否和top相同，相同就弹栈，不同就加进来
        """
        stack = []

        for c in s:
            if stack and c == stack[-1]:
                stack.pop()
            else:
                stack.append(c)

        return "".join(stack)
```

[1209. Remove All Adjacent Duplicates in String II](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/)

```py
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        """
        用stack同时存这个char和出现的次数，一旦出先次数达到k就pop，最后decode到需要的大小
        """
        stack = [["#", 0]]

        for c in s:
            if stack[-1][0] == c:
                stack[-1][1] += 1
                if stack[-1][1] == k:
                    stack.pop()
            else:
                stack.append([c, 1])

        return "".join(c * n for c, n in stack)
```

[5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

```py
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        从头到尾，依次遍历可能的回文字符，每次遍历之后都更新潜在的最大值
        """
        res = ""

        def findPalindrome(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1 # 从中间往两边走
                r += 1
            return s[l + 1: r] # 多走了一步，所以要返回之前的一步

        for i in range(len(s)):
            # odd length
            s1 = findPalindrome(i, i)

            # even length
            s2 = findPalindrome(i, i + 1)

            if len(s1) > len(s):
                res = s1
            if len(s2) > len(s):
                res = s2

        return res
```

[1332. Remove Palindromic Subsequences](https://leetcode.com/problems/remove-palindromic-subsequences/)
```py
class Solution:
    def removePalindromeSub(self, s: str) -> int:
        """
        Asked for subsequency instead of substring
        """
        if not s:
            return 0
        
        if s == s[::-1]:
            return 1
        
        return 2
```

[287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)


[41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)















https://leetcode.com/problems/longest-palindromic-substring/discuss/2030458/Python


