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

# Time 相关

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



