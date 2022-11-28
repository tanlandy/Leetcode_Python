# 基础知识
In computers, data are stored as bits. A bit stores either 0 or 1. 
A binary number is a number expressed in the base-2 system. Each digit can be 0 or 1.

010101 = 1\*2^0 + 1\*2^2 + 1\*2^4
in python: 
`bin(21)` returns `010101`

8进制转10进制：
720.5(8) = 7 * 8^2 + 2 * 8^1 + 0 * 8^0 + 5 * 8^-1 = 464.625

10进制转2进制：
50 / 2 = 25; 50 % 2 = 0

25 / 2 = 12; 25 % 2 = 1

12 / 2 = 6; 12 % 2 = 0

6 / 2 = 3; 6 % 2 = 0

3 / 2 = 1; 3 % 2 = 1

1 / 2 = 0; 1 % 2 = 1
最后把所有的余数倒过来排列
50(10) = 110010(2)

## bitwise operation
Bit-wise AND:
compare each digit, if both are 1, then resulting digit is 1.

Bitmask
construct a binary number, such that it can turns off all digits except the 1 digit in the mask.
-> 只保留自己是1的位置，把其他位置的1都掩盖掉了

### and &
for corresponding bits, if both numbers are 1, the result is 1.
0 & 0 = 0
0 & 1 = 0
1 & 0 = 0
1 & 1 = 1

### or |
when both are 0, the result is 0. 
0 | 0 = 0
0 | 1 = 1
1 | 0 = 1
1 | 1 = 1

### XOR ^
when both are the same, the result is 0
0 ^ 0 = 0
0 ^ 1 = 1
1 ^ 0 = 1
1 ^ 1 = 0

### ~
flip each bit
~ 0 = 1
~ 1 = 0


## shift operation

### <<
left shift: 往左移
29 << 2 = 116
0001101 往左移2位得到 01110100
同时相当于 29 * 2^2 = 29 * 4 = 116

## Properties

`1 << i` access the ith bit in the mask
`bitmask | (1 << i)` set the ith bit in the bitmask to 1
`bitmask & (1 << i)` check if the ith bit in the bitmask is set 1 or not
`bitmask & ~(1 << i)` set the ith bit in the bitmask to 0

AND: `a & 0 = 0`
OR: `a | 0 = a, a | (~a) = -1`
XOR: `a ^ 0 = a, a ^ a = 0`
`a & (a - 1)` 把a最后一个1变成0
`a & (-a)` 只保留最后一个1，其他都变成0



# 例题

[504. Base 7](https://leetcode.com/problems/base-7/description/)

```py
class Solution:
    def convertToBase7(self, num):
        # 十进制转七进制
        n, res = abs(num), ""
        while n:
            res = str(n % 7) + res # 从后往前加余数，所以干脆每次加到后面
            n //= 7
        return '-' * (num < 0) + res or "0" # 处理num <= 0的情况
```

[405. Convert a Number to Hexadecimal](https://leetcode.com/problems/convert-a-number-to-hexadecimal/description/)

```py
class Solution:
    def toHex(self, num: int) -> str:
        # 十进制转十六进制
        pos = "0123456789abcdef"
        res = ""
        for i in range(8): # 32 / 4 = 8, a 10-based num is 32 bytes, a 16-based num is 4 bytes
            idx = num % 16
            res = pos[idx] + res
            num //= 16
        return res.lstrip("0") or "0"
```

[191. Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/description/)

```py
class Solution:
    def hammingWeight(self, n: int) -> int:
        """
        不断把最后一位1变成0
        """
        res = 0
        while n:
            res += 1
            n &= (n - 1) # 把n的最后一位变成0
        return res
```

[190. Reverse Bits](https://leetcode.com/problems/reverse-bits/description/)

```py
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        power = 31
        while n:
            res += (n & 1) << power # 把对应位置的数反转后加过来，然后放到对应位置
            n >>= 1
            power -= 1
        return res
```

[201. Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/description/)

```py
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        """
        变成找这两个数最左边的1的位置
        """
        shift = 0
        while left < right:
            left >>= 1
            right >>= 1
            shift += 1
        return left << shift
```

```py
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        """
        也可以把大数只保留最左边的1
        """
        while left < right:
            right &= (right - 1)
        return right
```


[231. Power of Two](https://leetcode.com/problems/power-of-two/description/)

```py
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        return n & (n - 1) == 0 # 把最左边的1变成0的话，如果是2的倍数就应该变成0
```

[136. Single Number](https://leetcode.com/problems/single-number/)

```py
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
        XOR: 1^1 == 0, 0^0 == 0, n^0 == n

        Time: O(N)
        Space: O(1)
        """
        res = 0 # x ^ 0 = n
        for n in nums:
            res = n ^ res
        return res        

```

Flip all the bits and return the result as an unsigned integer.

```py
def flippingBits(n):
    ones = ["1"] * 32
    ones = "".join(ones)
    return int(n) ^ int(ones, 2)
```


[338. Counting Bits](https://leetcode.com/problems/counting-bits/)

```py
class Solution:
    def countBits(self, n: int) -> List[int]:
        """
        1 + dp[n-i], i is offset, the most significant bit reached so far 
        """
        dp = [0] * (n + 1)
        offset = 1

        for i in range(1, n + 1):
            if offset * 2 == i:
                offset = i
            dp[i] = 1 + dp[i - offset]

        return dp
```

[190. Reverse Bits](https://leetcode.com/problems/reverse-bits/)
```py
class Solution:
    def reverseBits(self, n: int) -> int:
        """
        01 << 1: shift left by one: 010 -> 10
        Time: O(1) only 32 bits
        Space: O(1)
        """
        res = 0
        for i in range(32):
            bit = (n >> i) & 1 # >> to get spot one by one 
            res = res | (bit << (31 - i))

        return res

```


[137. Single Number II](https://leetcode.com/problems/single-number-ii/)



pseudocode for DP
```shell
function f(int bitmask, int [] dp) {
    if calculated bitmask {
        return dp[bitmask];
    }
    for each state you want to keep track of {
        if current state not in mask {
            temp = new bitmask;
            dp[bitmask] = max(dp[bitmask], f(temp,dp) + transition_cost);
        }
    }
    return dp[bitmask];
}
```
Bitmask is helpful with problems that would normally require factorial complexity (something like n!) but can instead reduce the computational complexity to 2^n by storing the dp state. 

[136. Single Number](https://leetcode.com/problems/single-number/)

```py
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        a = 0
        for n in nums:
            a = a ^ n 
            # 0 XOR a = a
            # a XOR a = 0
        
        return a

```

[137. Single Number II](https://leetcode.com/problems/single-number-ii/)

```py
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        
        seen_once = seen_twice = 0
        
        for n in nums:
            seen_once = ~seen_twice & (seen_once ^ n)
            seen_twice = ~seen_once & (seen_twice ^ n)
            
        return seen_once
```

[260. Single Number III](https://leetcode.com/problems/single-number-iii/)

```py
class Solution:
    def singleNumber(self, nums: int) -> List[int]:
        # difference between two numbers (x and y) which were seen only once
        bitmask = 0
        for num in nums:
            bitmask ^= num
        # bitmask = 0 ^ x ^ y
        
        # rightmost 1-bit diff between x and y
        diff = bitmask & (-bitmask) # a & (-a) keeps the rightmost 1-bit and sets all the others to 0
        # -bitmask == ~bitmask + 1
        
        x = 0
        for num in nums:
            # bitmask which will contain only x
            if num & diff: # y & diff == 0
                x ^= num
        
        return [x, bitmask ^ x]
```


[1457. Pseudo-Palindromic Paths in a Binary Tree](https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/)

```py
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        """
        O(1) space complexity solution
        """
        
        res = [0]
        
        def dfs(node, path):
            if not node:
                return
            
            path = path ^ (1 << node.val) # left shift operator to define the bit. XOR to compute the digit frequency
            
            if not node.left and not node.right: # reach the leaf
                if path & (path - 1) == 0: # path & (path - 1) set the rightmost 1 to 0, if is equals 0, means there's only one 1 in path
                    res[0] += 1
                return
            
            dfs(node.left, path)
            dfs(node.right, path)
        
        dfs(root, 0)
        return res[0]
```