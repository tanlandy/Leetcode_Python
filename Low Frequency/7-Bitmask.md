In computers, data are stored as bits. A bit stores either 0 or 1. 
A binary number is a number expressed in the base-2 system. Each digit can be 0 or 1.

010101 = 1\*2^0 + 1\*2^2 + 1\*2^4
in python: 
`bin(21)` returns `010101`

Bit-wise AND:
compare each digit, if both are 1, then resulting digit is 1.

Bitmask
construct a binary number, such that it can turns off all digits except the 1 digit in the mask.
-> 只保留自己是1的位置，把其他位置的1都掩盖掉了

Common operation
`1 << i` access the ith bit in the mask
`bitmask | (1 << i)` set the ith bit in the bitmask to 1
`bitmask & (1 << i)` check if the ith bit in the bitmask is set 1 or not
`bitmask & ~(1 << i)` set the ith bit in the bitmask to 0

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