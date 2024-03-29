# 贪心

# 高频题

## 知乎

## Krahets精选题

## AlgoMonster

## Youtube

253， 1363，Minimum Cost to Connect Sticks, 122, 435

# 以题型分类

[53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
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

Permute A, B in some way that A'[i] + B'[i] >= k for all i

```py
def twoArrays(k, A, B):
    A.sort()
    B.sort()
    l, r = 0, len(B) - 1
    while l < len(A):
        if A[l] + B[r] < k:
            return "NO"
        l += 1
        r -= 1
    
    return "YES"
```

[134. Gas Station](https://leetcode.com/problems/gas-station/)
计算出diff

时间：O(N)
空间：O(1)

```py
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        
        total = 0
        res = 0

        for i in range(len(gas)):
            total += (gas[i] - cost[i])
            # 从i出发走到j恰好变成负数，那么i-j之间其他的都无法走到终点，所以再从i+1开始走，看能否走到终点
            if total < 0:
                total = 0
                res = i + 1
        
        return res
```

[55. Jump Game](https://leetcode.com/problems/jump-game/)
从后往前移动goal

```py
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) - 1

        for i in range(len(nums) - 1, -1, -1):
            if i + nums[i] >= goal:
                goal = i 
        
        return goal == 0

```

[45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)
BFS把每个位置需要的步数记录下来，用l, r来记录一个window

时间：O(N)
空间：O(1)

```py
class Solution:
    def jump(self, nums: List[int]) -> int:
        res = 0
        l = r = 0

        while r < len(nums) - 1 :
            farthest = 0
            for i in range(l, r + 1):
                farthest = max(farthest, i + nums[i])
            l = r + 1
            r = farthest
            res += 1

        return res

```

[991. Broken Calculator](https://leetcode.com/problems/broken-calculator/)

```py
class Solution:
    def brokenCalc(self, startValue: int, target: int) -> int:
        """
        solve it backward greedly
        """
        res = 0
        while target > startValue:
            res += 1
            if target % 2 == 1:
                target += 1
            else:
                target //= 2
        
        return res - target + startValue
```

[948. Bag of Tokens](https://leetcode.com/problems/bag-of-tokens/)

```py
class Solution:
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        """
        从最小的token开始来+score，power不够用之后从最大的token来-score再从最小的token+score
        """
        tokens.sort()
        
        queue = collections.deque(tokens)
        
        res = 0
        score = 0
        
        while queue and (power >= queue[0] or score):
            while queue and power >= queue[0]:
                score += 1
                power -= queue.popleft()
            
            res = max(res, score)
            
            if queue and score:
                score -= 1
                power += queue.pop()
        
        return res
```
