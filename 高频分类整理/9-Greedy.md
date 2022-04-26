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
            if total < 0:
                total = 0
                res = i + 1
        
        return res
```