## 回溯整理总结
### 形式一、元素无重不可复选，即nums中的元素都是唯一的，每个元素最多只能被使用一次

```py
"""子集问题 LC77, 78, 90""" 
for i in range(start, len(nums)):
    one_res.append(nums[i])
    backtrack(i + 1)
    one_res.pop()

"""排列问题 LC46"""
for i in range(len(nums)):
    if used[i]:
        continue
    
    used[i] = True
    one_res.append(nums[i]
    backtrack()
    one_res.pop()
    used[i] = False
```

### 形式二、元素可重不可复选，即nums中的元素可以存在重复，每个元素最多只能被使用一次，其关键在于排序和剪枝
```py
"""子集问题 LC40"""
nums.sort()
for i in range(start, len(nums)):
    if i > start and nums[i] == nums[i - 1]:
        continue
    
    one_res.append(nums[i])
    backtrack(i + 1)
    one_res.pop()

"""排列问题 LC47"""
nums.sort()
for i in range(len(nums)):
    if used[i]:
        continue
    
    if i > start and nums[i] == nums[i - 1] and not used[i - 1]:
        continue

    used[i] = True
    one_res.append(nums[i]
    backtrack()s
    one_res.pop()
    used[i] = False
```

### 形式三、元素无重可复选，即nums中的元素都是唯一的，每个元素可以被使用若干次，只要删掉去重逻辑即可
```py
"""子集问题 LC39"""
for i in range(start, len(nums)):
    one_res.append(nums[i])
    backtrack(i) # 注意这里是i
    one_res.pop()

"""排序问题"""
for i in range(len(nums)):
    one_res.append(nums[i])
    backtrack()
    one_res.pop()

```

[22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
添加close的条件：close<open

```py
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        """
        1. add ( if open < n
        2. add ) if close < open
        3. valid if open == close == n 
        """

        stack = []
        res = []
        
        def backtrack(openN, closedN):
            if openN == closedN == n: # base case
                res.append("".join(stack))
                return
            
            if openN < n:
                stack.append("(")
                backtrack(openN + 1, closedN)
                stack.pop()
            
            if closedN < openN:
                stack.append(")")
                backtrack(openN, closedN + 1)
                stack.pop()
        
        backtrack(0, 0)
        return res
```

[78. Subsets](https://leetcode.com/problems/subsets/)（子集 元素无重不可复选）
Given an integer array nums of unique elements, return all possible subsets (the power set).
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]   

时间：O(N*2^N) generate all subsets and copy them
空间：O(N) use O(N) for one_res
```py
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """通过保证元素之间的相对顺序不变来防⽌出现重复的⼦集"""
        one_res = []
        res = []
        
        def backtrack(start):
            # 添加条件：每个中间结果都是最终结果
            res.append(one_res.copy())
            
            # 子集问题：i从start开始
            # 通过 start 参数控制树枝的遍历，避免产生重复的子集            
            for i in range(start, len(nums)):
                one_res.append(nums[i]) # 做选择
                backtrack(i+1)
                one_res.pop() # 撤销选择
        
        backtrack(0)
        return res
```

[77. Combinations](https://leetcode.com/problems/combinations/) （组合 元素无重不可复选）
Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].

You may return the answer in any order.

Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

```py
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        one_res = []
        res = []

        def backtrack(start):
            # 添加条件：长度是k
            if len(one_res) == k: 
                res.append(one_res.copy())
                return

            # 子集问题：i从start开始
            # 通过 start 参数控制树枝的遍历，避免产生重复的子集
            for i in range(start, n + 1):
                one_res.append(i)
                backtrack(i+1)
                one_res.pop()
            
        backtrack(1)
        return res
```

[46. Permutations](https://leetcode.com/problems/permutations/) 排列（元素无重不可复选）

Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        one_res = []
        used = [False] * len(nums)

        def backtrack():
            # 添加条件： 长度
            if len(one_res) == len(nums):
                res.append(one_res.copy())
                return
            
            # 满树问题：i从0开始
            for i in range(len(nums)):
                # 跳过不合法的选择，否则结果有[1,1,1],[1,1,2]...
                if used[i]:
                    continue
                
                used[i] = True
                one_res.append(nums[i])
                backtrack()
                one_res.pop()
                used[i] = False
        
        backtrack()
        return res
```

[90. Subsets II](https://leetcode.com/problems/subsets-ii/)
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

```py
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """需要先进行排序，让相同的元素靠在一起，如果发现nums[i] == nums[i-1]，则跳过"""
        nums.sort()
        oneRes = []
        res = []
        
        def backtrack(start):
            # 添加条件：每个中间结果都是最终结果
            res.append(oneRes.copy())
            
            # 子集问题：i从start开始
            for i in range(start, len(nums)):
                # 跳过不合法选择，否则有最终结果有两个[1,2]
                if i > start and nums[i] == nums[i-1]:
                    continue
                oneRes.append(nums[i])
                backtrack(i+1)
                oneRes.pop()
        
        backtrack(0)
        return res
```

[40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

```py
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """类似LC90，只是base case不同"""
        candidates.sort() # 元素可以重复，所以要排序
        res = []
        one_res = []
        
        def backtrack(start, target):
            if target < 0:
                return
            
            if target == 0:
                res.append(one_res.copy())
            
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                
                one_res.append(candidates[i])
                target -= candidates[i]
                backtrack(i + 1, target)
                one_res.pop()
                target += candidates[i]
        
        backtrack(0, target)
        return res
```

[39. Combination Sum](https://leetcode.com/problems/combination-sum/)

Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]

```py
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        one_res = []
        candidates.sort() # 元素可重复，所以要排序
        
        def backtrack(start, target):
            if target < 0:
                return
            
            if target == 0:
                res.append(one_res.copy())
                return
            
            for i in range(start, len(candidates)):
                # 与LC40不同点：不怕重复，所以不需要选择条件
                one_res.append(candidates[i])
                target -= candidates[i]
                backtrack(i, target) # 与LC40不同点，可以重复，所以从i再用一次
                one_res.pop()
                target += candidates[i]
        
        backtrack(0, target)
        return res
```            
            

[47. Permutations II](https://leetcode.com/problems/permutations-ii/)

Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]

```py
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        one_res = []
        nums.sort()
        visit = [False] * len(nums)
        
        def backtrack():
            if len(one_res) == len(nums):
                res.append(one_res.copy())
                return
        
            for i in range(len(nums)):
                
                if visit[i]:
                    continue
            # 新添加的剪枝逻辑，固定相同的元素在排列中的相对位置
            # not visited[i - 1]保证相同元素在排列中的相对位置保持不变。
                if i > 0 and nums[i] == nums[i - 1] and not visit[i - 1]:
                    continue
                    
                visit[i] = True
                one_res.append(nums[i])
                backtrack()
                one_res.pop()
                visit[i] = False
        
        backtrack()
        return res
```
