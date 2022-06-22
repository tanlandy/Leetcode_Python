# OA

[2178. Maximum Split of Positive Even Integers](https://leetcode.com/problems/maximum-split-of-positive-even-integers/)

从2，4，6开始加同时finalSum-=246，直到curRes>finalSum，这个时候就res[-1]+=finalSum就可以了

```python
class Solution:
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        res = []
        curRes = 2
        
        if finalSum %2 == 0:
            while curRes <= finalSum:
                res.append(curRes)
                finalSum -= curRes
                curRes += 2
            res[-1] += finalSum
        
        return res

```

given a string S consisting of lowercase letters of the English alphabet, returns the longest consistent fragment of S which begins and ends with the same letter. If there are many possible answers you should return the one starting at the earliest position.
Examples:

1. Given S = "cbaabaab", your function should return "baabaab".

2. Given S = "performance", your function should return "erformance".

3. Given S = "cat", your function should return "c".


两个字典分别从前后来存字母，如果出现了就直接返回

```python
def findLS(s):
    di = {}
    dj = {}
    
    i, j = 0, len(s) - 1
    
    while i < j:
        if s[j] not in di and s[j] not in dj:
            dj[s[j]] = j
        if s[j] in di:
            return s[di[s[j]]:j + 1]
        if s[i] not in di and s[i] not in dj:
            di[s[i]] = i
        if s[i] in dj:
            return s[i:dj[s[i]] + 1]
        i += 1
        j -= 1
        
    if i == j:
        return s[0]

```


```python
def findLS(S):
    d = {}

    # store the furthest index of the letters in S
    for i in range(len(S) - 1, -1, -1):
        if S[i] not in d:
            d[S[i]] = i

    max_length = float("-infinity")
    best_index = 0

    # loop from the beginning of S
    for i, let in enumerate(S):
        # calculate the distance from current instance of the letter to the last
        sub_length = d[let] + 1 - i
        # only update if the distance is greater than max
        # this means we always start our answer from the earliest index possible. 
        if sub_length > max_length:
            best_index = i
            max_length = sub_length
            
    return S[best_index:d[S[best_index]] + 1]

print(findLS("performance") == "erformance")
print(findLS("adsaas") == "adsaa")
print(findLS("adsaass") == "adsaa")
print(findLS("adsaasss") == "saasss")

```


given an array consisting of N integers, returns the maximum possible number of pairs with the same sum. each array may belong to one pair only. (focus on the correctness, not the performance)

A = [1,9,8,100,2] output; 2 (A[0],A[1]) and (A[2], A[4])

A = [2,2,2,3] output; 1 (A[0], A[1])

排序之后，计算和为1-2001的每种可能的数量，取最大的
时间：O(NlogN)
空间：O(N)

```python
arr = [2,2,2,3]

a = arr
a.sort()
def kaafi_bekar(target_sum):
    i, j = 0, len(arr)-1
    counter = 0
    while(i<j):
        curr_sum = a[i]+a[j]
        if curr_sum<target_sum:
            i+=1
        elif curr_sum>target_sum:
            j-=1
        else:
            counter+=1
            j-=1
            i+=1
    return counter

max_counter = 0
for i in range(2001):
    curr = kaafi_bekar(i)
    max_counter = max(max_counter, curr)

print(max_counter)
```


Find the length of the longest substring that every character h‍as the same occurrences: input s="ababbcbc",出现次数相同的且最长的是"abab","bcbc"，长度是4

暴力解，把所有的substring都看一下，从中找出满足条件且最长的;字典：{letter: counter}，字典里面存每个字母出现的次数;看条件是否满足可以通过查看是否字典里的最大值和最小值相等

时间：O(N^2)
空间：O(N^2)

```python
def solution(s: str) -> int:
    """
    暴力解，把所有的substring都看一下，从中找出满足条件且最长的
    字典：{letter: counter}，字典里面存每个字母出现的次数
    看条件是否满足可以通过查看是否字典里的最大值和最小值相等
    
    时间：O(N^2)
    空间：O(N^2)
    """
    n = len(s)
    if n <= 1:
        return n
    counter = collections.defaultdict(int)
    curRes = 0
    res = 0
    
    for i in range(n - 1):
        counter.clear()
        for j in range(i, n):
            counter[s[j]] += 1
            if min(counter.values()) == max(counter.values()):
                curRes = j + 1 - i
                res = max(res, curRes)
            
    return res
 
 
if __name__ == "__main__":
    # assert solution("") == 0
    print(solution("ababbcbc") == 4)
    print(solution("aabcde") == 5)
    print(solution("aaaa") == 4)
    print(solution("beeebbbccc") == 9)
    print(solution("daababbd") == 6)
    print(solution("abcabcabcabcabcabcabcabcabcpabcabcabcabcabcabcabcabcabcabczabcabc") == 30)
    
    

```


# VO

[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)


时间：O(sqrt(N))
空间：O(N)

```python
class Solution:
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        res = []
        curRes = 2
        
        if finalSum %2 == 0:
            while curRes <= finalSum:
                res.append(curRes)
                finalSum -= curRes
                curRes += 2
            res[-1] += finalSum
        
        return res
```


[670. Maximum Swap](https://leetcode.com/problems/maximum-swap/)

先把num变成一个list，从后往前，i是index，如果这个值更小，就说明可以和max_idx互换，就把他们换一下；如果这个值更大，就说明更新max_idx；最后把list转换成num；num变成list：num = [int(x) for x in str(num)]；list变num：int("".join([str(x) for x in num])

时间：O(N)
空间：O(N)

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        num = [int(x) for x in str(num)]
        max_idx = len(num) - 1
        
        x_min = 0
        x_max = 0
        
        # 从后往前，i是index，如果这个值更小，就说明可以和max_idx互换，就把他们换一下；如果这个值更大，就说明更新max_idx
        for i in range(len(num) - 1, -1, -1):
            # 如果这个值更大，就更新max的idx
            if num[i] > num[max_idx]:
                max_idx = i
            # 如果这个值更小，就说明可以和max_idx互换，就把他们换一下
            elif num[i] < num[max_idx]:
                x_min = i
                x_max = max_idx
        
        num[x_min], num[x_max] = num[x_max], num[x_min]
        
        return int("".join([str(x) for x in num]))
```

# Top 200

[366. Find Leaves of Binary Tree](https://leetcode.com/problems/find-leaves-of-binary-tree/)

```py
class Solution:
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        站在每个节点：要知道自己的层数，然后把自己加到和自己层数相同的列表里
        知道层数：从子树返回高度->递归

        Time: O(N)
        Space: O(N)
        """
        res = collections.defaultdict(list)
        
        def dfs(node, height):
            if not node:
                return 0
            left = dfs(node.left, height)
            right = dfs(node.right, height)       
            height = max(left, right)
            res[height].append(node.val)
            return height + 1
        
        dfs(root, 0)
        return res.values()
```

[2096. Step-By-Step Directions From a Binary Tree Node to Another](https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/)

```py
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        """
        Find LCA of inputs
        get paths from LCA to start and destination
        convert LCA_start path to "U" and then concatenate with the other path
        """
        def LCA(node, p, q):
            if not node:
                return node
            if node.val == p or node.val == q:
                return node
            left = LCA(node.left, p, q)
            right = LCA(node.right, p, q)
            if left and right:
                return node
            else:
                return left or right
        
        lca = LCA(root, startValue, destValue)

        self.ps = self.pd = ""

        # backtracking
        def dfs(node, path):
            if not node or (self.ps and self.pd):
                return
            
            if node.val == startValue:
                self.ps = "U" * len(path)
            if node.val == destValue:
                self.pd = "".join(path)
            
            if node.left:
                path.append("L")
                dfs(node.left, path)
                path.pop()
            if node.right:
                path.append("R")
                dfs(node.right, path)
                path.pop()
        
        dfs(lca, [])
        return self.ps + self.pd
```


[250. Count Univalue Subtrees](https://leetcode.com/problems/count-univalue-subtrees/)
```py
class Solution:
    def countUnivalSubtrees(self, root):
        """
        需要知道子树信息
        dfs(node)返回boolean: 该节点和父亲节点相同，且自己的两个子节点也都满足，才返回True
        """
        res = [0]
        def dfs(node, parent):
            if not node:
                return True
            left = dfs(node.left, node.val)
            right = dfs(node.right, node.val)
            if left and right:
                res[0] += 1
            return left and right and node.val == parent
        dfs(root, None)
        return res[0]
```