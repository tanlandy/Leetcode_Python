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

[1293. Shortest Path in a Grid with Obstacles Elimination](https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/)
```py
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        """
        BFS on (x,y,r) x,y is coordinate, r is remain number of obstacles you can remove.
        queue: (step, state)
        visited: (state)
        state: (r, c, k)

        Time: O(NK), visit each cell K times
        Space: O(NK)
        """
        rows, cols = len(grid), len(grid[0])
        
        if k >= rows + cols - 2:
            return rows + cols - 2
        
        state = (0, 0, k)
        queue = collections.deque([(0, state)])
        visited = set([state])
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while queue:
            steps, (r, c, k) = queue.popleft()
            if r == rows - 1 and c == cols - 1:
                return steps
            
            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                if 0 <= nei_r < rows and 0 <= nei_c < cols:
                    nei_k = k - grid[nei_ar][nei_c]
                    nei_state = (nei_r, nei_c, nei_k)
                    if nei_k >= 0 and nei_state not in visited:
                        visited.add(nei_state)
                        queue.append((steps + 1, nei_state))
        
        return -1
```

```py
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        """
        prioritize exploring the most promising directions at each step: Use priority queue to store the order of visits, the order is based on the estimated total cost function f(n) = g(n) + h(n)
        
        Time: O(NK*log(NK)), visit each cell K times, each time: log(NK)
        Space: O(NK)
        """
        rows, cols = len(grid), len(grid[0])
        
        def dist(r, c):
            return rows - 1 - r + cols - 1 - c
        
        # (r, c, remaining_k)
        state = (0, 0, k)
        
        # (total_dist, steps, state)
        minHeap = [(dist(0, 0), 0, state)]
        visited = set([state])
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while minHeap:
            total_dist, steps, (r, c, remain_k) = heapq.heappop(minHeap)
            
            if total_dist - steps <= remain_k:
                return total_dist
            
            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                if 0 <= nei_r < rows and 0 <= nei_c < cols:
                    nei_k = remain_k - grid[nei_r][nei_c]
                    nei_state = (nei_r, nei_c, nei_k)
                    if nei_k >= 0 and nei_state not in visited:
                        visited.add(nei_state)
                        nei_total_dist = dist(nei_r, nei_c) + steps + 1
                        heapq.heappush(minHeap, (nei_total_dist, steps + 1, nei_state))
        
        return -1
```

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

```py
class Solution:
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        拓扑排序：把所有outdegree==0的node一一删除
        """
        
        # build graph: {child: parents} and outdegree
        graph = collections.defaultdict(list)
        outdegree = collections.defaultdict(int)
        
        def dfs(cur, prev):
            if cur == None:
                return
            outdegree[prev] += 1
            outdegree[cur] += 0
            graph[cur].append(prev)
            
            dfs(cur.left, cur)
            dfs(cur.right, cur)
        
        dfs(root, TreeNode(-101)) # the range of val is between -100 and 100
        
        # add valid nodes to queue
        queue = collections.deque()
        for node in outdegree:
            if outdegree[node] == 0:
                queue.append(node)
                        
        # build res using topological sort
        res = []
        while queue:
            size = len(queue)
            cur_level = []
            for _ in range(size): # build one level
                cur = queue.popleft()
                if cur.val > -101:
                    cur_level.append(cur.val)
                for nei in graph[cur]:
                    outdegree[nei] -= 1
                    if outdegree[nei] == 0:
                        queue.append(nei)
            if cur_level: # add one level to res
                res.append(cur_level)
        
        return res
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

[818. Race Car](https://leetcode.com/problems/race-car/)

```py
class Solution:
    def racecar(self, target: int) -> int:
        """
        Sol1: Greedily BFS
        For Naive BFS: 2^N, each time we have two choices. 
        ->Optimized with memorization: store the visited car speed and position
        """
        
        #1. Initialize double ended queue as 0 moves, 0 position, +1 velocity
        state = (0, 1)
        queue = collections.deque([(0, state)])
        visited = set(state)
        while queue:
            moves, (pos, vel) = queue.popleft()

            if pos == target:
                return moves
            
            #2. Always consider moving the car in the direction it is already going
            forward_state = (pos + vel, 2 * vel)
            if forward_state not in visited:
                queue.append((moves + 1, (forward_state)))
                visited.add(forward_state)
            
            #3. Only consider changing the direction of the car if one of the following conditions is true
            #   i.  The car is driving away from the target.
            #   ii. The car will pass the target in the next move.  
            if (pos + vel > target and vel > 0) or (pos + vel < target and vel < 0):
                back_state = (pos, -1 if vel > 0 else 1)
                if back_state not in visited:
                    queue.append((moves + 1,back_state))
                    visited.add(back_state)
```

```py
class Solution:
    def racecar(self, target: int) -> int:
        """
        Sol2: Basic DP 
        DP: dp(i) be the shortest instructions to move car from position 0 to position i. Return value is dp[target], base case is dp[0] = 0
        case1: 一直走m次就到了 dp[target] = m, when target == 2 ** m - 1
        case2: 走m次之后没到target就掉头，走了n次再掉头，再往前走直到到了 dp[target] = m + 1 + n + 1 + dp[target-(pm-pn)], where pm = 2 ** m - 1, pn = 2 ** n - 1
        case3: 走m次之后超过target再掉头，直到走到了 dp[target] = m + 1 + dp[pm - target], where pm = 2 ** m - 1 
        """
        dp = [float("inf")] * (target + 1)

        dp[0] = 0
        for i in range(1, target + 1):
            m, pm = 1, 1
            while pm < i:
                n, pn = 0, 0
                while pn < pm:
                    dp[i] = min(dp[i], m + 1 + n + 1 + dp[i - (pm - pn)])
                    n += 1
                    pn = 2 ** n - 1
                m += 1
                pm = 2 ** m - 1
            dp[i] = min(dp[i], m + (0 if i == pm else 1 + dp[pm - i]))

        return dp[target]
```

```py
class Solution:
    # def racecar(self, target: int) -> int:
    #     """
    #     Sol3: Greedily DP
    #     """
    dp = {0: 0}
    def racecar(self, t):
        if t in self.dp:
            return self.dp[t]
        n = t.bit_length()
        if 2**n - 1 == t:
            self.dp[t] = n
        else:
            self.dp[t] = self.racecar(2**n - 1 - t) + n + 1
            for m in range(n - 1):
                self.dp[t] = min(self.dp[t], self.racecar(t - 2**(n - 1) + 2**m) + n + m + 1)
        return self.dp[t]
            
```

[2128. Remove All Ones With Row and Column Flips](https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips/)
```py
class Solution:
    def removeOnes(self, grid: List[List[int]]) -> bool:
        """
        the order of the operations doesn't matter
        doing more than 1 operation on the same row/col is not useful
        
        step:
        1. flip rows, make sure all rows be the "same"
        2. flip cols

        Time: O(M*N)
        Space: O(M*N)
        """
        r1, r1_flip = grid[0], [1 - val for val in grid[0]]
        
        for i in range(1, len(grid)):
            if grid[i] != r1 and grid[i] != r1_flip:
                return False
        
        return True
```

[2115. Find All Possible Recipes from Given Supplies](https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies/)

```py
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        """
        拓扑排序
        graph: {ingredients: recipies}
        indegree: recipes
        遍历一遍recipes和ingredients，构建出来graph和indegree
        
        queue装所有的supplies，然后更新indegree，把indegree[i]==0的放进queue
        res是拓扑排序之后indegree[i]==0的那些i
        """
        
        # graph: {ingredients: recipies}
        graph = collections.defaultdict(list)
        # indegree: required for recipes
        indegree = collections.defaultdict(int)
        
        for i in range(len(recipes)):
            for ing in ingredients[i]:
                indegree[recipes[i]] += 1
                graph[ing].append(recipes[i])
        
        
        # 拓扑排序
        queue = collections.deque(supplies)
        while queue:
            cur = queue.popleft()
            for nei in graph[cur]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    queue.append(nei)
        
        res = []
        for recipe in recipes:
            if indegree[recipe] == 0:
                res.append(recipe)
        
        return res
```

```py
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        """
        use DFS with Memorization to find the one
        """
        suppliesSet = set(supplies)
        recipesMap = {recipes[i]: ingredients[i] for i in range(0, len(recipes))}
        ans = []
        
        for recipe in recipesMap:
            if self.canMake(recipe, suppliesSet, recipesMap, set()):
                ans.append(recipe)
                
        return ans
    
    def canMake(self, target, suppliesSet, recipesMap, seen):
        if target in suppliesSet:
            return True
        if target in seen:
            return False
        if target not in recipesMap:
            return False
        
        seen.add(target)
        
        for ingredient in recipesMap[target]:
            if not self.canMake(ingredient, suppliesSet, recipesMap, seen):
                return False
        
        suppliesSet.add(target)
        return True
        
```

[2013. Detect Squares](https://leetcode.com/problems/detect-squares/)
```py
class DetectSquares:

    def __init__(self):
        # counter: {point: freq}
        self.counter = collections.Counter()

    def add(self, point: List[int]) -> None:
        self.counter[tuple(point)] += 1

    def count(self, point: List[int]) -> int:
        res = 0
        x1, y1 = point
        # for each point, make it the potientail diagnal point
        for (x3, y3), freq in self.counter.items():
            # use the diagnal points to generate square
            if x1 == x3 or abs(x1 - x3) != abs(y1 - y3):
                continue
            # += freq * p2_freq * p4_freq
            res += freq * self.counter[(x3, y1)] * self.counter[(x1, y3)]
        return res
```

[359. Logger Rate Limiter](https://leetcode.com/problems/logger-rate-limiter/)
```py
class Logger:

    def __init__(self):
        self.msg_time = {}

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message not in self.msg_time:
            self.msg_time[message] = timestamp
            return True
        else:
            if timestamp >= self.msg_time[message] + 10:
                self.msg_time[message] = timestamp
                return True
            else:
                return False
```

再看一下set和queue的解法

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

```py
class Solution:
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        拓扑排序：把所有outdegree==0的node一一删除
        """
        
        # build graph: {child: parents} and outdegree
        graph = collections.defaultdict(list)
        outdegree = collections.defaultdict(int)
        
        def dfs(cur, prev):
            if cur == None:
                return
            outdegree[prev] += 1
            outdegree[cur] += 0
            graph[cur].append(prev)
            
            dfs(cur.left, cur)
            dfs(cur.right, cur)
        
        dfs(root, TreeNode(-101)) # the range of val is between -100 and 100
        
        # add valid nodes to queue
        queue = collections.deque()
        for node in outdegree:
            if outdegree[node] == 0:
                queue.append(node)
        
        # build res using topological sort
        res = []
        while queue:
            size = len(queue)
            cur_level = []
            for _ in range(size): # build one level
                cur = queue.popleft()
                if cur.val > -101:
                    cur_level.append(cur.val)
                for nei in graph[cur]:
                    outdegree[nei] -= 1
                    if outdegree[nei] == 0:
                        queue.append(nei)
            if cur_level: # add one level to res
                res.append(cur_level)
        
        return res
```

```py
class Solution(object):
    
    def findLeaves(self, root):
        
        if not root:
            return []
        result=[]
        while root:
            curLeaves = []
            root = self._findLeaves(root, curLeaves)
            
            result.append(curLeaves)
        
        return result 

    def _findLeaves(self, root, curLeaves):
        """
        remove the leave
        """
        if not root:
            return None
        if not root.left and not root.right:
            curLeaves.append(root.val)
            return None
        else:
            root.left = self._findLeaves(root.left, curLeaves)
            root.right = self._findLeaves(root.right, curLeaves)
            return root
```


[2034. Stock Price Fluctuation](https://leetcode.com/problems/stock-price-fluctuation/)

```py
from sortedcontainers import SortedDict

class StockPrice:

    def __init__(self):
        """
        need to record the price at different timestamps: hashmap
        need to get the lowest and highest and latest price: sorted map(treemap in Java)
        hashmap: {timestamp : price}
        sorted map: {price : count} 为了便于之后删除
        
        """
        self.time_price = {}
        self.price_count = SortedDict()
        self.cur = 0


    def update(self, timestamp: int, price: int) -> None:
        """
        Time: O(NlogN) each time in sorted map takes logN time
        Space: O(N) eacxh time takes O(1)
        """
        self.cur = max(timestamp, self.cur)
        if timestamp in self.time_price:
            old_price = self.time_price[timestamp]
            self.price_count[old_price] -= 1
            if not self.price_count[old_price]:
                del self.price_count[old_price]
            
        self.time_price[timestamp] = price
        if price in self.price_count:
            self.price_count[price] += 1
        else:
            self.price_count[price] = 1
        
    def current(self) -> int:
        """
        Time: O(1)
        Space: O(1)
        """
        return self.time_price[self.cur]

    def maximum(self) -> int:
        """
        Time: O(1)
        Space: O(1)
        """
        return self.price_count.peekitem(-1)[0]

    def minimum(self) -> int:
        """
        Time: O(1)
        Space: O(1)
        """        
        return self.price_count.peekitem(0)[0]
```

相同思路用HashMap in Java
语法上的几个区别：
1. cur, timePrice都声明写在class内，在构造函数里面生成
2. map的几个函数的使用.put(), .get(), .containsKey()
3. map在原有数据map.put(key, map.getOrDefault(key, defaultValue) + 1)
```java
class StockPrice {

    int cur;
    Map<Integer, Integer> timePrice;
    TreeMap<Integer, Integer> priceCount;

    public StockPrice() {
        cur = 0;
        timePrice = new HashMap<>();
        priceCount = new TreeMap<>();
    }
    
    public void update(int timestamp, int price) {
        cur = Math.max(cur, timestamp);
        if (timePrice.containsKey(timestamp)) {
            int oldPrice = timePrice.get(timestamp);
            priceCount.put(oldPrice, priceCount.get(oldPrice) - 1);

            if (priceCount.get(oldPrice) == 0) {
                priceCount.remove(oldPrice);
            }
        }
        timePrice.put(timestamp, price);
        priceCount.put(price, priceCount.getOrDefault(price, 0) + 1);
    }
    
    public int current() {
        return timePrice.get(cur);
    }
    
    public int maximum() {
        return priceCount.lastKey();
    }
    
    public int minimum() {
        return priceCount.firstKey();
    }
}
```

双Heap法
```py
class StockPrice:

    def __init__(self):
        """
        need to record the price at different timestamps: hashmap
        need to get the lowest and highest and latest price: minHeap and maxHeap
        hashmap: {timestamp : price}
        
        """
        self.time_price = {}
        self.cur = 0
        self.minHeap = []
        self.maxHeap = []

    def update(self, timestamp: int, price: int) -> None:
        """
        Time: O(NlogN) each call for heappush is O(logN)
        Space: O(N) each call is O(1)
        """
        self.cur = max(self.cur, timestamp)
        self.time_price[timestamp] = price
        heapq.heappush(self.minHeap, (price, timestamp))
        heapq.heappush(self.maxHeap, (-price, timestamp))
        
    def current(self) -> int:
        """
        Time: O(1)
        Space: O(1)
        """
        return self.time_price[self.cur]
        
    def maximum(self) -> int:
        """
        check if the maxHeap[0] is in time_price, if not, means that had been updated, thus keeping popping until find the valid one.
        Time: O(NlogN)
        Space: O(1)
        """
        price, time = self.maxHeap[0]
        while -price != self.time_price[time]:
            heapq.heappop(self.maxHeap)
            price, time = self.maxHeap[0]
        return -price
        
    def minimum(self) -> int:
        """
        similar to maximum()
        Time: O(NlogN)
        Space: O(1)
        """
        price, time = self.minHeap[0]
        while price != self.time_price[time]:
            heapq.heappop(self.minHeap)
            price, time = self.minHeap[0]
            
        return price
```


[1146. Snapshot Array](https://leetcode.com/problems/snapshot-array/)

```py
class SnapshotArray:

    def __init__(self, length: int):
        """
        BF: use a lot of memory to store snap; also take times to copy arr to do snap; quick to access
        array
        Dict: {id: array}, id is len(dict)
        get: dict[id][idx]
        
        """
        self.arr = [0] * length
        self.id_arr = {}

    def set(self, index: int, val: int) -> None:
        self.arr[index] = val

    def snap(self) -> int:
        id = len(self.id_arr)
        self.id_arr[id] = self.arr.copy()
        return id

    def get(self, index: int, snap_id: int) -> int:
        return self.id_arr[snap_id][index]
```

instead of record the whole array, we record the history of each cell -> minimum space to record all information
for each arr[i], record the history with a snap_id and value.
when call the get(), do binary search the time snap_id

```py
class SnapshotArray:
    def __init__(self, length: int):
        self.map = defaultdict(list)
        self.snapId = 0

    def set(self, index: int, val: int) -> None:
        if self.map[index] and self.map[index][-1][0] == self.snapId:
            self.map[index][-1][1] = val
            return
        self.map[index].append([self.snapId, val])

    def snap(self) -> int:
        self.snapId += 1
        return self.snapId - 1

    def get(self, index: int, snap_id: int) -> int:
        """
        find the right most valid
        """
        arr = self.map[index]
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid][0] <= snap_id:
                left = mid + 1
            else:
                right = mid - 1
        return 0 if right < 0 else arr[right][1]
```

[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

```py
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        
        for c in tokens:
            if c in "+-*/":
                a, b = stack.pop(), stack.pop()
                if c == "+":
                    stack.append(a + b)
                elif c == "-":
                    stack.append(b - a)
                elif c == "*":
                    stack.append(a * b)
                else:
                    stack.append(int(b / a))
            else:
                stack.append(int(c))  
        
        return stack[0]
```

[843. Guess the Word](https://leetcode.com/problems/guess-the-word/)

```py
class Solution:
    def getMatch(self,word1, word2):
        count = 0
        for x,y in zip(word1,word2):
            if x == y:
                count +=1
        return count
                
    def findSecretWord(self, wordlist: List[str], master: 'Master') -> None:
        """
        每次.guess(word)之后，缩小wordlist的范围，只把matches与word相同的保留下来

        Time: O(N)
        Space: O(N)
        """
        i = 0
        matches = 0
        while i < 10 and matches != 6:
            index = random.randint(0,len(wordlist)-1)
            word = wordlist[index]
            matches = master.guess(word)
            candidates = []
            for w in wordlist:
                if matches == self.getMatch(word,w):
                    candidates.append(w)
            wordlist = candidates
        return word
```

[1937. Maximum Number of Points with Cost](https://leetcode.com/problems/maximum-number-of-points-with-cost/)
类似LC931的解法
```py
class Solution:
    def maxPoints(self, P: List[List[int]]) -> int:
            m, n = len(P), len(P[0])
            if m == 1: return max(P[0])
            if n == 1: return sum(sum(x) for x in P)

            def left(arr):
                lft = [arr[0]] + [0] * (n - 1)
                for i in range(1, n): lft[i] = max(lft[i - 1] - 1, arr[i])
                return lft

            def right(arr):
                rgt = [0] * (n - 1) + [arr[-1]]
                for i in range(n - 2, -1, -1): rgt[i] = max(rgt[i + 1] - 1, arr[i])
                return rgt

            pre = P[0]
            for i in range(m - 1):
                lft, rgt, cur = left(pre), right(pre), [0] * n
                for j in range(n):
                    cur[j] = P[i + 1][j] + max(lft[j], rgt[j])
                pre = cur[:]

            return max(pre)
```

类似LC121和1014的解法
```py
class Solution:
    def maxPoints(self, A: List[List[int]]) -> int:
        m, n = len(A), len(A[0])
        for i in range(m - 1):
            for j in range(n - 2, -1, -1):
                A[i][j] = max(A[i][j], A[i][j + 1] - 1)
            for j in range(n):
                A[i][j] = max(A[i][j], A[i][j - 1] - 1 if j else 0)
                A[i + 1][j] += A[i][j]
        return max(A[-1])
```

[1014. Best Sightseeing Pair](https://leetcode.com/problems/best-sightseeing-pair/)

```py
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        res = 0
        max_so_far = 0
        for i in range(len(values)):
            res = max(res, max_so_far + values[i] - i)
            max_so_far = max(max_so_far, values[i] + i)
        
        return res
```

[1610. Maximum Number of Visible Points](https://leetcode.com/problems/maximum-number-of-visible-points/)
```py
class Solution:
    def visiblePoints(self, points: List[List[int]], angle: int, location: List[int]) -> int:
        """
        convert all coordinates to radians
        sort the array
        use sliding window to find the longest window that satisfies arr[r] - arr[l] <= angle.
        
        Time: O(NlogN)
        Space: O(N)
        """
        if not points or len(points) == 0:
            return 0
        
        pointsAtLocationCount = 0
        pointsAngles = []
        
        for x, y in points:
            dx = x - location[0]
            dy = y - location[1]
            if dx == 0 and dy == 0:
                pointsAtLocationCount += 1
            else:
                radAngle = math.atan2(dy, dx)
                degAngle = math.degrees(radAngle)
                pointsAngles.append(degAngle)
        
        pointsAngles = sorted(pointsAngles)

        # Add the additional points to make circular array to handle and points start from the second half
        pointsAngles += [i+360 for i in pointsAngles]
        
        l = 0
        res = 0
        for r in range(len(pointsAngles)):
            while(pointsAngles[r] - pointsAngles[l] > angle):
                l += 1
            res = max(res, r - l + 1)

        # Add the points those are at the location and return.
        return res + pointsAtLocationCount
```

[1048. Longest String Chain](https://leetcode.com/problems/longest-string-chain/)
```py
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        dp = {}
        result = 1

        for word in sorted(words, key=len):
            dp[word] = 1

            for i in range(len(word)):
                prev = word[:i] + word[i + 1:]

                if prev in dp:
                    dp[word] = max(dp[prev] + 1, dp[word])
                    result = max(result, dp[word])

        return result
```
```py
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        """
        map: {ending word: length of the longest sequence ending with the word}
        res = max(map.values())
        
        Time: O(L^2 * N)
        Space: O(N)
        """
        memo = {}
        visited = set(words)
        
        def dfs(word):
            if word in memo:
                return memo[word]
            
            memo[word] = 1
            for i in range(len(word)):
                prev = word[:i] + word[i + 1:] # create all possible words
                if prev in visited: # if in the wordlist, perform dfs, and update the memo[word]
                    memo[word] = max(memo[word], dfs(prev) + 1)
            return memo[word] # return it
    
        res = 0
        for word in words:
            res = max(res, dfs(word)) # dfs each word
        return res
```

[539. Minimum Time Difference](https://leetcode.com/problems/minimum-time-difference/)

```py
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        """
        convert to minutes, sort, and then iterate through to find the minimum difference.
        to compare the last one and the first one: add (times[0] + 24 * 60) to the end of times
        """
        times = []
        for t in timePoints:
            minute = int(t[:2]) * 60 + int(t[-2:])
            times.append(minute)
        times.sort()
        print(times)
        times.append(times[0] + 24 * 60) # calc the final point with the first point
        
        res = float("inf")
        for i in range(1, len(times)):
            res = min(res, times[i] - times[i - 1])
        
        return res
```

[833. Find And Replace in String](https://leetcode.com/problems/find-and-replace-in-string/)
```py
class Solution:
    def findReplaceString(self, s: str, indices: List[int], sources: List[str], targets: List[str]) -> str:
        """
        convert to list
        when the src meet the same, change it to tar. after that, delete the original char in the res
        
        Time: O(N)
        Space: O(N)
        """
        res = list(s)
        for idx, src, tar in zip(indices, sources, targets):
            if s[idx: idx + len(src)] == src:
                res[idx] = tar
                for j in range(idx + 1, idx + len(src)): # delete the original char in the res
                    res[j] = ""
        return "".join(res)

```

[2178. Maximum Split of Positive Even Integers](https://leetcode.com/problems/maximum-split-of-positive-even-integers/)

```python
class Solution:
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        """
        从2，4，6开始加同时finalSum-=246，直到curRes>finalSum，这个时候就res[-1]+=finalSum就可以了
        """
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

[2135. Count Words Obtained After Adding a Letter](https://leetcode.com/problems/count-words-obtained-after-adding-a-letter/)

```py
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False 
        
class Solution:
    def __init__(self):
        self.root = TrieNode()
    
    def add(self, word):
        curt = self.root
        for c in word:
            if c not in curt.children:
                curt.children[c] = TrieNode()
            
            curt = curt.children[c]
        curt.end = True
        
    def find(self, word):
        curt = self.root
        for c in word:
            if c not in curt.children:
                return False 
            curt = curt.children[c]
        return curt.end
        
        
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        for word in startWords:
            self.add(sorted(list(word)))
            
        res = 0
        for word in targetWords:
            target = sorted(list(word))
            for i in range(len(target)):
                w = target[:i] + target[i+1:]
                if self.find(w):
                    res += 1
                    break
        return res
```

```py
class Solution:
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        """
        add bitmasks of start word to the hash set
        obtaion bitmask of a word: map each letter to a power of 2 number: {a:1, b:2, c:4, d:8...}, then for all letters in the word, we take the sum
        for a target word, remove one of the char, and check if in the hash set
        bitmask(abc) == bitmask(bca)

        Time: O(N)
        Space: O(N)
        """
        seen = set()
        for word in startWords: # bitmasks all word in startWord
            m = 0
            for ch in word:
                # m ^= 1 << ord(ch)-ord("a")  
                m ^= 2 ** (ord(ch)-ord("a"))
            seen.add(m)
            
        ans = 0 
        for word in targetWords: 
            m = 0 
            for ch in word: # bitmask the target word
                m ^= 2 ** (ord(ch)-ord("a"))
            for ch in word: # remove one char
                if m ^ (2 ** (ord(ch)-ord("a"))) in seen: 
                    ans += 1
                    break 
        return ans 
```



[2158. Amount of New Area Painted Each Day](https://leetcode.com/problems/amount-of-new-area-painted-each-day/)
```py
class Solution:
    def amountPainted(self, paint: List[List[int]]) -> List[int]:
        # constructure the sweep line
        records = []
        max_pos = 0
        for i, [start, end] in enumerate(paint):
            records.append((start, i, 1)) # use 1 and -1 to records the type.
            records.append((end, i, -1))
            max_pos = max(max_pos, end)
        records.sort()

        # sweep across all position
        ans = [0 for _ in range(len(paint))]
        indexes = []
        ended_set = set()
        i = 0
        for pos in range(max_pos + 1):
            while i < len(records) and records[i][0] == pos:
                pos, index, type = records[i]
                if type == 1:
                    heapq.heappush(indexes, index)
                else:
                    ended_set.add(index)
                i += 1
            
            while indexes and indexes[0] in ended_set:
                heapq.heappop(indexes)

            if indexes:
                ans[indexes[0]] += 1
        return ans
```

```py
class Solution:
    def amountPainted(self, paint: List[List[int]]) -> List[int]:
        """
        For each interval traverse start to end marking jump value as end.
        if any jump index is already marked (i.e. > 0) skip to the jump value, saving traversal. and continue #3 till interval is complete.
        """
        p = [0] * 50000 # 1D number array
        res = []
        for (start,end) in paint:
            cur_res = 0
            # loop from start to end of the interval
            while start < end : 
                # if jump value is set
                if p[start] != 0 : 
                    start = p[start]
                # if jump value is not set
                else :
                    cur_res += 1
                    p[start] = end
                    start += 1
           
            res.append(cur_res)
        return res
```





# VO
[745. Prefix and Suffix Search](https://leetcode.com/problems/design-add-and-search-words-data-structure/)