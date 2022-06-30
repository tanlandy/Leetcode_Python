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

```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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
when call the get(), do binary search tthe time snap_id


## 06/24/22 Engineering Round Table
google coursera

GTI training: Google tech immersion
could you tell me more about the self-development opportunities that google provides: technical and non-technical?
- intro to technical writing


three skills that you think are most critical to success in your role at Google?
1. ask for help and be open
2. express your idea in any way you fell comfortable with
3. find your group, find allies with same interests or share the same value to find support and inclusive

what was something that surprised you about Google:
1. scale
2. google code search
3. how easy to reach anyone

how did you prepare for your interviews
1. leedcode
2. youtube video into different categories
3. mock interviews to get feedback
4. Communication!!!


common mistakes you see candidates make during the interviews
1. communication is key
2. not stick to a single algo, but to analyze the problem and try different approaches
3. listen to their hints, pay attention to hints
4. don't give up, try hard, come up with the correct code for the end minute

ckech the prep email, google suite interview perp guide