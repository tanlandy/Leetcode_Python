# Educative基础知识

## Tree
### Terminology
1. Depth of Node: depth of root node is 0
2. Height of Node: 
   - height of leaf node is 0
   - height of a tree = height of the root node
3. Complete Binary Tree
   - every level except possibly the last, is completed filled. 
   - all nodes in the last level are as far left as possible
4. Full Binary Tree: every node has either 0 or 2 children
5. Binary Search Trees
   - The value of the left child of any node in a binary search tree will be less than whatever value we have in that node, and the value of the right child of a node will be greater than the value in that node.
   - AVL tree: a kind of BST, that rebalances the nodes so that we won't get a linear BST

| BST | Average | Worst |
| --- | --- | --- | 
| Search | O(logN) | O(N) |
| Insert | O(logN) | O(N) |
| Delete | O(logN) | O(N) |

implementation in python
```py
class Node(object):
  def __init__(self, value):
    self.value = value
    self.left = None
    self.right = None
    
class BinaryTree(object):
  def __init__(self, root):
    self.root = Node(root)


tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)
tree.root.right.left = Node(6)
tree.root.right.right = Node(7)
```

## Graph

def: A graph is a data structure consists of finite set of vertecies and finite set of edges

| application     | vertices | edges             |
|:--------------- |:-------- |:----------------- |
| maps            | 交叉口      | roads             |
| social networks | people   | friendship status |
| contact tracing | people   | with people       |

G = (V, E)
dense graph: E ~ $V^2$ -> interactions at family dinner
sparse graph: E ~ V -> facebooks
important to think about the type of graph

3 variants of graph

1. undirected: (u, v) = (v, u) -> facebook -> V = {1,2,3}, E = {(1,2)(2,1)(1,3)(3,1)(2,3)(3,2)}
2. directed: (u, v) from u to v -> instagram
3. weighted: 

def of adjacency: given vertex u, and vertex v, vertex u is adjacent to v iff (u, v) in E

store graph: depends on the basic desired operations on the graph: search, edit, calculation, time, space

| type             | general space | dense: E=$V^2$ | sparse: E=V | Does an edge (i,j) exist? |
| ---------------- | ------------- | -------------- | ----------- | ------------------------- |
| adjacency list   | V + E         | V + $V^2$      | V + V ✅     | O(V): Read entire adj[i]  |
| adjacency matrix | $V^2$         | $V^2$ ✅        | $V^2$       | O(1)                      |

Adjacency list: 
build: for every vertex v, adj[v] = {vertecies adj to v}; space: V+E for both undirected and directed graph. but could be written as 2E for undirected graph when consider graph to be unique

Adjacency matrix:
matrix be a V*V matrix, aij = 1 iff (i, j) in E; space: $V^2$

## Path

a path in G(V, E) from u to v is a sequence of vertecies from u to v
v and v are connected if there is a path

## Connected component

C is a connected component in G(V, E) iff C 里面的点 are connected & C不和外面的点 connected

## BFS

input: G(V, E), and source vertex s
output: d[v] btw source s and all other nodes in graph
queue.append(x) // put x as last element in queue
queue.popleft() // return and remove the first element in queue

basic idea:

1. discover all vertices at distance k from s before discovering vertices at distance k+1 from s
2. expand a fronter greedly one edge distance at a time

```shell
BFS(G, s): 
for each v in (V - s): // if they are not connected, the distance is infinity
 d[v] = float("inf")
 d[s] = 0
 queue = empty
 queue.append(s)
 while queue is not empty:
 u = queue.popleft()
 for each v in adj[u]:
 if d[v] = inf:
 d[v] = d[u] + 1
 queue.append(v)
```

Runtime: each node enter queue only once: O(V), each edge only checked once: O(V+E)
Space: graph space: Adjlist: O(V+E)
queue: O(V)
distance array: O(V)

Shortest Path Problem

> given a graph, want to find the path btw u to w with min cost
> 
> well-defined shorted path:
> 
> 1. negative weight edges are OK
> 
> 2. negative cycles are not

Input: Directed weighted graph G(V, E, W)

Output: W(p), where the weight is minimum. If there's no path: inf

1. single source shortest path

2. single destination

3. single pair: single source and single destination -> SFO to JFK

4. All pairs:
   
   1. Run BFS on every node: $O(V(V+E)) -> V^2 + VE$

### Single source shortest path(SSSP)

1. unweighted/weight of all edges is 1 -> BFS

2. negative/positive all reals -> Bellman-ford

```shell
for every vertex in V: => O(V)
    d[v] = inf
    pre[v] = null
d[s] = 0
# relaxing that try to go from s to v
d[v] = estimate so far how to get to v
if d[u] + w(u, v) < d[v]:
    d[v] = d[u] + w(u, v)
```

## Dijkstra

two version: Priority queue & array

|             | PQ      | Array                 |
| ----------- | ------- | --------------------- |
| extract min | O(logn) | O(n)                  |
| update-key  | O(logn) | O(1), if have the idx |

### PQ

```shell
Dijkstra_algo(G(V, E, W), s) { -> return d[] pre[c]
    for v in V:
        d[v] = inf
        pre[v] = null
    d[s] = 0
    minHeap = V # PQ with d[v] as key -> O(V) in total from line0
    while minHeap: # -> O(V+E)
        u = minHeap.popleft() # finalize the distance to vertex u->VO(logV)
        for v in adj[u]:
            if d[v] > d[u] + w(u, v):
                d[v] = d[u] + w(u, v) # update key: EO(logV)
                pre[v] = u
}
```

In total: O(VlogV + ElogV) -> O(ElogV)

### Array

```shell
Dijkstra_algo(G(V, E, W), s) { -> return d[] pre[c]
    for v in V:
        d[v] = inf
        pre[v] = null
    d[s] = 0
    vertex = V # PQ with d[v] as key -> O(V)
    while vertex: # O(V+E)
        u = min(vertex) # finalize the distance to vertex u->VO(V)
        for v in adj[u]:
            if d[v] > d[u] + w(u, v):
                d[v] = d[u] + w(u, v) # EO(1)
                pre[v] = u
}
```

In total: $V+E+V^2+E -> O(V^2)$

|         | V(extract_min) | E(update_keys) | Total | Sparse(E=V) | Dense(E=$V^2$) |
| ------- | -------------- | -------------- | ----- | ----------- | -------------- |
| PQ      | VlogV          | ElogV          | ElogV | VlogV       | V^2logV        |
| Array   | V*V            | E              | $V^2$ | $V^2$       | $V^2$          |
| Bellman |                |                | EV    | $V^2$       | $V^3$          |

## All pairs shortest path

input: G(V, E, W)

output: min weight of path btw every two nodes

|                 | solution          | Time                                   | Dense($E=V^2$)                      |
| --------------- | ----------------- | -------------------------------------- | ----------------------------------- |
| no edge weight  | BFS on every node | V*O(BFS)=V(V+E)                        | $V^3$                               |
| all are positve | Dijkstra          | minHeap:V*(ElogV)<br/>Array: V*($V^2$) | minHeap: $V^3logV$<br/>Array: $V^3$ |
| have negative   | Bellman           | V*O(VE) = $V^2E$                       | $V^4$                               |
| any pos/neg     | Floyd-warshall    | $O(V^3)$                               | $V^3$                               |

[1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance]([Loading...](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/))







# Tree

## 模板

```py
def BFS(root):
    if not root:
        return root

    queue = collections.deque([root]) # initate queue and add root
    visit = set()  # use a set to keep track of visited node, no need for traversing a tree
    visit.add((root))
    step = 0 # depends on the target

    while queue:
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            if node is target: # depends on the target
                return
            for nei in node.adj(): # traverse the graph or the tree
                if nei not in visit: # no cycle
                    queue.append(nei)
                    visit.add(nei)
        step += 1
```

[111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

```py
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue = collections.deque([root])
        step = 1

        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()

                # case that meet the target
                if not node.left and not node.right:
                    return step

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            # after the level, update the step
            step += 1

        return step
```

```py
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        """
        完全理解BFS的实现过程的好题
        """
        if not root:
            return root

        queue = collections.deque([root])

        while queue:
            size = len(queue)

            for i in range(size):
                node = queue.popleft()

                if i < size - 1: # 点睛之笔，每次要链接的，其实就是把刚popleft出来的连接到[0]
                    node.next = queue[0]

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return root
```

[752. Open the Lock](https://leetcode.com/problems/open-the-lock/)

```py
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        """
        BFS templete, trick point is to find the adjcent wheels each time
        """
        # edge case
        if "0000" in deadends:
            return -1

        # find the adjacent locks
        def children(wheel):
            res = []
            for i in range(4): # 8 adjcents in toal
                digit = str((int(wheel[i]) + 1) % 10) # up 1
                res.append(wheel[:i] + digit + wheel[i+1:])
                digit = str((int(wheel[i]) + 10 - 1) % 10) # down 1
                res.append(wheel[:i] + digit + wheel[i+1:])
            return res    

        # BFS structrue
        queue = collections.deque()
        visit = set(deadends)
        queue.append(["0000", 0]) # queue stores both [wheel, turns]
        while queue:
            wheel, turns = queue.popleft()
            # target to return
            if wheel == target:
                return turns

            # traverse other
            for child in children(wheel):
                if child not in visit:
                    visit.add(child)
                    queue.append([child, turns + 1])
        return -1 
```

# 2D Grid

[994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)

```py
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        """
        Multi-source BFS 
        需要同时从每个点来看下一个点什么时候rotten
        通过比较最开始的fresh和最后queue之后剩余的fresh个数，来确定是否有剩余

        时间：O(M*N)
        空间：O(M*N)
        """

        queue = collections.deque()
        time, fresh = 0, 0

        rows = len(grid)
        cols = len(grid[0])

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    fresh += 1
                if grid[r][c] == 2:
                    queue.append([r, c])

        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        while queue and fresh > 0:
            for i in range(len(queue)):
                r, c = queue.popleft()

                for dr, dc in dirs:
                    row, col = r + dr, c + dc
                    # if in bounds and fresh, make it rotten
                    if row in range(rows) and col in range(cols) and grid[row][col] == 1:
                        grid[row][col] = 2
                        queue.append((row, col))
                        fresh -= 1

            time += 1

        return time if fresh == 0 else -1
```

# 拓扑排序

假设L是存放结果的列表，
Step1: 找到那些入度为零的节点，把这些节点放到L中。
Step2: 因为这些节点没有任何的父节点。所以可以把与这些节点相连的边从图中去掉。
Step3: 再次寻找图中的入度为零的节点。对于新找到的这些入度为零的节点来说，他们的父节点已经都在L中了，所以也可以放入L。
Step4: 重复上述操作，直到找不到入度为零的节点。
Step5: 如果此时L中的元素个数和节点总数相同，说明排序完成；如果L中的元素个数和节点总数不同，说明原图中存在环，无法进行拓扑排序。

```code
L ← 包含已排序的元素的列表，目前为空
S ← 入度为零的节点的集合
当 S 非空时：
    将节点n从S移走
    将n加到L尾部
    选出任意起点为n的边e = (n,m)，移除e。如m没有其它入边，则将m加入S。
    重复上一步。
如图中有剩余的边则：
    return error   (图中至少有一个环)
否则： 
    return L   (L为图的拓扑排序)
```

[207. Course Schedule](https://leetcode.com/problems/course-schedule/)

```py
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Algorithm: BFS Topological Sorting
        Time: O(E + V)
        Space: O(E + V)
        """
        # step1: build graph and store 入边的数量
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for to_, from_ in prerequisites:
            graph[from_].append(to_)
            indegree[to_] += 1

        # step2: 把入边为0的点加到queue中
        queue = collections.deque()
        for v in range(numCourses):
            if indegree[v] == 0:
                queue.append(v)

        # step3: 进行拓扑排序
        count = 0 # 记录走过的数量
        order = [] # 可以用来记录topo_order
        while queue:
            v = queue.popleft()
            count += 1
            order.append(v)
            for to_ in graph[v]:
                indegree[to_] -= 1
                if indegree[to_] == 0:
                    queue.append(to_)

        print(order) # 打印出来排序的结果
        return count == numCourses # 根据是否记录的数量等于总课程数量
```

[210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)

```py
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        和LC207相同，只是在拓扑排序的时候把结果同时存一下即可
        """
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for to_, from_ in prerequisites:
            graph[from_].append(to_)
            indegree[to_] += 1

        # step2: 把入边为0的点加到queue中
        queue = collections.deque()
        for v in range(numCourses):
            if indegree[v] == 0:
                queue.append(v)

        # step3: 进行拓扑排序
        count = 0 # 记录走过的数量
        order = [] # 可以用来记录topo_order
        while queue:
            v = queue.popleft()
            count += 1
            order.append(v)
            for to_ in graph[v]:
                indegree[to_] -= 1
                if indegree[to_] == 0:
                    queue.append(to_)

        return order if count == numCourses else [] # 根据是否记录的数量等于总课程数量
```

[261. Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)

```py
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
    """
    构建adjList

    """        
```

# Dijkstra
