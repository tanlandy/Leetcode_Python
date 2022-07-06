# 基础知识

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


## BFS

input: G(V, E), and source vertex s
output: d[v] btw source s and all other nodes in graph
queue.append(x) // put x as last element in queue
queue.popleft() // return and remove the first element in queue

basic idea:

1. discover all vertices at distance k from s before discovering vertices at distance k+1 from s
2. expand a fronter greedly one edge distance at a time

Since the difference between a tree and a graph is the possibility of having a cycle, we just have to handle this situation. We use an extra visited variable to keep track of vertices we have already visited to prevent re-visiting and getting into infinite loops


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


## Single source shortest path(SSSP)

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

## BFS vs DFS
BFS:
1. find the shortest distance
2. graph of unknown size (word ladder), or infinite size (knight shortest path)

DFS:
1. less memory. as BFS has to keep all the nodes in the queue for wide graph
2. find nodes far away from the root, eg looking for an exit in a maze



## 模版
### BFS
In an adjacency list representation, this would be returning the list of neighbors for the node. 
If the problem is about a matrix, this would be the surrounding valid cells as we will see in number of islands and knight shortest path. 
If the graph is implicit, we have to generate the neighbors as we traverse. We will see this in word ladder.
```python
from collections import deque

def bfs(root):
    queue = deque([root])
    visited = set([root])
    while len(queue) > 0:
        node = queue.popleft()
        for neighbor in get_neighbors(node):
            if neighbor in visited:
                continue
            queue.append(neighbor)
            visited.add(neighbor)
```
使用：shortest path, graph of unknown or even infinite size

### DFS
```py
def dfs(root, visited):
    if not root:
        return 
    for neighbor in get_neighbors(root):
        if neighbor in visited:
            continue
        visited.add(neighbor)
        dfs(neighbor, visited)
```
# 2D Grid
## 模版

### BFS
shortest path btw A and B
```py
def shortest_path(graph: List[List[int]], a: int, b: int) -> int:
    queue = collections.deque([a])
    visited = set([a])
    res = 0
    while queue:

        for _ in range(len(queue)):
            node = queue.popleft()
            if node == b:
                return res   
            for nei in graph[node]:
                if nei not in visited:
                    queue.append(nei)
                    visited.add(nei)
        res += 1
    
    return -1
```
一般情况下的BFS
```py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Time：O(M*N)
        Space：O(min(M,N))
        """
        rows, cols = len(grid), len(grid[0])
        visited = set()
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
       
        # BFS方法
        def bfs(r, c):
            queue = collections.deque([(r, c)])
            visited.add((r, c))

            while queue:
                r, c = queue.popleft()

                for dx, dy in dirs:
                    nei_r, nei_c = r + dx, c + dy
                    if 0 <= nei_r < rows and 0 <= nei_c < cols and (nei_r, nei_c) not in visited and grid[nei_r][nei_c] == "1":
                        queue.append((nei_r, nei_c))
                        visited.add((nei_r, nei_c))
        
        # 调用DFS的时机
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visited:
                    count += 1
                    bfs(r, c)
        
        return count
```

### DFS
一般情况下的DFS
```py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Time：O(M*N)
        Space：O(M*N)
        """
        rows, cols = len(grid), len(grid[0])
        visit = set()
        
        # DFS方法
        def dfs(r, c):
            if (r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or grid[r][c] != "1"):
                return
            
            visit.add((r, c)) # can also flood fill to "0"
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
            
        # 调用DFS的时机
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visit:
                    count += 1
                    dfs(r, c)      
        return count

```
## 例题

[733. Flood Fill](https://leetcode.com/problems/flood-fill/)

```py
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        """
        最基本的BFS
        """
        rows, cols = len(image), len(image[0])   
        queue = collections.deque([(sr, sc)])
        visited = set([(sr, sc)])
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while queue:
            r, c = queue.popleft()
            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                if 0 <= nei_r < rows and 0 <= nei_c < cols and (nei_r, nei_c) not in visited and image[nei_r][nei_c] == image[sr][sc]:
                    image[nei_r][nei_c] = newColor
                    queue.append((nei_r, nei_c))
                    visited.add((nei_r, nei_c))
        image[sr][sc] = newColor
        return image
```

```py
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        基本的DFS
        """
        rows, cols = len(image), len(image[0])
        color = image[sr][sc]
        
        def dfs(r, c):
            if 0 <= r < rows and 0 <= c < cols and image[r][c] == color:
                image[r][c] = newColor
                dfs(r-1, c)
                dfs(r+1, c)
                dfs(r, c-1)
                dfs(r, c+1)
        
        if color == newColor:
            return image
        dfs(sr, sc)
        return image
```

[200. Number of Islands](https://leetcode.com/  problems/number-of-islands/)
BFS
```py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Time：O(M*N)
        Space：O(min(M,N))
        """
        rows, cols = len(grid), len(grid[0])
        visited = set()
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
       
        def bfs(r, c):
            queue = collections.deque([(r, c)])
            visited.add((r, c))

            while queue:
                r, c = queue.popleft()

                for dx, dy in dirs:
                    nei_r, nei_c = r + dx, c + dy
                    if 0 <= nei_r < rows and 0 <= nei_c < cols and (nei_r, nei_c) not in visited and grid[nei_r][nei_c] == "1":
                        queue.append((nei_r, nei_c))
                        visited.add((nei_r, nei_c))
        
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visited:
                    count += 1
                    bfs(r, c)
        
        return count
```

DFS
```py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Time：O(M*N)
        Space：O(M*N)
        """
        rows, cols = len(grid), len(grid[0])
        visit = set()
        
        def dfs(r, c):
            if (r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or grid[r][c] != "1"):
                return
            
            visit.add((r, c)) # can also flood fill to "0"
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
        
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visit:
                    count += 1
                    dfs(r, c)      
        return count
```

[1197. Minimum Knight Moves](https://leetcode.com/problems/minimum-knight-moves/)

```py
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        """
        BFS one-direction
        """
        
        queue = collections.deque([(0, 0, 0)])
        x, y, visited = abs(x), abs(y), set([(0, 0)])
        dirs = [(1, 2), (2, 1), (1, -2), (-2, 1), (-1, 2), (2, -1)]
        while queue:
            a, b, step = queue.popleft()
            if (a, b) == (x, y):
                return step
            
            for dx, dy in dirs:
                nei_x, nei_y = a + dx, b + dy
                if (nei_x, nei_y) not in visited and -1 <= nei_x <= x + 2 and -1 <= nei_y <= y + 2:
                    visited.add((nei_x, nei_y))
                    queue.append((nei_x, nei_y, step + 1))
            
        return -1

```

```py
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        """
        BFS two-direction: Start BFS from both origin and target position
        
        Time: O(|x|*|y|)
        Space: O(|x|*|y|)
        """
        x, y = abs(x), abs(y)
        queue_ori = collections.deque([(0, 0, 0)])
        queue_tar = collections.deque([(x, y, 0)])
        # use two dicts to map the position to step
        d_ori, d_tar = {(0, 0): 0}, {(x, y): 0}
        dirs = [(1, 2), (2, 1), (1, -2), (-2, 1), (-1, 2), (2, -1), (-1, -2), (-2, -1)]
        
        while True:
            # if already visited in the other dict: return 
            ox, oy, ostep = queue_ori.popleft()
            if (ox, oy) in d_tar:
                return ostep + d_tar[(ox, oy)]
            tx, ty, tstep = queue_tar.popleft()
            if (tx, ty) in d_ori:
                return tstep + d_ori[(tx, ty)]
            
            # visit new nodes, add it to queue and dict
            for dx, dy in dirs:
                nei_ox, nei_oy = ox + dx, oy + dy
                if (nei_ox, nei_oy) not in d_ori and -1 <= nei_ox <= x + 2 and -1 <= nei_oy <= y + 2:
                    queue_ori.append((nei_ox, nei_oy, ostep + 1))
                    d_ori[(nei_ox, nei_oy)] = ostep + 1
                    
                nei_tx, nei_ty = tx + dx, ty + dy
                if (nei_tx, nei_ty) not in d_tar and -1 <= nei_tx <= x + 2 and -1 <= nei_ty <= y + 2:
                    queue_tar.append((nei_tx, nei_ty, tstep + 1))
                    d_tar[(nei_tx, nei_ty)] = tstep + 1
        
        return -1
```

[286. Walls and Gates](https://leetcode.com/problems/walls-and-gates/)

```py
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        找每个格子到gate的最短距离：从gate出发bfs
        """
        rows, cols = len(rooms), len(rooms[0])
        queue = collections.deque()
        visit = set()
        
        def addRoom(r, c):
            if (r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or rooms[r][c] == -1):
                return
            queue.append((r, c))
            visit.add((r, c))
        
        # 先把gates都添加到queue中，然后同时的bfs
        for r in range(rows):
            for c in range(cols):
                if rooms[r][c] == 0:
                    queue.append((r, c))
                    visit.add((r, c))
                    
        dist = 0
        while queue:
            size = len(queue)
            for i in range(size):
                r, c = queue.popleft()
                rooms[r][c] = dist
                addRoom(r + 1, c)
                addRoom(r - 1, c)
                addRoom(r, c + 1)
                addRoom(r, c - 1)
            dist += 1
```



# Implicit Graph

[752. Open the Lock](https://leetcode.com/problems/open-the-lock/)

```py
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        """
        BFS templete, trick point is to find the adjcent wheels each time

        Time: O(N^2 * A^N + D) N is number of digtis, A is number of alphabets: O(N^2) for each combination, we spend
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

[773. Sliding Puzzle](https://leetcode.com/problems/sliding-puzzle/)
```py
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        """
        BFS上下左右移动空格，和对应的格子交换
        """     
        rows = 2
        cols = 3
        def find_zero(board):
            for i in range(rows):
                for j in range(cols):
                    if board[i][j] == 0:
                        return i, j
        
        steps = 0
        queue = collections.deque([board])
        visited = set([str(board)]) # list is unhashable, so converted to string

        target = [[1, 2, 3], [4, 5, 0]]

        while queue:
            for _ in range(len(queue)):
                board = queue.popleft()
                i, j = find_zero(board)
                if board == target:
                    return steps
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for nei in neighbors:
                    new_board = [row[:] for row in board]
                    r, c = nei
                    if 0 <= r < rows and 0 <= c < cols:
                        new_board[r][c], new_board[i][j] = new_board[i][j], new_board[r][c]
                        if str(new_board) not in visited:
                            queue.append(new_board)
                            visited.add(str(new_board))
            steps += 1
        
        return -1

```

```py
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        # Define the function for any size
        # @params: m, n are the size of the board, final is the final state of the board
        # @return: least number of moves (or -1 if there's no result)
        def sliding_puzzle_any_size(board, m, n, final):
            def find_zero(board):
                for i in range(n):
                    for j in range(m):
                        if board[i][j] == 0:
                            return i, j

            moves_so_far = 0
            queue = collections.deque([board])
            visited = set([str(board)])

            while queue:
                
                for _ in range(len(queue)):
                    board = queue.popleft()
                    i, j = find_zero(board)
                    if board == final:
                        return moves_so_far
                    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                    for neighbor in neighbors:
                        new_board = [row[:] for row in board]
                        row, col = neighbor
                        if row < 0 or row >= n or col < 0 or col >= m:
                            continue
                        new_board[row][col], new_board[i][j] = new_board[i][j], new_board[row][col]
                        if str(new_board) not in visited:
                            queue.append(new_board)
                            visited.add(str(new_board))

                moves_so_far += 1
            return -1

        # Define the final state and pass it with m=3 n=2 as arguments to the function
        final = [[1, 2, 3],[4, 5, 0]]
        return sliding_puzzle_any_size(board, 3, 2, final)  
```

[261. Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)

```py
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        """
        构建adjList
        然后DFS，最后看visit过的是否和n相同
        dfs: 用prev来记录入边，这样就避免false positive的hasCycle

        时间：O(V+E)
        空间：O(V+E)
        """       
        if not n:
            return True
        if len(edges) != n - 1: return False
        graph = [[] for _ in range(n)]
        for n1, n2 in edges:
            graph[n1].append(n2)
            graph[n2].append(n1)
        
        seen = set()

        def dfs(node):
            if node in seen: return
            seen.add(node)
            for neighbour in graph[node]:
                dfs(neighbour)

        dfs(0)
        return len(seen) == n     
```

# Dijkstra

## 基础知识
类似于BFS，但是要去掉while中的for循环：
对于加权图，for循环遍历帮助维护depth层数，但是在Dijkstra中层数无意义，要考虑的是路径的权重和
-> 去掉for循环，在queue中存[node, depth]

the distance increases by the weight instead of 1 -> need to visit a node more than once to guarantee minimum distance to that node

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


## 模板
```py
def shortest_path(graph: List[List[Tuple[int, int]]], a: int, b: int) -> int:

    def bfs(root: int, target: int):
        min_heap = [(0, root)]
        distances = [float('inf')] * len(graph)
        distances[root] = 0
        while len(min_heap) > 0:
            distance, node = heappop(min_heap)
            if distance > distances[node]:
                continue
            for neighbor, weight in graph[node]:
                d = distances[node] + weight
                if distances[neighbor] <= d:
                    continue
                heappush(min_heap, (d, neighbor))
                distances[neighbor] = d
        return distances[target]

    return -1 if bfs(a, b) == float('inf') else bfs(a, b)
```

## 例题

[743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)
```py
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        """
        Dijstra: find the shortest single path in weighted graph
        
        Time: O(ElogV)
        Space: O(E)
        """
        
        # build a graph: adjacency list
        graph = collections.defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w)) # {n1:[(n2, w2), (n3, w3)], n2:[], n3:[]}
            
        # Dijkstra
        min_heap = [(0, k)] # (time, node)
        visited = set()
        res = 0
        while min_heap:
            # pop and visit the node with minimum time
            t1, n1 = heapq.heappop(min_heap)
            if n1 in visited:
                continue
            res = t1
            visited.add(n1)
            
            # traverse its unvisited neighbors
            for n2, w2 in graph[n1]:
                if n2 not in visited:
                    heapq.heappush(min_heap, (t1 + w2, n2))
        
        return res if len(visited) == n else -1
```

[1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/)


[1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/)
```py
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        """
        The absolute difference between adjacent cells A and B can be perceived as the weight of an edge from cell A to cell B.
        
        Time: O(ElogV), E = 4MN, V = MN
        Space: O(MN)
        """
        rows, cols = len(heights), len(heights[0])
        
        # dist[r][c] stores max diff btw (r, c) and (0, 0)
        dist = [[float("inf")] * cols for _ in range(rows)]
        dist[0][0] = 0
        visited = set()
        minHeap = [(0, 0, 0)] # (dist, r, c)
        dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        while minHeap:
            d, r, c = heapq.heappop(minHeap)
            visited.add((r, c)) # 在这里add而不是在加入的时候add，是因为只有在现在才pop出来并且查看
            
            if r == rows - 1 and c == cols - 1:
                return d
            
            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                if 0 <= nei_r < rows and 0 <= nei_c < cols and (nei_r, nei_c) not in visited:
                    new_d = max(d, abs(heights[nei_r][nei_c] - heights[r][c])) # always try to get the larger diff, and then store it into dist[][]
                    if dist[nei_r][nei_c] > new_d:
                        dist[nei_r][nei_c] = new_d
                        heapq.heappush(minHeap, (new_d, nei_r, nei_c))
        
        return -1
```

```py
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        """
        Using binary search to find the minimum threadshold: [False, ..., True, True, True...] find the left most valid position
        
        Time: O(M*N)
        Space: O(M*N)
        """
        rows, cols = len(heights), len(heights[0])
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        def canReach(k): # check whether the maximum diff is smaller than k
            """BFS, check whether can reach to the target position within k"""
            queue = collections.deque([(0, 0)]) # (r, c)
            visited = set((0, 0))
            while queue:
                r, c = queue.popleft()
                if r == rows - 1 and c == cols - 1:
                    return True
                
                for dx, dy in dirs:
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        diff = abs(heights[nr][nc] - heights[r][c])
                        if diff <= k: # only add valid node into queue: the diff btw is smaller than k
                            queue.append((nr, nc))
                            visited.add((nr, nc))
            return False
        
        l, r = 0, 10000000
        while l <= r:
            mid = (l + r) // 2
            if canReach(mid):
                r = mid - 1
            else:
                l = mid + 1
        
        return l
```


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







# Topological Sort
## 基础知识
Topological sort or topological ordering of a directed graph is an ordering of nodes such that every node appears in the ordering before all the nodes it points to.

Topological sort is not unique

Graph with cycles do not havfe topological ordering

## 模版：Kahn's Algorithm

To obtain a tolopogical order, we can use Kahn's algorithm

假设L是存放结果的列表，
Step1: 找到那些入度为零的节点，把这些节点放到L中。initialize a hashmap of node to parent
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

和BFS的区别是，Topological sort只push入度为0的点到queue，而BFS把所有的邻居都push到queue

## 例题

[207. Course Schedule](https://leetcode.com/problems/course-schedule/)

```py
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Algorithm: BFS Topological Sorting
        
        Time: O(E + V) This is because we have to go through every connection and node once when we sort the graph.
        Space: O(E + V)
        """
        # step1: build graph and 入边的数量
        graph = {x: [] for x in range(numCourses)}
        indegree = {x: 0 for x in range(numCourses)}
        # graph = [[] for _ in range(numCourses)]
        # indegree = [0] * numCourses
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
        # graph = [[] for _ in range(numCourses)]
        # indegree = [0] * numCourses
        graph = {x: [] for x in range(numCourses)}
        indegree = {x: 0 for x in range(numCourses)}
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

[444. Sequence Reconstruction](https://leetcode.com/problems/sequence-reconstruction/)

```py
class Solution:
    def sequenceReconstruction(self, nums: List[int], sequences: List[List[int]]) -> bool:
        """
        We can try to construct a topological ordering and see if it's the same as original
        
        The key to determine uniqueness is checking the number of nodes in the queue at each step. If there is more than one node in the queue, we can pop any of them and still obtain a valid ordering and that means there will be more than one way to reconstruct the original sequence and therefore not unique.
        """
        # build graph and indgree
        graph = {x:[] for x in nums}
        indegree = {x: 0 for x in nums}
        # graph = collections.defaultdict(list) 不能用这个，因为直接忽略了x:0 x:[]的情况
        # indegree = collections.defaultdict(int)
        for seq in sequences:
            for i in range(len(seq) - 1):
                from_, to_ = seq[i], seq[i + 1]
                graph[from_].append(to_)
                indegree[to_] += 1
        
        # add valid nodes to queue
        queue = collections.deque()
        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)
        
        # check using topological sort
        res = []
        while queue:
            if len(queue) != 1:
                return False
            node = queue.popleft()
            res.append(node)
            for to_ in graph[node]:
                indegree[to_] -= 1
                if indegree[to_] == 0:
                    queue.append(to_)
        
        return len(res) == len(nums)
```


[953. Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary/)
```python
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        """
        用map存{letter: rank}，然后比较相邻的word，不符合的条件有两个：1，前面相同时len(words[i]) > len(words[i+1]；2，不同时候rank不对。如果不同但是满足，可以就直接break这两个word的比较；enumerate(string)返回(index, val)

        时间：O(M)； M is total number of char in words
        空间：O(1)
        """
        letter_order = {}
        
        for idx, val in enumerate(order):
            letter_order[val] = idx
        
        # 两两比较
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            min_len = min(len(w1), len(w2))
            # edge case：两个前缀一样，但是第一个更长
            if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                return False
            # 正常比较
            for j in range(min_len):
                if w1[j] != w2[j]:
                    if letter_order[w1[j]] > letter_order[w2[j]]:
                        return False
                    break
                    
        return True
```


[269. Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) 先做LC953

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        """
        先建adj{ch:set()}：两两word比较，得到两两letter的顺序；之后用postDFS放进来，DFS需要一个visit{ch:T/F}，每次看是否在里面，在的话就返回visit[c]，然后在dfs内部看ch的nei，如果dfs(nei)返回true就说明这条路不通，最后把res加进去；从adj的任意一个ch走dfs，最后reverse这个结果

        时间：O(M), M is number of char of words，决定了graph大小
        空间：O(1)
        """
        # adj list存储两两字母之间的order
        # 对于words里的每个w，对于w里的每个character
        # {c : set()}
        adj = {c:set() for w in words for c in w}

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            minLen = min(len(w1), len(w2))
            if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
                return ""
            for j in range(minLen):
                if w1[j] != w2[j]:
                    adj[w1[j]].add(w2[j])
                    break
        
        # DFS来遍历postorder，根据排好的顺序来画图
        # 用visit来看是否有loop->
        # visit{character: False/True} 给每个字母一个映射
        # False说明已经visit过了
        # True说明是在当前路径里
        visit = {} # False = visited, True = visited & current path
        res = []

        # post-order traversal
        def dfs(c): # 如果dfs返回True，说明这个ch已经看过并且是在当前路径，也就是cycle
            if c in visit:
                return visit[c] #如果返回true: cycle

            visit[c] = True

            for nei in adj[c]: # 看这个ch的每一个neighbor
                if dfs(nei):
                    return True
            
            visit[c] = False # 已经看过，但是不在当前路径了就
            res.append(c)
        
        for c in adj:
            if dfs(c):
                return ""
        
        res.reverse()
        return "".join(res)
```
# MST
It is a graph that connects all the vertices together, withoug cycles and with the minimum total edge weight
## Kruskal's Algo
Kruskal's algorithm generates the Minimum Spanning Tree by always choosing the smallest weigthed edge in the graph and consistently growing the tree by one edge.
1. Sort the edge besed on weights
2. Try every edge, add the edge to res as long as they are not connected -> Disjoint Sets
3. Repeat until have connected n-1 edges

Time: O(ElogE) union find is logE, do it E times. we also sort the graph

```py
class UnionFind:
    def __init__(self):
        self.id = {}

    def find(self, x):
        y = self.id.get(x, x)
        if y != x:
            self.id[x] = y = self.find(y)
        return y

    def union(self, x, y):
        self.id[self.find(x)] = self.find(y)

class Edge:
    def __init__(self, weight, a, b):
        self.weight = weight
        self.a = a
        self.b = b
def cmp():
    def compare(x, y):
        return x.weight < y.weight
    return compare
def minimum_spanning_tree(n : int, edges : List[edge]) -> int:
    # sort list, make sure to define custom comparator class cmp to sort edge based on weight from lowest to highest
    edges.sort(key = cmp)
    dsu = UnionFind()
    ret, cnt = 0, 0
    for edge in edges:
      # check if edges belong to same set before merging and adding edge to mst
      if dsu.find(edge.a) != dsu.find(edge.b):
        dsu.union(edge.a, edge.b)
        ret = ret + edge.weight
        cnt += 1
        if cnt == n - 1:
          break
    return ret
```


# Word 系列
## 模版

## 例题

[139. Word Break](https://leetcode.com/problems/word-break/)

[140. Word Break II](https://leetcode.com/problems/word-break-ii/)

[79. Word Search](https://leetcode.com/problems/word-search/)


[127. Word Ladder](https://leetcode.com/problems/word-ladder/)

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        先用nested loop建一个adjacent list，然后用BFS
        adj: {pattern: [words]} : {*ot: [hot, dot, lot]}；
        找pattern = word[:j] + "*" + word[j+1:]；
        res起点是1；visited和queue一开始要把beginWord放进来：visit.add([beginWord]); queue.append([beginWord])

        时间：O(M^2*N), M is len(word)
        空间：O(M^2*N)
        """
        if endWord not in wordList:
            return 0
        
        pattern_word = collections.defaultdict(list)
        wordList.append(beginWord)
        
        # 对于wordList的每个word，把每一个pattern都找到，然后构建adj
        for word in wordList:
            for j in range(len(word)):
                pattern = word[:j] + "*" + word[j+1:]
                pattern_word[pattern].append(word)
        
        visited = set([beginWord])
        queue = collections.deque([beginWord])
        res = 1
        
        while queue:
            for i in range(len(queue)):
                word = queue.popleft()
                if word == endWord:
                    return res
                
                # 对于每一个word, 如果在map对应的pattern里面，说明是一个选择，就queue加进去visited加进去
                for j in range(len(word)):
                    pattern = word[:j] + "*" + word[j+1:]
                    for nei_word in pattern_word[pattern]:
                        if nei_word not in visited:
                            visited.add(nei_word)
                            queue.append(nei_word)
            res += 1
        
        return 0
```

[126. Word Ladder II](https://leetcode.com/problems/word-ladder-ii/)
```py
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        
        if endWord not in wordList:
            return []
        
        word_queue = deque([[beginWord, [beginWord]]])
        visited = set([beginWord])
        
        patterns = defaultdict(list)
        paths = []
        word_len = len(beginWord)
        
        for word in wordList:
            for index in range(word_len):
                patterns[word[:index] + '*' + word[index + 1:]].append(word)
        
        while word_queue:
            current_level_length = len(word_queue)
            current_level_visited = set()
            
            for _ in range(current_level_length):
                word, word_sequence = word_queue.popleft()
                if word == endWord:
                    paths.append(word_sequence)
                    continue
                for index in range(word_len):
                    pattern  = word[ : index] + '*' + word[index + 1: ]
                    for adjacent_word in patterns[pattern]:
                        if adjacent_word not in visited:
                            word_queue.append([adjacent_word, word_sequence[:] + [adjacent_word]])
                            current_level_visited.add(adjacent_word)
            visited.update(current_level_visited)
        
        return paths
```

```py
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        prefix_d = defaultdict(list)
        for word in wordList:
            for i in range(0,len(word)):
                prefix_d[word[0:i]+"*"+word[i+1:]].append(word)
        
        order = {beginWord: []}
        queue = deque([beginWord])
        temp_q = deque()
        go_on = True
        end_list = []
        
        while queue and go_on:  # There is no node even added to temp_q
            temp_d = {}
            while queue:        # Pop every node on this level
                cur = queue.popleft()
                for i in range(0, len(cur)):
                    for j in prefix_d[cur[0:i]+"*"+cur[i+1:]]:
                        if j == endWord:
                            end_list.append(j)
                            go_on = False
                        if j not in order:
                            if j not in temp_d:
                                temp_d[j] = [cur]
                                temp_q.append(j)
                            else:
                                temp_d[j].append(cur)
            queue = temp_q
            temp_q = deque()
            order.update(temp_d)
        
        ret = []
        
        # DFS to restore the paths
        def dfs(path, node):
            path = path + [node]    # add the node(Deepcopy)
            if order[node] == []:
                ret.append(list(path[::-1]))
                return
            for i in order[node]:
                dfs(path, i)
        if endWord in order:
            dfs([], endWord)
        else:
            return []
        
        return ret
```