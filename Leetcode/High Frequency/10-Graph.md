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

# DFS

1. capture cycles
2. identify connected components

input: G(V, E)
output: two timestamps for every v in V, d[v]=time you first enter or discover a node, f[v]=time finish with that node, classification of edges

idea:
go as deep as you can and then backup

dfs(G):
    for each v in V:
        if v not visited:
            dfsVisit(v)

dfsVisit(v):
    for each u in adj[v]:
        if u not visited:
            dfsVisit(u)
    u is now visited

Time: O(V+E): only visit vertex and edges once

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

BFS层序遍历

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

        # 调用BFS的时机
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

[2352. Equal Row and Column Pairs](https://leetcode.cn/problems/equal-row-and-column-pairs/description/)

```py
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        """
        重点是如何转置二维数组

        时间：O(MN)
        空间：O(MN)
        """
        row_freq = collections.defaultdict(int)
        col_freq = collections.defaultdict(int)
        for row in grid:
            row_freq[tuple(row)] += 1

        # col_grid = [[row[col] for row in grid] for col in range(len(grid[0]))]
        for c in range(len(grid[0])):
            col = [grid[i][c] for i in range(len(grid))]
            col_freq[tuple(col)] += 1
        


        count = 0
        for item in row_freq:
            if item in col_freq:
                count += row_freq[item] * col_freq[item]
        
        return count
```

[36. Valid Sudoku](https://leetcode.cn/problems/valid-sudoku/)

```py
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        N = 9

        # Use hash set to record the status
        rows = [set() for _ in range(N)]
        cols = [set() for _ in range(N)]
        boxes = [set() for _ in range(N)]

        for r in range(N):
            for c in range(N):
                val = board[r][c]
                # Check if the position is filled with number
                if val == ".":
                    continue

                # Check the row
                if val in rows[r]:
                    return False
                rows[r].add(val)

                # Check the column
                if val in cols[c]:
                    return False
                cols[c].add(val)

                # Check the box
                idx = (r // 3) * 3 + c // 3
                if val in boxes[idx]:
                    return False
                boxes[idx].add(val)

        return True
```

[733. Flood Fill](https://leetcode.cn/problems/flood-fill/)

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

        if color == newColor: # 要注意这个特殊情况：一开始的点就和color一样，这样直接返回
            return image
        dfs(sr, sc)
        return image
```

[1197. Minimum Knight Moves](https://leetcode.cn/problems/minimum-knight-moves/)

```py
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        """
        BFS one-direction
        """
        queue = collections.deque([(0, 0, 0)]) # (r, c, step)
        x, y, visited = abs(x), abs(y), set([(0, 0)])
        dirs = [(1, 2), (2, 1), (1, -2), (-2, 1), (-1, 2), (2, -1)]
        while queue:
            a, b, step = queue.popleft()
            if (a, b) == (x, y):
                return step

            for dx, dy in dirs:
                nei_x, nei_y = a + dx, b + dy
                if (nei_x, nei_y) not in visited and -1 <= nei_x <= x + 2 and -1 <= nei_y <= y + 2: # inbound的条件要注意
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

[286. Walls and Gates](https://leetcode.cn/problems/walls-and-gates/)

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

[1293. Shortest Path in a Grid with Obstacles Elimination](https://leetcode.cn/problems/shortest-path-in-a-grid-with-obstacles-elimination/)

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

        state = (0, 0, k) # (r, c, remaining obstacle that can remove)
        queue = collections.deque([(0, state)]) # (step, state)
        visited = set([state])
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while queue:
            steps, (r, c, k) = queue.popleft()
            if r == rows - 1 and c == cols - 1: # reaches the end
                return steps

            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                if 0 <= nei_r < rows and 0 <= nei_c < cols: # in bound
                    nei_k = k - grid[nei_ar][nei_c]
                    nei_state = (nei_r, nei_c, nei_k)
                    if nei_k >= 0 and nei_state not in visited: # 除了visited条件外，还有一个remaining k的条件
                        visited.add(nei_state)
                        queue.append((steps + 1, nei_state))

        return -1
```

[1905. Count Sub Islands](https://leetcode.cn/problems/count-sub-islands/)

```py
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """
        Step1: iterate through all islands in grid2, if any cell is not island in grid1, flood or visit the entire island of grid2
        Step2: count the num of islands in grid2

        Time: O(M*N)
        Space: O(M*N)
        """
        rows, cols = len(grid1), len(grid1[0])

        visit = set()

        def dfs(r, c):
            if r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or grid2[r][c] != 1:
                return

            visit.add((r, c))
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)


        # step1
        for r in range(rows):
            for c in range(cols):
                if grid2[r][c] == 1 and grid1[r][c] == 0 and (r, c) not in visit:
                    dfs(r, c)

        # step2
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid2[r][c] == 1 and (r, c) not in visit:
                    dfs(r, c)
                    count += 1

        return count
```

[694. Number of Distinct Islands](https://leetcode.cn/problems/number-of-distinct-islands/)

```py
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        """
        dfs() returns the current path of the island

        Time: O(M*N)
        Space: O(M*N)
        """
        rows, cols = len(grid), len(grid[0])
        visit = set()

        def dfs(r, c, path):
            if r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or grid[r][c] == 0:
                return "0"

            visit.add((r, c)) # enter the traverse
            d = dfs(r + 1, c, path)
            u = dfs(r - 1, c, path)            
            right = dfs(r, c + 1, path)            
            l = dfs(r, c - 1, path)            
            path = d + u + right + l + "1" # exit the traverse

            return path

        count = set()

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1 and (r, c) not in visit:
                    one_path = dfs(r, c, "o")
                    count.add(one_path)

        return len(count)
```

[130. Surrounded Regions](https://leetcode.cn/problems/surrounded-regions/)

```py
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        走三遍
        1. 先从边界，把所有延伸到边界的O变成T
        2. 把剩下的O变成X
        3. 把T变回O
        时间：O(M*N) 每个都会走到
        空间：O(M*N)
        """

        rows, cols = len(board), len(board[0])

        def capture(r, c):
            if r < 0 or c < 0 or r == rows or c == cols or board[r][c] != "O":
                return

            board[r][c] = "T"

            capture(r + 1, c)
            capture(r - 1, c)
            capture(r, c + 1)
            capture(r, c - 1)

        # O -> T
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "O" and (r in [0, rows - 1] or c in [0, cols - 1]):
                    capture(r, c)

        # O -> X
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "O":
                    board[r][c] = "X"

        # T -> O 
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "T":
                    board[r][c] = "O"
```

[1730. Shortest Path to Get Food](https://leetcode.cn/problems/shortest-path-to-get-food/)

```py
class Solution:
    def getFood(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        
        queue = collections.deque()
        visited = set()
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "*":
                    queue.append((r, c))
                    visited.add((r, c))
                    break
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        step = 0
        
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                if grid[r][c] == "#":
                    return step
                for dx, dy in dirs:
                    nr, nc = r + dx, c + dy
                    if nr in range(rows) and nc in range(cols) and (nr, nc) not in visited and grid[nr][nc] != "X":
                        queue.append((nr, nc))
                        visited.add((nr, nc))
            step += 1
        
        return -1
```

[323. Number of Connected Components in an Undirected Graph](https://leetcode.cn/problems/number-of-connected-components-in-an-undirected-graph/)

```py
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        """
        DFS
        Step1: 构建Adj_list
        Step2: 利用visited set，dfs所有的点，同时记录数量
        """
        graph = collections.defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)

        def dfs(node):
            visit.add(node)
            for nei in graph[node]:
                if nei not in visit:
                    dfs(nei)

        count = 0
        visit = set()
        for node in range(n):
            if node not in visit:
                dfs(node)
                count += 1

        return count
```

[1091. Shortest Path in Binary Matrix](https://leetcode.cn/problems/shortest-path-in-binary-matrix/)

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        """
        就是八个方向的dfs,queue里面放dist：queue=deque([(0,0,1)])

        时间：O(N*N)
        空间：O(N*N)
        """
        n = len(grid)
        if grid[0][0] or grid[n-1][n-1]:
            return -1

        dirs = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,-1],[-1,1]]
        queue = collections.deque([(0,0,1)]) # (r, c, dist)
        visit = set()
        visit.add((0,0))

        while queue:
            r, c, dist = queue.popleft()
            if r == n - 1 and c == n - 1:
                return dist

            for dr, dc in dirs:
                nei_r, nei_c = r + dr, c + dc

                if nei_r in range(n) and nei_c in range(n) and grid[nei_r][nei_c] == 0 and (nei_r, nei_c) not in visit:
                    queue.append((nei_r, nei_c, dist + 1))
                    visit.add((nei_r, nei_c))

        return -1
```

Given a square grid of characters in the range ascii[a-z], rearrange elements of each row alphabetically, ascending. Determine if the columns are also in ascending alphabetical order, top to bottom.

```py
def gridChallenge(grid):
    s = [sorted(i) for i in grid] # 解决行
    for i in (zip(*s)): # 解决列
        if list(i) != sorted(i):
            return "NO"
    return "YES"
```

Find an element of the array such that the sum of all elements to the left is equal to the sum of all elements to the right.
input: arr = [5,6,8,11], return "YES" as the sum before and after 8 are the same

```py
def balancedSums(arr):
    if len(arr) <= 1:
        return "YES"
    prefix = [0] # add[0] for [2,0,0] case
    cur_sum = 0
    for n in arr:
        cur_sum += n
        prefix.append(cur_sum)
    
    # start from idx 1:
    # [1,2,3]
    # pre [0,1,3,6]
    
    for i in range(1, len(prefix)):
        if prefix[i - 1] == prefix[-1] - prefix[i]:
            return "YES"
    
    return "NO"
```

num 和 str的相互转化

```py
def superDigit(n, k):
    """
    总是用str，算完和之后赶紧再转回str进行操作
    """
    # input n is str
    n = list(n) * k #['1', '4', '8', '1', '4', '8', '1', '4', '8']
    a = float("inf")
    while a >= 10:
        a = sum([int(x) for x in n]) # 39
        n = str(a) # n = "39"
        n = list(n) # n = ["3", "9"]
    return a
```

[766. Toeplitz Matrix](https://leetcode.cn/problems/toeplitz-matrix/)

```py
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        """
        for every element, check whether it is the same as it's top-left neighbor when in bound
        """
        
        rows, cols = len(matrix), len(matrix[0])
        
        for r in range(rows):
            for c in range(cols):
                # if (r - 1, c - 1) in bound
                if 0 <= r - 1 < rows and 0 <= c - 1 < cols:
                    # check if they are the same
                    if matrix[r - 1][c - 1] != matrix[r][c]:
                        return False
        
        return True
```

Follow-up:
当流数据每次只能来一行的时候，用一个deque存expected values，每次把最右边的删掉，然后把下一行第一个放到最左边

[1706. Where Will the Ball Fall](https://leetcode.cn/problems/where-will-the-ball-fall/)

```py
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        """
        simulate the condition
        """
        rows, cols = len(grid), len(grid[0])

        def drop(i,j):
            if i == rows: # reaches to the end 
                return j
            if j == cols-1 and grid[i][j] == 1: # hit right bound
                return -1
            if j == 0 and grid[i][j] == -1: # hit left bound
                return -1
            if grid[i][j] == 1 and grid[i][j + 1] == -1: # v shape
                return -1
            if grid[i][j] == -1 and grid[i][j - 1] == 1: # v shape
                return -1
            return drop(i + 1, j + grid[i][j]) # to the next row
        
        return [drop(0, j) for j in range(cols)]
```

### 迷宫问题

[490. The Maze](https://leetcode.cn/problems/the-maze/)
一次走到底的情况

```py
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        """
        时间：O(M*N)
        空间：O(M*N)
        """
        rows, cols = len(maze), len(maze[0])
        visited = set()
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        queue = collections.deque([start])
        while queue:
            r, c = queue.popleft()
            if [r, c] == destination: # 找到了
                return True

            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                while 0 <= nei_r < rows and 0 <= nei_c < cols and maze[nei_r][nei_c] == 0: # 一直走
                    nei_r += dx
                    nei_c += dy
                nei_r -= dx # 这时候多走了一步，已经站在墙上了，所以退回来一步
                nei_c -= dy
                if (nei_r, nei_c) not in visited:
                    visited.add((nei_r, nei_c))
                    queue.append((nei_r, nei_c))

        return False
```

```py
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        """
        时间：O(M*N)
        空间：O(M*N)
        """
        rows, cols = len(maze), len(maze[0])
        visited = set()
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def dfs(r, c):
            if [r, c] == destination: # base case
                return True
            for dx, dy in dirs: # visit neighbors in four directions
                nei_r, nei_c = r + dx, c + dy
                while 0 <= nei_r< rows and 0 <= nei_c< cols and maze[nei_r][nei_c] == 0: # 如果新的位置满足要求，就一直走一直走，走到新的位置
                    nei_r += dx # 这时候多走了一步，已经站在墙上了，所以退回来一步
                    nei_c += dy
                nei_r -= dx
                nei_c -= dy
                if (nei_r, nei_c) not in visited: # 如果没走过：加到set里，然后试着走一下
                    visited.add((nei_r, nei_c))
                    if dfs(nei_r, nei_c):
                        return True
            return False

        return dfs(start[0], start[1])
```

[490变形]：问一共转了几次弯

```py
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        rows, cols = len(maze), len(maze[0])
        visited = set()
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        res = float("inf")

        queue = collections.deque([(start[0], start[1], 0)])
        while queue:
            r, c, step = queue.popleft()
            if [r, c] == destination:
                return step

            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                while 0 <= nei_r < rows and 0 <= nei_c < cols and maze[nei_r][nei_c] == 0:
                    nei_r += dx
                    nei_c += dy
                nei_r -= dx # 这时候多走了一步，已经站在墙上了，所以退回来一步
                nei_c -= dy
                nei_step = step + 1
                if (nei_r, nei_c) not in visited:
                    visited.add((nei_r, nei_c))
                    queue.append((nei_r, nei_c, nei_step))

        return -1
```

[505. The Maze II](https://leetcode.cn/problems/the-maze-ii/)

```py
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        """
        相比LC490，queue另外记录了距离，由此可以作为条件来判断是否“走过”，以及导出结果
        visited是一个dict，用来记录走过的位置的步数

        时间：O(M*N*min(M, N))
        空间：O(M*N)
        """
        rows, cols = len(maze), len(maze[0])
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        if start == destination:
            return 0

        # (r, c, distance)
        queue = deque([(start[0], start[1], 0)])

        # start position marked visited: {(r, c): dist}
        visited = {(start[0], start[1]): 0}
        res = []

        while queue:
            r, c, dist = queue.popleft()
            if [r, c] == destination:
                res.append(dist)

            for dx, dy in dirs:
                nei_r, nei_c, nei_dist = r + dx, c + dy, dist + 1

                while 0 <= nei_r < rows and 0 <= nei_c < cols and maze[nei_r][nei_c] == 0:
                    nei_dist += 1
                    nei_r += dx
                    nei_c += dy

                nei_r -= dx
                nei_c -= dy
                nei_dist -= 1

                # TWO CONDITIONS ==> there is better way to visit the previously visited position, mark the distance OR not visited before
                if ((nei_r, nei_c) not in visited) or (nei_dist < visited[(nei_r, nei_c)]):
                    visited[(nei_r, nei_c)] = nei_dist
                    queue.append((nei_r, nei_c, nei_dist))

        return min(res) if res else -1
```

```py
class Solution:
    def shortestDistance(self, maze, start, destination):
        """
        用min_heap来存，key是dist，这样第一次走到destination就是最小值
        时间：O(M*N*log(M*N))
        空间：O(M*N)
        """
        rows, cols = len(maze), len(maze[0])
        min_heap = [(0, start[0], start[1])]
        visited = {(start[0], start[1]):0}
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while min_heap:
            dist, r, c = heapq.heappop(min_heap)
            if [r, c] == destination:
                return dist
            for dx, dy in dirs:
                nei_r, nei_c, nei_dist = r + dx, c + dy, dist + 1

                while 0 <= nei_r < rows and 0 <= nei_c < cols and maze[nei_r][nei_c] == 0:
                    nei_dist += 1
                    nei_r += dx
                    nei_c += dy

                nei_r -= dx
                nei_c -= dy
                nei_dist -= 1
                if ((nei_r, nei_c) not in visited) or (nei_dist < visited[(nei_r, nei_c)]):
                    visited[(nei_r, nei_c)] = nei_dist
                    heapq.heappush(min_heap, (nei_dist, nei_r, nei_c))
        return -1
```

[341. Flatten Nested List Iterator](https://leetcode.cn/problems/flatten-nested-list-iterator/)

[339. Nested List Weight Sum](https://leetcode.cn/problems/nested-list-weight-sum/)

[364. Nested List Weight Sum II](https://leetcode.cn/problems/nested-list-weight-sum-ii/)

[419. Battleships in a Board](https://leetcode.cn/problems/battleships-in-a-board/)

```py
class Solution:
    def countBattleships(self, board):
        """
        as battleships are 1*k or k*1:
        we only count the top-left corner "X", we move on if there's "X" to the top or to the left
        """
        total = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'X':
                    flag = 1
                    if j > 0 and board[i][j-1] == 'X': flag = 0
                    if i > 0 and board[i-1][j] == 'X': flag = 0
                    total += flag
        return total
```

```py
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        rows, cols = len(board), len(board[0])

        def dfs(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] == ".":
                return
            
            board[r][c] = "."
            dfs(r, c + 1)
            dfs(r + 1, c)
            dfs(r, c - 1)
            dfs(r - 1, c)
        
        count = 0
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "X":
                    dfs(r, c)
                    count += 1
        
        return count
```

[531. Lonely Pixel I](https://leetcode.cn/problems/lonely-pixel-i/)

```py
class Solution:
    def findLonelyPixel(self, picture: List[List[str]]) -> int:
        """
        先算出来每行每列有几个B，然后同时满足是B且行列都是1的就是需要的点
        """
        rows, cols = len(picture), len(picture[0])
        row_count = [0] * cols
        col_count = [0] * rows
        
        for r in range(rows):
            for c in range(cols):
                if picture[r][c] == "B":
                    row_count[c] += 1
                    col_count[r] += 1
        
        res = 0
        for r in range(rows):
            for c in range(cols):
                if picture[r][c] == "B" and row_count[c] == col_count[r] == 1:
                    res += 1
        
        return res

```

### Matrix

[59. Spiral Matrix II](https://leetcode.cn/problems/spiral-matrix-ii/)

```py
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        """
        走到头的时候换方向：
        1. 走到边界
        2. 走到走过的点
        
        dx, dy的位置，组合起来对应着右、下、左、上
        
        """
        res = [[0] * n for _ in range(n)]
        
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        r, c = 0, 0
        d = 0
        
        for num in  range(1, n * n + 1):
            res[r][c] = num
            nr, nc = r + dx[d], c + dy[d]
            if nr < 0 or nr >= n or nc < 0 or nc >= n or res[nr][nc] != 0:
                d = (d + 1) % 4
                nr, nc = r + dx[d], c + dy[d]
            r, c = nr, nc
        
        return res
```

[542. 01 Matrix](https://leetcode.cn/problems/01-matrix/)

```py
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        """
        把所有为0的点放进queue，然后开始BFS

        Time: O(M*N)
        Space: O(M*N)
        """
        rows, cols = len(mat), len(mat[0])
        queue = collections.deque([])
        seen = set()  # 最经典解法，使用seen来记录更新过的值

        for r in range(rows):
            for c in range(cols):
                if mat[r][c] == 0:
                    queue.append((r, c))
                    seen.add((r, c))
        dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        while queue:
            r, c = queue.popleft()

            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                if 0 <= nei_r < rows and 0 <= nei_c < cols and (nei_r, nei_c) not in seen:  # 如果没有更新过
                    queue.append((nei_r, nei_c))
                    seen.add((nei_r, nei_c))  # 更新了
                    mat[nei_r][nei_c] = mat[r][c] + 1  # 更新掉
        
        return mat
            
```

```py
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        """
        把所有为0的点放进queue，然后开始BFS

        Time: O(M*N)
        Space: O(M*N)
        """
        rows, cols = len(mat), len(mat[0])
        queue = collections.deque([])

        for r in range(rows):
            for c in range(cols):
                if mat[r][c] == 0:
                    queue.append((r, c))
                else: # 不为0的，标记为-1，替代seen作为是否更新过的标准
                    mat[r][c] = -1

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # 从所有0开始BFS往四周走，因为已经赋值-1，所以不用visited set
        while queue:
            r, c = queue.popleft()

            for dx, dy in dirs:
                nei_r, nei_c = r + dx, c + dy
                if 0 <= nei_r < rows and 0 <= nei_c < cols and mat[nei_r][nei_c] == -1: #条件不用visited set，因为-1肯定就是没走过的点
                    mat[nei_r][nei_c] = mat[r][c] + 1 # 附近的新值就是原来的+1
                    queue.append((nei_r, nei_c))

        return mat
```

```py
class Solution:  # 520 ms, faster than 96.50%
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        """
        四周都“加”一圈float("inf")
        然后从左上到右下走一遍
        再从右下到左上走一遍

        Time: O(M*N)
        Space: O(1)
        """
        rows, cols = len(mat), len(mat[0])

        # from top left to bottom right
        for r in range(rows):
            for c in range(cols):
                if mat[r][c] > 0:
                    top = mat[r - 1][c] if r > 0 else float("inf")
                    left = mat[r][c - 1] if c > 0 else float("inf")
                    mat[r][c] = min(top, left) + 1

        # from bottom right to top left
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                if mat[r][c] > 0:
                    bottom = mat[r + 1][c] if r < rows - 1 else float("inf")
                    right = mat[r][c + 1] if c < cols - 1 else float("inf")
                    mat[r][c] = min(mat[r][c], bottom + 1, right + 1)

        return mat
```

求两对角线之和的差

```py
def diagonalDifference(arr):
    diag1 = 0
    diag2 = 0
    rows, cols = len(arr), len(arr[0])
    for i in range(rows):
        for j in range(cols):
            if i == j:
                diag1 += arr[i][j]
            if i + j == rows - 1: # 唯一新颖的点：反对角线的和相同
                diag2 += arr[i][j]
    res = abs(diag1 - diag2)
    return res
```

求可以调整顺序的N*N最大和

```py

# find the maximum value
# of top N/2 x N/2 matrix using row and column reverse operations
def maxSum(mat):
 
    Sum = 0
    for i in range(0, R // 2):
        for j in range(0, C // 2):
         
            r1, r2 = i, R - i - 1
            c1, c2 = j, C - j - 1
                 
            # We can replace current cell [i, j]
            # with 4 cells without changing/affecting
            # other elements.
            Sum += max(max(mat[r1][c1], mat[r1][c2]),
                       max(mat[r2][c1], mat[r2][c2]))
         
    return Sum
```

[200. Number of Islands](https://leetcode.cn/problems/number-of-islands/)
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
            if (r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or grid[r][c] != "1"): # grid[r][c] != "1"要放在最后，避免提前因为out of index而报错
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

[417. Pacific Atlantic Water Flow](https://leetcode.cn/problems/pacific-atlantic-water-flow/)

```py
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        """
        分别从Pac和Atl来看哪些点满足，最后都满足的点就是最终结果
        从Pac来看哪些点满足：
        1，从第一行和第一列，分别作dfs
        2，dfs的时候：提前返回的条件只多一个cur_height < pre_height
        """
        pac, atl = set(), set()
        rows, cols = len(heights), len(heights[0])

        def dfs(r, c, visit, pre_height):
            if (r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or heights[r][c] < pre_height):
                return
            visit.add((r, c))

            dfs(r + 1, c, visit, heights[r][c])
            dfs(r - 1, c, visit, heights[r][c])
            dfs(r, c + 1, visit, heights[r][c])
            dfs(r, c - 1, visit, heights[r][c])

        # 第一行最后一行，第一列最后一列分别找满足的点
        for c in range(cols):
            dfs(0, c, pac, heights[0][c]) # 第一行Pac
            dfs(rows - 1, c, atl, heights[rows - 1][c]) # 最后一行Atl

        for r in range(rows):
            dfs(r, 0, pac, heights[r][0]) # 第一列Pac
            dfs(r, cols - 1, atl, heights[r][cols - 1]) # 最后一列Atl

        # 都满足的点就是最终的点
        res = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in pac and (r, c) in atl:
                    res.append([r, c])
        return res
```

[695. Max Area of Island](https://leetcode.cn/problems/max-area-of-island/)

```py
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        """
        find the area of each island, then return the maximum one
        the dfs() returns the current area while traversing the island
        Time: O(M*N)
        Space: O(M*N)
        """
        rows, cols = len(grid), len(grid[0])
        visited = set()

        def dfs(r, c):  # return the current area while traversing the island
            if r not in range(0, rows) or c not in range(0, cols) or (r, c) in visited or grid[r][c] != 1:
                return 0  # reach the end, the current area is 0

            visited.add((r, c))
            
            return (
                1 +
                dfs(r + 1, c) +
                dfs(r - 1, c) +
                dfs(r, c + 1) +
                dfs(r, c - 1)            
            )

        max_area = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1 and (r, c) not in visited:
                    cur_area = dfs(r, c)
                    max_area = max(max_area, cur_area)

        return max_area               
```

[1020. Number of Enclaves](https://leetcode.cn/problems/number-of-enclaves/)

```py
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        """
        The same as LC1254

        Step1: visit all the cells that can walk off the boundary
        Step2: count the remaining cells

        Time: O(M*N)
        Space: O(M*N)
        """

        rows, cols = len(grid), len(grid[0])

        visit = set()

        def dfs(r, c):
            if r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or grid[r][c] != 1:
                return

            visit.add((r, c))
            dfs(r + 1, c)
            dfs(r - 1, c)            
            dfs(r, c + 1)            
            dfs(r, c - 1)

        # step1
        for r in range(rows):
            dfs(r, 0)
            dfs(r, cols - 1)

        for c in range(cols):
            dfs(0, c)
            dfs(rows - 1, c)

        # step2
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1 and (r, c) not in visit:
                    count += 1

        return count
```

[1254. Number of Closed Islands](https://leetcode.cn/problems/number-of-closed-islands/)

```py
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        """
        step1: flood or visit all the islands that in the boarder
        step2: count the num of islands

        Time: O(M*N)
        Space: O(M*N)
        """

        rows, cols = len(grid), len(grid[0])

        visit = set()

        def dfs(r, c):
            if r < 0 or r == rows or c < 0 or c == cols or grid[r][c] != 0 or (r, c) in visit:
                return

            visit.add((r, c))
            dfs(r + 1, c)
            dfs(r - 1, c)            
            dfs(r, c + 1)            
            dfs(r, c - 1)

        # step1
        for r in range(rows):
            dfs(r, 0)
            dfs(r, cols - 1)

        for c in range(cols):
            dfs(0, c)
            dfs(rows - 1, c)

        # step2
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0 and (r, c) not in visit:
                    dfs(r, c)
                    count += 1

        return count
```

# Implicit Graph

[752. Open the Lock](https://leetcode.cn/problems/open-the-lock/)

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

[994. Rotting Oranges](https://leetcode.cn/problems/rotting-oranges/)

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
        rows, cols = len(grid), len(grid[0])
        queue = collections.deque()
        time, fresh = 0, 0

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

```py
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        """
        add all rotten oranges to queue
        do bfs: pop oranges from queue, change the adj fresh oranges to rotten, while maintaining a timer
        iterate through all oranges and see if there are any remaining fresh one, if so, return -1, else return timer
        """
        
        rows, cols = len(grid), len(grid[0])
        queue = collections.deque()
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))
        
        timer = 0
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for dx, dy in dirs:
                    nr, nc = r + dx, c + dy
                    if nr in range(rows) and nc in range(cols) and grid[nr][nc] == 1:
                        queue.append((nr, nc))
                        grid[nr][nc] = 2
            timer += 1
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    return -1

        return timer - 1 if timer > 0 else 0 # 如果有的话，每次最后都会多走一个timer，所以要-1。其他的bfs题目都是提前在while queue里面就返回了，不会再走一个多的timer+=1
```

[773. Sliding Puzzle](https://leetcode.cn/problems/sliding-puzzle/)

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

[261. Graph Valid Tree](https://leetcode.cn/problems/graph-valid-tree/)

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

[815. Bus Routes](https://leetcode.cn/problems/bus-routes/)

```py
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        """
        画一个决策树，最短路径就是到达终点所需要的最小层数：BFS
        Step1: 构建一个图{each stop: bus}
        Step2: bfs放进source，然后开始遍历所有的bus，对于走过的bus就不再遍历，对于走过的stop也不再遍历
        """
        graph = collections.defaultdict(list)

        for idx, route in enumerate(routes):
            for stop in route:
                graph[stop].append(idx)

        visited_bus = set()
        visited_stop = set()
        # queue stores (stop, step)
        queue = collections.deque([(source, 0)])
        visited_stop.add(source)

        while queue:
            stop, steps = queue.popleft()
            if stop == target: # base case
                return steps

            for bus in graph[stop]:
                if bus not in visited_bus: # new bus
                    visited_bus.add(bus)
                    for next_stop in routes[bus]:
                        if next_stop not in visited_stop: # new stop
                            visited_stop.add(next_stop)
                            queue.append((next_stop, steps + 1))                

        return -1
```

[841. Keys and Rooms](https://leetcode.cn/problems/keys-and-rooms/)

```py
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        """
        graph traverse
        """
        visited = set()
        
        def dfs(room):
            if room in visited:
                return
            visited.add(room)
            for v in rooms[room]:
                dfs(v)
        
        dfs(0)
        return len(visited) == len(rooms)
```

[1376. Time Needed to Inform All Employees](https://leetcode.cn/problems/time-needed-to-inform-all-employees/)

```py

class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        q = collections.deque([(headID, 0)])
        subordinates = collections.defaultdict(list)
        res = 0
        for i, v in enumerate(manager):
            subordinates[v].append(i)
            
        while q:
            u, time = q.popleft()
            res = max(res, time)
            for v in subordinates[u]:
                q.append((v, time + informTime[u]))
        return res
```

# Un-weighted Graph

[1615. Maximal Network Rank](https://leetcode.cn/problems/maximal-network-rank/description/)

```py
class Solution:
    def maximalNetworkRank(self, n: int, roads: List[List[int]]) -> int:
        res = 0
        adj = defaultdict(set)

        # adjacent list
        for road in roads:  
            adj[road[0]].add(road[1])
            adj[road[1]].add(road[0])
        
        # for each pair, calculate the rank
        for node1 in range(n):
            for node2 in range(node1+1, n):
                cur_rank = len(adj[node1]) + len(adj[node2])
                if node2 in adj[node1]:
                    cur_rank -= 1
                res = max(res, cur_rank)
        
        return res
```

[261. Graph Valid Tree](https://leetcode.cn/problems/graph-valid-tree/)

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

        graph = [[] for _ in range(n)]
        for n1, n2 in edges:
            graph[n1].append(n2)
            graph[n2].append(n1)

        visit = set()
        def hasCycle(i, prev):
            if i in visit: # detect a loop
                return True

            visit.add(i)
            for j in graph[i]:
                if j == prev:
                    continue
                if hasCycle(j, i):
                    return True

            return False

        return not hasCycle(0, -1) and n == len(visit)
```

## 图的遍历

[797. All Paths From Source to Target](https://leetcode.cn/problems/all-paths-from-source-to-target/)

```py
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        """
        traverse the graph from graph[0], while maintaining an one_res path, the base case is reaching the end of the graph
        """
        res = []

        def dfs(cur_node, one_res):
            # base case
            if cur_node == len(graph)-1:
                res.append(one_res.copy())
                return

            # traverse neighbors
            for node in graph[cur_node]:
                one_res.append(node)
                dfs(node, one_res) 
                one_res.pop()

        # initiate one_res with [0], as the first element is not added from dfs()：only added 0'neighbors at the beginning
        dfs(0, [0])
        return res
```

[785. Is Graph Bipartite?](https://leetcode.cn/problems/is-graph-bipartite/)

```py
class Solution(object):
    def isBipartite(self, graph):
        """
        遍历一遍图，一边遍历一边染色，看看能不能用两种颜色给所有节点染色，且相邻节点的颜色都不相同。

        时间：O(V+E)
        空间：O(V) 用来存visit
        """

        def traverse(v, color):
            # base case 已经走过
            if v in visit:
                return visit[v] == color: # 判断颜色是否相同

            # 没有走过
            visit[v] = color
            for nei in graph[v]:
                if not traverse(nei, -color): # 给nei涂上不同的颜色
                    return False
            return True

        visit = {} # {visited vertex: color}既能确认是否visited，又能比较color

        # 对每一个点都要遍历
        for i in range(len(graph)):
            if i not in visit: # 对于没有走过的点，都作为起点，来检查是否分别是二分图
                if not traverse(i, 1): # 有一个不是二分图，就return False
                    return False
        return True
```

[886. Possible Bipartition](https://leetcode.cn/problems/possible-bipartition/)

```py
class Solution:
    def possibleBipartition(self, N: int, dislikes: List[List[int]]) -> bool:
        """
        与LC785一样，只是多了一个建图的过程

        时间：O(V+E)
        空间：O(V) 用来存visit
        """
        def traverse(v, color):                
            if v in visit:
                return visit[v] == color

            visit[v] = color        
            for nei in graph[v]:                
                if not traverse(nei, -color):
                    return False
            return True

        graph = collections.defaultdict(list)
        visit = {}      

        for a,b in dislikes: 
            graph[a].append(b)
            graph[b].append(a)

        for i in range(1,N+1):            
            if i not in visit:
                if not traverse(i, 1):
                    return False
        return True
```

[133. Clone Graph](https://leetcode.cn/problems/clone-graph/)

```python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        """
        HashMap:{oldNode: newNode}；
        dfs(node)返回node对应的copy
        每次如果在map里面就直接返回copy后的node，如果不在就copy然后copy自己的neighbors；
        复制neighbors: for nei in node.neighbors: copy.neighbors.append(dfs(nei))


        时间：O(V+E)
        空间：O(V)
        """
        if node is None:
            return None
        oldToNew = {}

        def dfs(node):
            if node in oldToNew:
                return oldToNew[node]

            copy = Node(node.val)
            oldToNew[node] = copy

            for nei in node.neighbors:
                copy.neighbors.append(dfs(nei))

            return copy

        return dfs(node)
```

[1059. All Paths from Source Lead to Destination](https://leetcode.cn/problems/all-paths-from-source-lead-to-destination/)

```py
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        # Creating the graph
        g = [set() for _ in range(n)]
        for edge in edges:
            g[edge[0]].add(edge[1])
        
        # destination should not point to any other node
        if len(g[destination]) > 0:
            return False

        seen = set()
        
        def dfs(node):
            # If can't reach any other node, node has to be destination
            if len(g[node]) == 0:
                return node == destination

            for neighborg in g[node]:
                if neighborg in seen:
                    # Cycle Found!!!
                    return False
                
                seen.add(neighborg)
                if not dfs(neighborg):
                    # We stop if the path could not reach destination
                    return False
                seen.remove(neighborg)
            
            # Congratulations all paths reaches destination
            return True
        
        return dfs(source)
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

[743. Network Delay Time](https://leetcode.cn/problems/network-delay-time/)

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
            if n1 in visited: # if visited from other nodes
                continue
            res = t1
            visited.add(n1)

            # traverse its unvisited neighbors
            for n2, w2 in graph[n1]:
                if n2 not in visited:
                    heapq.heappush(min_heap, (t1 + w2, n2))

        return res if len(visited) == n else -1
```

[1135. Connecting Cities With Minimum Cost](https://leetcode.cn/problems/connecting-cities-with-minimum-cost/)
也可以用MST来做

```py
class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        # build a adjcency list with N nodes
        adj = {i:[] for i in range(1, n + 1)} # 这些点都从1开始
        for x, y, cost in connections:
            adj[x].append([cost, y]) # i: [cost, node]
            adj[y].append([cost, x])
        
        # dijkstra to find the min cost to visit all nodes
        minH = [(0, 1)] # (cost, node)
        visited = set()
        res = 0
        
        while minH:
            cost, node = heapq.heappop(minH)
            if node in visited:
                continue
                
            visited.add(node)
            res += cost
            for nei_cost, nei in adj[node]:
                if nei not in visited:
                    heapq.heappush(minH, (nei_cost, nei))
        
        return res if len(visited) == n else -1
```

[1514. Path with Maximum Probability](https://leetcode.cn/problems/path-with-maximum-probability/)

[1631. Path With Minimum Effort](https://leetcode.cn/problems/path-with-minimum-effort/)

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

[1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance]([Loading...](https://leetcode.cn/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/))

# Topological Sort

## 基础知识

Topological sort or topological ordering of a directed graph is an ordering of nodes such that every node appears in the ordering before all the nodes it points to.

Topological sort is not unique

Graph with cycles do not have topological ordering

## 模版：Kahn's Algorithm

To obtain a topological order, we can use Kahn's algorithm

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

## 高频题

### 知乎

### Krahets精选题

### AlgoMonster

### Youtube

207/210

## 题目

[207. Course Schedule](https://leetcode.cn/problems/course-schedule/)

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

        while queue:
            v = queue.popleft()
            count += 1
            for to_ in graph[v]:
                indegree[to_] -= 1
                if indegree[to_] == 0:
                    queue.append(to_)

        return count == numCourses # 根据是否记录的数量等于总课程数量
```

DFS成环问题

```py
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        就是看是否会成环
        用adjcent list存[course:[prereq_courses]]
        从0到n-1，按adj list跑到底，看这个course能否上掉
        用visit set来记录是否已经走过

        时间：O(V + E)
        空间：O(V + E)
        """        
        graph = [[] for _ in range(numCourses)]
        for course, pre in prerequisites:
            graph[course].append(pre)

        # visit = all courses along the curr DFS path
        visit = set()

        def hasCycle(course):
            # base case
            if course in visit: # already processed: detect a loop
                return True
            if graph[course] == []:
                return False

            visit.add(course)
            for pre in graph[course]:  # 对于每个点，走到底
                if hasCycle(pre): # 发现一个是False，就返回False
                    return True

            visit.remove(course)
            graph[course] = []
            return False

        for course in range(numCourses): # graph可能不是fully connected，所以要从每个点开始走
            if hasCycle(course):
                return False

        return True
```

[210. Course Schedule II](https://leetcode.cn/problems/course-schedule-ii/)

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
        count = 0  # 记录走过的数量
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

[444. Sequence Reconstruction](https://leetcode.cn/problems/sequence-reconstruction/)

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

[953. Verifying an Alien Dictionary](https://leetcode.cn/problems/verifying-an-alien-dictionary/)

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

[269. Alien Dictionary](https://leetcode.cn/problems/alien-dictionary/) 先做LC953

```py
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        """
        compare word pairs one by one. no need to compare n^2 times.
        t<f
        w<e
        r<t
        e<r
        
        w<e<r<t<f
        
        step1: adj_list, indegree
        step2: start from nodes which indegree == 0, then go from its neighbor
        """
        
        graph = {c: set() for w in words for c in w}
        indegree = {c: 0 for w in words for c in w}
        
        # step1: build graph and indegree
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            minLen = min(len(w1), len(w2))
            if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
                return ""
            for j in range(minLen):
                if w1[j] != w2[j]: # 当有不同就比较，同时break
                    if w2[j] not in graph[w1[j]]: # ["ac", "ab", "zc", "zb"] 避免两次计算c->b
                        graph[w1[j]].add(w2[j])
                        indegree[w2[j]] += 1
                    break                        

        # step2: initialize queue
        queue = collections.deque()
        for ch in indegree:
            if indegree[ch] == 0:
                queue.append(ch)

        # step3: topological sort
        res = []
        while queue:
            ch = queue.popleft()
            res.append(ch)
            for nei in graph[ch]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    queue.append(nei)

        # if there are any loop, the length of res would be shorter
        if len(res) < len(indegree):
            return ""
        
        return "".join(res)
```

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

[2115. Find All Possible Recipes from Given Supplies](https://leetcode.cn/problems/find-all-possible-recipes-from-given-supplies/)

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

[329. Longest Increasing Path in a Matrix](https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/)

```py
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        """
        Topological sort:
        1. initial a 2D grid with 0, give 1 to those have a larger value than neighbor.
        2. add grid with 0 to queue
        3. start kahn's algo

        """
        rows, cols = len(matrix), len(matrix[0])
        indegree = [[0 for _ in range(cols)] for _ in range(rows)]

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for r in range(rows):
            for c in range(cols):
                for dx, dy in dirs:
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if matrix[nr][nc] < matrix[r][c]:
                            indegree[r][c] += 1

        queue = collections.deque()
        for r in range(rows):
            for c in range(cols):
                if indegree[r][c] == 0:
                    queue.append((r, c))

        res = 0

        while queue:
            size = len(queue)
            for i in range(size):
                r, c = queue.popleft()
                for dx, dy in dirs:
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if matrix[nr][nc] > matrix[r][c]:
                            indegree[nr][nc] -= 1
                            if indegree[nr][nc] == 0:
                                queue.append((nr, nc))
            res += 1

        return res
```

[310. Minimum Height Trees](https://leetcode.cn/problems/minimum-height-trees/)

```py
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        build the adjlist, then trim out the node with indegree of 1, layer by layer
        ==> trim out the leaf nodes again and again. At the end, the process terminates at the centroids nodes
        """
        if n <= 2:
            return [i for i in range(n)]
        graph = {x: [] for x in range(n)}
        
        for start, end in edges:
            graph[start].append(end)
            graph[end].append(start)
        
        # add node with indgree of 1 to queue
        queue = collections.deque()
        for i in range(n):
            if len(graph[i]) == 1:
                queue.append(i)
        
        remaining_nodes = n
        
        while remaining_nodes > 2:
            remaining_nodes -= len(queue) # trim out leaf nodes layer by layer
            for _ in range(len(queue)):
                leaf = queue.popleft()
                nei = graph[leaf][0] # the only node the linked with leaf
                # remove the edge
                graph[nei].remove(leaf)
                graph[leaf].remove(nei)
                if len(graph[nei]) == 1: # update queue for next trimming round
                    queue.append(nei)
        
        return queue
```

# Word 系列

## 模版

## 例题

[433. Minimum Genetic Mutation](https://leetcode.cn/problems/minimum-genetic-mutation/)

```py
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        queue = deque([(start, 0)])
        seen = set(start)
        
        while queue:
            node, steps = queue.popleft()
            if node == end:
                return steps
            
            for c in "ACGT":
                for i in range(len(node)):
                    nei = node[: i] + c  + node[i + 1:]
                    if nei not in seen and nei in bank:
                        queue.append((nei, steps + 1))
                        seen.add(nei)
        
        return -1
```

[290. Word Pattern](https://leetcode.cn/problems/word-pattern/)

```py
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        map_char = {}
        map_word = {}
        
        words = s.split(' ')
        if len(words) != len(pattern):
            return False
        
        for c, w in zip(pattern, words):
            if c not in map_char:
                if w in map_word:
                    return False
                else:
                    map_char[c] = w
                    map_word[w] = c
            else:
                if map_char[c] != w:
                    return False
        return True
```

[291. Word Pattern II](https://leetcode.cn/problems/word-pattern-ii/)

[139. Word Break](https://leetcode.cn/problems/word-break/)

```py
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        dp[i] means s[i:] whether can be formed by words in wordDict or not

        From right to left
        Time: O(N*M*N), N is len(s), M is len(wordDict)
        Space: O(N+M)
        """

        dp = [False] * (len(s) + 1)
        dp[len(s)] = True

        for i in range(len(s) - 1, -1, -1):
            for w in wordDict:
                if (i + len(w) <= len(s)) and s[i:i + len(w)] == w:
                    dp[i] = dp[i + len(w)] # at idx i, dp[i] determines at dp[i+len(w)] if s[i:i+len(w)] == w
                if dp[i]:
                    break
        return dp[0]
```

```py
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        DFS + Memo

        Time: O(N^3)
        Space: O(N)
        """
        if not s:
            return False
        words = set(wordDict)
        memo = {}

        def dfs(s):
            if s in memo:
                return memo[s]
            if not s:
                return True
            for word in words:
                # 前面不同就跳过
                if s[:len(word)] != word:
                    continue
                # 前面相同就可以往后看
                remain = dfs(s[len(word):])
                if remain:
                    memo[s] = True # 保存remain的结果
                    return True
            memo[s] = False
            return False

        return dfs(s)
```

[140. Word Break II](https://leetcode.cn/problems/word-break-ii/)

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        def backtrack(res, one_res, s):
            if len(s) == 0:
                res.append(" ".join(one_res))
                return
            
            for w in wordDict:
                if s.startswith(w):
                    one_res.append(w)
                    backtrack(res, one_res, s[len(w):])
                    one_res.pop()
                    
        res = []
        backtrack(res, [], s)
        return res
```

[79. Word Search](https://leetcode.cn/problems/word-search/)

```py
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        dfs(r, c, i)同时传入一个idx

        时间：O(M*N*4 ^ N)
        空间：O(L) L is len(words)
        """
        rows, cols = len(board), len(board[0])
        visited = set()
        
        def dfs(r, c, i):
            # Base case
            if i == len(word):
                return True
            
            # 排除的条件
            if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in visited or board[r][c] != word[i]:
                return False
            
            # 做选择
            visited.add((r, c))
            # Backtrack
            res =  (dfs(r + 1, c, i + 1) or
                    dfs(r - 1, c, i + 1) or         
                    dfs(r, c + 1, i + 1) or         
                    dfs(r, c - 1, i + 1)
                  )
            # 回溯
            visited.remove((r, c))            
            return res
        
        for r in range(rows):
            for c in range(cols):
                if dfs(r, c, 0):
                    return True
        
        return False
```

[127. Word Ladder](https://leetcode.cn/problems/word-ladder/)

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
        # wordList.append(beginWord) # 开始的word可放可不放进来

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

[126. Word Ladder II](https://leetcode.cn/problems/word-ladder-ii/)

```py
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return []

        queue = deque([(beginWord, [beginWord])]) # 同时存(word,path)
        visited = set([beginWord])

        pattern_word = defaultdict(list)
        paths = []

        for word in wordList:
            for i in range(len(word)):
                pattern = word[:i] + "*" + word[i + 1:]
                pattern_word[pattern].append(word)

        while queue:
            current_level_visited = set()
            for _ in range(len(queue)):
                word, path = queue.popleft()
                if word == endWord:
                    paths.append(path)
                    continue
                for i in range(len(word)):
                    pattern  = word[:i] + '*' + word[i + 1: ]
                    for nei in pattern_word[pattern]:
                        if nei not in visited: # 只要不在visited，就可以加进来
                            queue.append([nei, path[:] + [nei]])
                            current_level_visited.add(nei)
            visited.update(current_level_visited) # 把这层见过的set加到visited里

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
