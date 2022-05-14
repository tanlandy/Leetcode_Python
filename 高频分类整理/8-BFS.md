[toc]

# Tree

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