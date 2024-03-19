# BFS

# 高频题

## 知乎

## Krahets精选题

## AlgoMonster

## Youtube

103, 127/126, 490/505, 994, 1730

# 例题

[103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        反转list: oneRes.reverse()；翻转isOdd: isOdd = not isOdd
        """
        if not root:
            return []
        odd = True
        res = []
        queue = collections.deque([root])

        while queue:
            one_res = []
            size = len(queue)

            for _ in range(size):
                node = queue.popleft()
                one_res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            if not odd:
                one_res.reverse()
            odd = not odd
            res.append(one_res)

        return res

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

[490. The Maze](https://leetcode.cn/problems/the-maze/)
[787. 迷宫](https://www.lintcode.com/problem/787/)

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

[490变形]：问一共转了几次弯

```py
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        rows, cols = len(maze), len(maze[0])
        visited = set()
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

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

        while queue and fresh > 0:  # 加上fresh>0的条件
            for i in range(len(queue)):
                r, c = queue.popleft()

                for dx, dy in dirs:
                    nei_r, nei_c = r + dx, c + dy
                    # if in bounds and fresh, make it rotten
                    if 0 <= nei_r < rows and 0 <= nei_c < cols and grid[nei_r][nei_c] == 1:
                        grid[nei_r][nei_c] = 2
                        queue.append((nei_r, nei_c))
                        fresh -= 1

            time += 1

        return time if fresh == 0 else -1
```

[1730. Shortest Path to Get Food](https://leetcode.cn/problems/shortest-path-to-get-food/)
[3719 获取奶茶的最短路径](https://www.lintcode.com/problem/3719/)

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
                    nei_r, nei_c = r + dx, c + dy
                    if 0 <= nei_r < rows and 0 <= nei_c < cols and (nei_r, nei_c) not in visited and grid[nei_r][nei_c] != "X":
                        queue.append((nei_r, nei_c))
                        visited.add((nei_r, nei_c))
            step += 1
        
        return -1
```
