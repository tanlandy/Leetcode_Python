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