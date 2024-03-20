# DFS

# 高频题

## 知乎

## Krahets精选题

## AlgoMonster

## Youtube

[200. 岛屿数量]
[236. 二叉树的最近公共祖先]
[297. 二叉树的序列化与反序列化]
[543. 二叉树的直径]
[733. 图像渲染]

# 例题

[543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

```py
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """
        每一条二叉树的「直径」长度，就是一个节点的左右子树的最大深度之和
        遇到子树问题，首先想到的是给函数设置返回值，然后在后序位置做文章
        
        Time: O(N)
        Space: O(N)
        """
        res = [0] # wrap in a list, to pass by reference
        def dfs(root): # 返回该节点最大深度
            if not root:
                return 0
            left = dfs(root.left) # 左子树最大深度
            right = dfs(root.right) # 右子树最大深度
            cur_max = left + right
            res[0] = max(res[0], cur_max)
            
            return 1 + max(left, right)
        
        dfs(root)
        return res[0]
```

[200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

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

[236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        解题思路：每个节点要知道什么、做什么：什么时候做
        遍历or递归
        要知道自己的子树里是否有这两个数字->递归
        要做什么：返回自己子树是否有这两个数字->递归
        什么时候做：后序遍历，传递子树信息

        自下而上，这个函数就返回自己左右子树满足条件的node：返回自己或者不为None的一边。base case就是找到了
        如果一个节点能够在它的左右子树中分别找到 p 和 q，则该节点为 LCA 节点。

        时间：O(N)
        空间：O(N)
        """
        if root is None: # base case 走到了根节点
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # 后序遍历
        if root == p or root == q: # Case 1：公共祖先就是我自己，也可以放在前序位置（要确保p,q在树中）
            return root
        
        if left and right: # Case 2：自己子树包含这两个数
            return root
        else:
            return left or right # Case 3：其中一个子树包含节点 
```

[297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/description/?envType=study-plan-v2&envId=selected-coding-interview)

```python
class Codec:

    def serialize(self, root):
        """
        用一个list记录，最后转为string导出：
        前序遍历，空节点计作N，然后用","连接
        """
        res = []
        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        return ",".join(res)

    def deserialize(self, data):
        """
        先确定根节点 root，然后遵循前序遍历的规则，递归生成左右子树
        """
        vals = data.split(",")
        self.i = 0
        
        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node
        
        return dfs()
```

[733. 图像渲染](https://leetcode.cn/problems/flood-fill/)

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
