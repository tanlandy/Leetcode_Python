[79. Word Search](https://leetcode.com/problems/word-search/)
dfs(r, c, i)同时传入一个idx

时间：O(M*N*4 ^ N)
空间：O(L) L is len(words)
```py
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        rows = len(board)
        cols = len(board[0])
        visit = set()
        
        def dfs(r, c, i):
            # Base case
            if i == len(word):
                return True
            
            # 排除的条件
            if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in visit or board[r][c] != word[i]:
                return False
            
            # 做选择
            visit.add((r, c))
            # Backtrack
            res =  (dfs(r + 1, c, i + 1) or
                    dfs(r - 1, c, i + 1) or         
                    dfs(r, c + 1, i + 1) or         
                    dfs(r, c - 1, i + 1)
                  )
            # 回溯
            visit.remove((r, c))            
            return res
        
        for r in range(rows):
            for c in range(cols):
                if dfs(r, c, 0):
                    return True
        
        return False
```

Tree

[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
每走到一个节点，看一下他的两个子节点，然后swap子节点的位置

时间：O(N)
空间：O(N)
```py
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        
        # swap the children
        root.left, root.right = root.right, root.left
        
        self.invertTree(root.left)
        self.invertTree(root.right)
        
        return root
```        

[110. Balanced Binary Tree](https://www.youtube.com/watch?v=QfJsau0ItOY)
每走到一个节点，问左右子树是否balanced，再问左右子树是否balanced直到叶子节点，然后从下到上来看各个节点是否balance：先计算出来每个子节点的height，再比较是否balanced

时间：O(N)
空间：O(N)
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        
        def dfs(root):
            """
            Return T/F and its height
            """
            # base case
            if not root: # empty tree
                return [True, 0]
            
            left, right = dfs(root.left), dfs(root.right) # 看左右子树是否balance
            balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1 # 自己这个节点是否balance

            return [balanced, 1 + max(left[1], right[1])]]

        balan, height = dfs(root)
        return balan 
```

[100. Same Tree](https://www.youtube.com/watch?v=vRbbcKXCxOw)
每到一个节点，看这个节点的数是否相同

时间：O(p+q)
空间：P(p+q)

```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        
        if not p or not q:
            return False

        if p.val != q.val:
            return False
        
        return (self.isSameTree(p.left, q.left) and 
                self.isSameTree(p.right, q.right))
```


[572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)


时间：O(M*N) M is len(root), N is len(subRoot)
空间：O(M*N)
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """
        从左边树的每一个节点来看，这个节点对应的树是否和另一个树相同
        走每一个节点是用的isSubtree
        """        
        if not subRoot:
            return True
        
        if not root:
            return False

        # 先检查从这个点来看，对应的树是否和另一个数相同
        if self.sameTree(root, subRoot):
            reture True
        
        # 如果发现不同，那就往左右走
        return (self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot))

    def sameTree(self, p, q):
        if not p and not q:
            return True
        
        if not p or not q:
            return False
        
        if p.val != q.val:
            return False
        
        return (self.sameTree(p.left, q.left) and self.sameTree(p.right, q.right))
```


[235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

时间：O(logN) 每一层只用看一个点
空间：O(1)
```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        从root开始看(root肯定是一个common ancestor，但不一定是LCA)，如果p, q都比这个节点大，那就在右边找，如果都小在左边找，否则自己这个节点就是LCA
        
        """
        cur = root

        while cur:
            if p.val > cur.val and q.val > cur.val:
                cur = cur.right
            elif p.val < cur.val and q.val < cur.val:
                cur = cur.left
            else:
                return cur
```

[1448. Count Good Nodes in Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/)

时间：O(N)
空间：O(H)
```py
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        """
        Preorder遍历，每次往下走需要把当前的最大值传下去
        """        
        def dfs(node, max_val): # 判断这个节点是否是good node，返回个数
            if not node:
                return 0

            res = 1 if node.val >= max_val else 0
            max_val = max(node.val, max_val)
            res += dfs(node.left, max_val)
            res += dfs(node.right, max_val)
            return res
        
        return dfs(root, root.val)
```

[98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)


```py
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        因为是BST，所以不能只比较当前节点和两个子节点的大小，而是要把之前一层的值也传进来比较，所以需要一个helper function：往右走就update左边界，往左走就update右边界

        时间：O(N): visit each node exactly once
        空间：O(N): keep up to entire tree
        """
        def valid(node, left, right):
            """
            left, right分别是当前的左右边界
            一直走到底再return True，中间只关心是否return False
            """
            if not node:
                return True
            
            if not (left < node.val and node.val < right):
                return False
            
            return (valid(node.left, left, node.val) and valid(node.right, node.val, right))
        
        return valid(root, float("-inf"), float("inf"))
        
```

[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)


```py
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        """
        inorder iterativly遍历，这样一旦达到k就可以return
        当然也可以递归遍历，但就没有那么那么快了

        时间：O(H+K)
        空间：O(H) to keep the stack，对于BST平均时间就是O(logN)，最坏时间就是O(N)
        """
        n = 0
        stack = []
        cur = root

        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            n += 1
            if n == k:
                return cur.val
            cur = cur.right
            
```

[105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```py
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        Preorder的第一个是root，第二个数是左子树的root
        Inorder的root左边的值都在左子树，root右边的都是右子树
        时间：O(N)
        空间：O(N)
        """
        # base case
        if not preorder or not inorder:
            return None
        
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0]) #找到root在inorder的index

        # inorder的root左边都是左子树，右边都是右子树
        # preorder：根据左子树的数量，root之后[1:mid+1]左闭右开都是左子树，[mid+1:]都是右子树
        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid]) # 右开
        root.right = self.buildTree(preorder[mid+1:], inorder[mid:])

        return root
```

[124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

```py
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        res = [root.val]
        
        # return max path sum without split
        def dfs(root):
            if not root:
                return 0
            
            leftMax = dfs(root.left)
            rightMax = dfs(root.right)
            leftMax = max(leftMax, 0)
            rightMax = max(rightMax, 0)
            
            # compute max path sum WITH split
            res[0] = max(res[0], root.val + leftMax + rightMax)
            return root.val + max(leftMax, rightMax)
        
        dfs(root)
        return res[0]
```

# 2D Grid遍历

[130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)


```py
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        走三遍
        1. 先从边界，把所有延伸出来的O变成T
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


# 成环问题

[207. Course Schedule](https://leetcode.com/problems/course-schedule/)

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
            for pre in graph[course]: # 对于每个点，走到底
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

# 图遍历
[797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)

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

[785. Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)

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


[886. Possible Bipartition](https://leetcode.com/problems/possible-bipartition/)

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
