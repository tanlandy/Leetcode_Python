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
