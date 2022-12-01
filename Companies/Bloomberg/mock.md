# 09/28/22
## BQ
tell me about yourself
talk more about Jenkins

## Coding
via codeshare

given the root of a complete binary tree, return the number of the nodes in the tree
design an algorithm that runs in less than O(n) time complexity


[222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)

```py
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        l = r = root
        l_length, r_length = 0, 0
        
        while l:
            l = l.left
            l_length += 1
        
        while r:
            r = r.right
            r_length += 1
        
        if l_length == r_length:
            return 2 ** l_length - 1
        
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```