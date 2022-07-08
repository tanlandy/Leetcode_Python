# 基础知识
## Terminology
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

## Implementation
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


# Binary Tree

## 模板
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

DFS








## 题目
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

[314. Binary Tree Vertical Order Traversal](https://leetcode.com/problems/binary-tree-vertical-order-traversal/)

```python
from collections import deque
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Queue存((node, col)), 用一个map{col, oneRes}
        遍历的时候，更新HashMap,最后用HashMap来导出，但是不知道最小值最大值，所以实时更新一下，这样就不用sort
        col_res=defaultlist(list); queue=deque([(root, 0)]); queue.append((node,col-1))

        Time: O(N),
        Space: O(N)
        """
        if root is None: # 不要忘了base case
            return []
        col_res = defaultdict(list) #When the list class is passed as the default_factory argument, then a defaultdict is created with the values that are list.
        min_col = max_col = 0
        queue = deque([(root, 0)]) # 初始化加是deque([(root, 0)])
        res = []
        while queue:
            node, col = queue.popleft()
            col_res[col].append(node.val)
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            if node.left:
                queue.append((node.left, col - 1)) # 双(())
            if node.right:
                queue.append((node.right, col + 1)) 

        for i in range(min_col, max_col + 1): # 左闭右开，需要加一
            res.append(col_res[i])
        return res

        """
        如果是直接colTable = {}
            if col not in colTable:
                colTable[col] = [node.val]
            else:
                colTable[col].append(node.val)
        """
```


[987. Vertical Order Traversal of a Binary Tree](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

```python
class Solution(object):
    def verticalTraversal(self, root):
        """
        与上一题唯一不同就是每一层新建一个map，然后排序好之后加到最终的map里；走完一层如何放进来：one_res[col] += sorted(temp[col])
        """    
        col_res = collections.defaultdict(list) 
        queue = collections.deque([(root, 0)])
        min_col, max_col = 0, 0
        while queue:
            tmp = collections.defaultdict(list)  # 每层之前先另外搞一个map
            for _ in range(len(queue)):
                node, col = queue.popleft()
                tmp[col].append(node.val) 
                min_col = min(min_col, col)
                max_col = max(max_col, col)

                if node.left:
                    queue.append((node.left, col - 1))
                if node.right: 
                    queue.append((node.right, col + 1)) 
                    
            for col in tmp: # 走完一层再把map按顺序加进去，不能用.append，否则某一层是[[3],[15]]
                col_res[col] += sorted(tmp[col])

        res = []
        for col in range(min_col, max_col + 1): # 左开右闭，需要加一
            res.append(col_res[col])
        return res
```

[310. Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/)



# BST
BST is often used to look up the existence of certain objects. Compared to sorted arrays, the insertion has way lower time complexity, so it's good for dynamic insertion of items. If you don't need to dynamically insert new items, then you can simply sort the collection first and use binary search to look up.

However, most modern languages offers hash tables, which is another way of looking up the existence of an object in a collection. Most implementations are dynamically sized, which can cause the lookup and insertion of items to approach O(1), so usually hash tables are preferred over BST. Nevertheless, there are some advantages to using a BST over a hash table.

Hash tables are unsorted, while BSTs are. If you want to constantly maintain a sorted order while inserting, using a BST is more efficient than a hash table or a sorted list.
It's easy to look up the first element in the BST that is greater/smaller than a lookup value than a hash table.
It's easy to find the k-th largest/smallest element.
Dynamic hash tables usually have a lot of unused memory in order to make the insertion/deletion time approach O(1), whereas BST uses all the memory they requested.



[510. Inorder Successor in BST II](https://leetcode.com/problems/inorder-successor-in-bst-ii/)
```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Optional[Node]':
        """
        往右找，如果右边没有，就是自己的父
        """
        # 往右找：右一步，然后一直往左
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            return node
        
        # 往上找：一直往左上，走到头之后再上一个就是successor
        while node.parent and node == node.parent.right:
            node = node.parent
        return node.parent
```



# 多叉树
