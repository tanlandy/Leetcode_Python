# 基础知识
DFS:
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

## classify edges
1. Tree edges
2. Back edge: descendant going to ancestor
3. Foward edge: from ancestor to desendant
4. Cross edge: any edge btw subtrees or trees 


### Tree
DFS is essentially pre-order tree traversal.

- Traverse and find/create/modify/delete node
- Traverse with return value (finding max subtree, detect balanced tree)

### Think like a node
when you are a node, only things you know are:
1. your value
2. how to get to your children

### Defining the recursive function
two things needed
1. Return value (Passing value up from child to parent)
   - Ask what information we need at the current node to make a decision 
2. Identify states (Passing value down from parent to child)
   - what states do we need to maintain to compute the return value for the current node

### Pre-order, In-order, and Post-order
1. Pre-order: make the decision before looking at your children
2. Post-order: make the dicision after collecting information on children

### Combinatorial problems
DFS/backtracking and combinatorial problems are a match made in heaven (or silver bullet and werewolf 😅). As we will see in the Combinatorial Search module, combinatorial search problems boil down to searching in trees.

- How many ways are there to arrange something
- Find all possible combinations of ...
- Find all solutions to a puzzle

### Graph
Trees are special graphs that have no cycle. We can still use DFS in graphs with cycles. We just have to record the nodes we have visited and avoiding re-visiting them and going into an infinite loop.

- Find a path from point A to B
- Find connected components
- Detect cycles





# 题目
## Algo Monster
[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

```py
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_max_depth(root: Node) -> int:
    """
    1. Return value: return the depth for the current subtree after we visit a node
    2. Identify states: to decide the depth of current node, we only need depth from its children, don't need info from parents

    Time: O(N)
    Space: O(N)
    """
    if not root:
        return 0
    left_max = tree_max_depth(root.left)
    right_max = tree_max_depth(root.right)
    return max(left_max, right_max) + 1
    

def build_tree(nodes, f):
    val = next(nodes)
    if val == 'x': return None
    left = build_tree(nodes, f)
    right = build_tree(nodes, f)
    return Node(f(val), left, right)

tree = [5, 4, 3, "x", "x", 8, "x", "x", 6, "x", "x"]

if __name__ == '__main__':
    root = build_tree(iter(tree), int)
    res = tree_max_depth(root)
    print(res)
```

Iterative Solution:
```py
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        stack = [[root, 1]]
        res = 1
        
        while stack:
            node, depth = stack.pop()
            if node:
                res = max(res, depth)
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])
        
        return res
```

[1448. Count Good Nodes in Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/)
```py
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        """
        只需要遍历一遍
        """
        res = [0]
        def dfs(root, path_max):
            if not root:
                return
            if root.val >= path_max:
                res[0] += 1
            path_max = max(root.val, path_max)
            dfs(root.left, path_max)
            dfs(root.right, path_max)
        dfs(root, float("-inf"))
        return res[0]
```

[110. Balanced Binary Tree](https://www.youtube.com/watch?v=QfJsau0ItOY)

```py
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """
        站在每个节点：知道两边子树的高度差，并比较；
        返回什么：要返回当前节点的高度
        -> 后序遍历，返回当前高度

        时间：O(N)
        空间：O(N)
        """
        def node_height(node):
            if not node:
                return 0
            left_h = node_height(node.left)
            right_h = node.height(node.right)

            if left_h == -1 or right_h == -1:
                return -1

            if abs(left_h - right_h) > 1:
                return -1
            
            return 1 + max(left_h, right_h)
        
        return node_height(root) != -1

```







## 算法笔记

### 解题过程
首先思考二叉树的每一个节点需要做什么，需要在什么时候（前中后序）做。
接下来二选一：
1、是否可以通过遍历一遍二叉树得到答案？如果可以，用一个 traverse 函数配合外部变量来实现。

2、是否可以定义一个递归函数，通过子问题（子树）的答案推导出原问题的答案？如果可以，写出这个递归函数的定义，并充分利用这个函数的返回值。
    - 一旦你发现题目和子树有关，那大概率要给函数设置合理的定义和返回值，在后序位置写代码了 -> 后序位置才能收到子树的信息


### 例题


#### Tree
[543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
```py
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """
        每一条二叉树的「直径」长度，就是一个节点的左右子树的最大深度之和
        遇到子树问题，首先想到的是给函数设置返回值，然后在后序位置做文章
        
        Time: O(N)
        Space: O(N)
        """
        res = [0]
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


[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
```py
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        traversely solve: 前序遍历每个节点，每个节点交换左右子节点
        """
        
        def traverse(root):
            if not root:
                return
            root.left, root.right = root.right, root.left
            traverse(root.left)
            traverse(root.right)
            
        traverse(root)
        return root
```

```py
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        recursively solve: 分治：定义一个递归函数，通过子问题（子树）的答案推导出原问题的答案
        invertTree(): 将以 root 为根的这棵二叉树翻转，返回翻转后的二叉树的根节点
        """
        if not root:
            return
        
        left = self.invertTree(root.left) # 把root的左子树反转
        right = self.invertTree(root.right) # 把root的右子树反转
        
        root.left, root.right = root.right, root.left # 反转root自己的左右子树
        
        return root
```

[116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
```py
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        """
        遍历的方法：前序遍历每个节点，每个节点给连接到右边
        """
        if not root:
            return
        
        def traverse(node1, node2):
            if not node1 or not node2:
                return
            node1.next = node2
            
            traverse(node1.left, node1.right) # 连接相同父节点的
            traverse(node2.left, node2.right)
            traverse(node1.right, node2.left) # 连接不同父节点的
        
        traverse(root.left, root.right)
        return root
```            


```py
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        """
        level order traversal
        when i < size - 1, connect the node with the next one

        Time: O(N)
        Space: O(N)
        """
        if not root:
            return
        queue = collections.deque([root])
        
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                if i < size - 1:
                    node.next = queue[0]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return root
```

```py
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        """
        Use previously established next pointers
        站在n层来连接n+1层的节点
        
        Time: O(N)
        Space: O(1)
        """
        if not root:
            return root
        
        left_most = root
        
        while left_most.left:
            node = left_most
            while node: # 把这一层走完
                # connection 1
                node.left.next = node.right
                
                # connection 2
                if node.next:
                    node.right.next = node.next.left
                
                node = node.next # 把这一层走完
                
            left_most = left_most.left # 这一层结束，走下一层
        
        return root
```

[654. Maximum Binary Tree](https://leetcode.com/problems/maximum-binary-tree/)

```py
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """
        分解：这个函数就返回构造好的树的根节点，对于每个节点，只需要找到当前范围的最大值，然后构造根节点，最后把左右接上
        
        Time: O(NLogN)
        Space: O(N)
        """
        if not nums:
            return None
        
        max_val = max(nums)
        max_idx = nums.index(max_val)
        pivot = TreeNode(max_val)
        pivot.left = self.constructMaximumBinaryTree(nums[: max_idx])       
        pivot.right = self.constructMaximumBinaryTree(nums[max_idx + 1:])
        
        return pivot
```

```py
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """
        单调递减栈：[]存node；对新node，如果比栈顶大，就一直pop，同时把pop的节点变成其左子节点，直到找到比他大的节点；如果比栈顶小，那栈顶的右子节点就暂时是他
        
        Time: O(N)
        Space: O(N)
        """
        
        stack = []
        
        for val in nums:
            node = TreeNode(val)
            
            # 新node如果比栈顶大，就一直pop，同时pop的节点是新node的左子节点
            while stack and stack[-1].val < val:
                node.left = stack.pop()
            
            # 新node如果比栈顶小，就是栈顶node的右子节点；直到之后遇到更大的节点，就会被pop出来变成更大节点的左子节点
            if stack:
                stack[-1].right = node
            
            # 放入新node，栈内node的值单调递减
            stack.append(node)
        
        return stack[0]
```
[105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

先找到根节点位置，然后分治左右

![前序中序遍历效果图](https://labuladong.github.io/algo/images/%E4%BA%8C%E5%8F%89%E6%A0%91%E7%B3%BB%E5%88%972/1.jpeg)


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

[106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

先找到根节点位置，然后分治左右

![后序中序遍历效果图](https://labuladong.github.io/algo/images/%E4%BA%8C%E5%8F%89%E6%A0%91%E7%B3%BB%E5%88%972/5.jpeg)

```py
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """
        通过postorder找到root的大小
        然后通过inorder的root位置来确定接下来的范围
        """
        if not inorder or not postorder:
            return None
        
        root = TreeNode(postorder[-1])
        pivot = inorder.index(root.val)
        
        root.left = self.buildTree(inorder[:pivot], postorder[:pivot])
        root.right = self.buildTree(inorder[pivot + 1:], postorder[pivot: -1])
        
        return root
```

[889. Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

```py
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """
        通过preorder[0]找到root大小
        postorder倒数第二个数，是root的右子树
        利用这个值在preorder的位置，从而确定接下来的范围
        """
        if not preorder or not postorder:
            return None
        
        root = TreeNode(preorder[0])

        # 因为用到了postorder[-2]，所以要检查一下长度
        if len(postorder) == 1:
            return root
        # The second to last of "post" should be the value of right child of the root.
        idx = preorder.index(postorder[-2])
        root.left = self.constructFromPrePost(preorder[1:idx], postorder[: idx - 1])
        root.right = self.constructFromPrePost(preorder[idx:], postorder[idx-1:-1])
        
        return root


```


[297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

preorder遍历

```python
class Codec:

    def serialize(self, root):
        """
        用一个list记录，最后转为string导出：前序遍历，空节点计作N，然后用,连接
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
        转化为list，然后用i遍历
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

```py

class Codec:

    def serialize(self, root):
        """
        用queue，把root添加进来，如果有值就加入res[]，并且更新左右子树，如果是空就跳过。
        """
        if not root:
            return ""
        res = []
        queue = collections.deque()
        queue.append(root)
        while queue:
            node = queue.popleft()
            if not node:
                res.append("N")
                continue
            res.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        
        return ",".join(res)
        

    def deserialize(self, data):
        if not data:
            return None
        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        queue = collections.deque()
        queue.append(root)
        i = 1
        while queue and i < len(vals):
            node = queue.popleft()
            if vals[i] != "N":
                left = TreeNode(int(vals[i]))
                node.left = left
                queue.append(left)
            i += 1
            if vals[i] != "N":
                right = TreeNode(int(vals[i]))
                node.right = right
                queue.append(right)
            i += 1
        
        return root
```

[652. Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees/)

每个节点要做什么：
1. 我为根的子树什么样子 -> 后序遍历，返回一个序列化字符串
2. 其他节点为根的子树什么样子 -> 用HashMap，存子树及其数量

```py
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        """
        对每个节点，序列化得到以自己为根的子树，并且添加到hashmap中
        hashmap{tree:count}，当count==2导出
        """
        counter = collections.defaultdict(int)
        res = []
        
        # dfs(node)返回以node为根的序列化子树
        def dfs(node):
            if not node:
                return "N"
            left = dfs(node.left)
            right = dfs(node.right)
            tree = str(node.val) + "," + left + "," + right
            
            counter[tree] += 1
            if counter[tree] == 2: # 导出这棵树
                res.append(node)
            
            return tree
        
        dfs(root)
        return res
```
#### BST

基本性质：对于每个node，左子树节点的值都更小，右子树节点的值都更大；中序遍历结果是有序的


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

[538. Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/)
```py
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        倒过来的中序遍历，形成递减排列的数组，计算出来大小并加进来
        """
        if not root:
            return root
        
        suf_sum = [0]
        
        def inorder(node):
            if not node:
                return
            
            inorder(node.right)
            suf_sum[0] += node.val
            node.val = suf_sum[0]
            inorder(node.left)
        
        inorder(root)
        return root
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

[700. Search in a Binary Search Tree](https://leetcode.com/problems/search-in-a-binary-search-tree/)
```py
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        
        while root is not None and root.val != val:
            if root.val < val:
                root = root.right
            else:
                root = root.left
        
        return root # 要么是找到了，要么是就没有
```

[701. Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
```py
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """
        如果加入的值更大，就一直往右走，直到走到空，就作为最大值的右边的值
        """
        node = root
        while node:
            if val > node.val:
                if not node.right:
                    node.right = TreeNode(val)
                    return root
                else:
                    node = node.right
            else:
                if not node.left:
                    node.left = TreeNode(val)
                    return root
                else:
                    node = node.left
        
        return TreeNode(val)
```

[450. Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/)
```py
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """
        一共三种情况：key是叶子，key只有一个node，key有2个nodes
        如果有2个nodes，就找到key的successor并相互交换，然后删掉successor就可以了

        Time: O(H)
        Space: O(H)
        """
        if not root:
            return None
        
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            # has single node, or is a leaf
            if not root.right:
                return root.left
            if not root.left:
                return root.right
            # has two nodes
            if root.left and root.right:
                # find the successor, replace the root with successor, delete the duplicate successor
                successor = root.right
                while successor.left:
                    successor = successor.left # found the successor
                root.val = successor.val # replace
                root.right = self.deleteNode(root.right, successor.val) # delete
        
        return root
```

[222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)
```py
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        l = r = root
        l_height = r_height = 0
        
        # 沿着最左侧和最右侧计算高度
        while l:
            l = l.left
            l_height += 1
        
        while r:
            r = r.right
            r_height += 1
        
        if l_height == r_height: # Perfect Binary Tree
            return 2 ** l_height - 1
        
        # 高度不同，那么普通二叉树的逻辑计算
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```

#### 公共祖先
[236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

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
        if root is None: # base case
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # 后序遍历
        if root == p or root == q: # Case 1：公共祖先就是我自己，也可以放在前序位置（要确保p,q在树中）
            return root
        
        if left and right: # Case 2：自己子树包含这两个数
            return root
        else:
            return left or right # Case 3：
```

[1676. Lowest Common Ancestor of a Binary Tree IV](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iv/)

```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
        """
        和LC236类似，只是检查公共祖先就是我自己的时候有变化

        Time: O(N)
        Space: O(N)
        """
        nodes_set = set(nodes)
        
        def dfs(node):
            if node is None:
                return None
            if node in nodes_set: # 可在前序位置，也可后
                return node
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            if left and right:
                return node
            else:
                return left or right
        
        return dfs(root)
```


[1644. Lowest Common Ancestor of a Binary Tree II](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/)

```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        Node不一定在树里，当root == p or root == q必须后序遍历判断
        dfs() 除了返回当前节点外，还返回是否存在
        
        Time: O(N)
        Space: O(N)
        """
        def dfs(root):
            if not root:
                return None, False
            
            left, l_exist = dfs(root.left)
            right, r_exist = dfs(root.right)
            
            if root == p or root == q:
                return root, left or right
            
            if left and right:
                return root, True
            else:
                if left:
                    return left, l_exist
                else:
                    return right, r_exist
                 
        lca, exist = dfs(root)
        if not exist:
            return None
        return lca
```


[235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        不需要去遍历子树，由于 BST 左小右大的性质，将当前节点的值与 val1 和 val2 作对比即可判断当前节点是不是 LCA

        Time: O(H)
        Space: O(1)
        """
        cur = root
        
        while cur:
            # cur太小就往右
            if p.val > cur.val and q.val > cur.val:
                cur = cur.right
            # cur太大就往左
            elif p.val < cur.val and q.val < cur.val:
                cur = cur.left
            else: # p.val <= cur.val <= q.val
                return cur
```

[1650. Lowest Common Ancestor of a Binary Tree III](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/)

```python
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        """
        先求各自深度，再把深的往上走直到当前深度相同，最后一起往上走找parent；注意找深度是while p；深度就是层数root的深度是1

        时间：O(H)
        空间：O(1)
        """

        # get_depth(p) 返回节点p的深度
        def get_depth(p):
            depth = 0
            while p:
                p = p.parent
                depth += 1
            return depth
    
        d1 = get_depth(p)
        d2 = get_depth(q)
        
        # 把更深的往上走，直到相同深度
        while d1 > d2:
            p = p.parent
            d1 -= 1
                
        while d1 < d2:
            q = q.parent
            d2 -= 1
        
        # 现在在相同深度，一起往上走找LCA
        while p != q:
            p = p.parent
            q = q.parent
        
        return p      
```

# Tree

[101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

```py
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        对于每个节点来说：看自己的左右节点是否对称，看自己的子树是否对称->返回自己是否满足对称
        遍历一遍不可以，需要知道自己的子节点是否对称这一信息
        -> 递归，同时看两个节点，然后左左右右，左右右左看对称
        """
        if not root:
            return True
        
        def dfs(left, right):
            if not left and not right:
                return True
            
            if not left or not right:
                return False
                
            if left.val == right.val:
                return dfs(left.left, right.right) and dfs(left.right , right.left)
            else:
                return False
        
        return dfs(root, root)
```

```py
class Solution:
    def isSymmetric(self, root):
        if not root:
            return True
        
        queue = collections.deque([root.left, root.right])
      
        while queue:
            t1, t2 = queue.popleft(), queue.popleft()

            if not t1 and not t2:
                continue
            elif (not t1 or not t2) or (t1.val != t2.val):
                return False
            
            queue.extend([t1.left, t2.right, t1.right, t2.left])
        return True
```

[951. Flip Equivalent Binary Trees](https://leetcode.com/problems/flip-equivalent-binary-trees/)

```py
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        """
        在每个节点：看左右子树是否相同，相同就继续往下看，不相同就flip一下再检查是否相同，如果还不相同说明不满足条件
        节点需要告诉父节点自己是否满足条件 -> 递归

        Time: O(min(N1, N2))
        Space: O(min(N1, N2))
        """
        def dfs(node1, node2):
            if not node1 and not node2:
                return True
            if not node1 or not node2:
                return False
            
            if node1.val != node2.val:
                return False
            
            return (dfs(node1.left, node2.left) and dfs(node1.right, node2.right)) or 
                    (dfs(node1.left, node2.right) and dfs(node1.right, node2.left))
                
        
        return dfs(root1, root2)
```


[572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)

```py
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """
        从左边树的每一个节点来看，这个节点对应的树是否和另一个树相同
        走每一个节点是用的isSubtree
        
        时间：O(M*N) M is len(root), N is len(subRoot)
        空间：O(M*N)
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

[863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)


[1110. Delete Nodes And Return Forest](https://leetcode.com/problems/delete-nodes-and-return-forest/)

```py
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        """
        在这个节点：如果父节点被删了，那就要添加到res。
        遍历的同时，把要删除的节点变成None
        """
        to_delete = set(to_delete)
        res = []
        
        def dfs(root, parent_exist):
            if not root:
                return None
            
            if root.val in to_delete:
                root.left = dfs(root.left, False)
                root.right = dfs(root.right, False)
                return None # root.left/right = None这个节点删除掉了
            else:
                if not parent_exist:
                    res.append(root)
                root.left = dfs(root.left, True)
                root.right = dfs(root.right, True)
                return root
        
        dfs(root, False)
        return res
```

[270. Closest Binary Search Tree Value](https://leetcode.com/problems/closest-binary-search-tree-value/)

```py
class Solution:
    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        """
        站在每个节点：更新res，然后往左或者右走
        不需要知道子树信息：直接中序遍历
        
        Time: O(H)
        Space: O(H)
        """
        res = [root.val]
        def dfs(node):
            if not node:
                return
            
            if abs(node.val - target) < abs(res[0] - target):
                res[0] = node.val
            
            if target < node.val:
                dfs(node.left)
            else:
                dfs(node.right)
        
        dfs(root)
        return res[0]
```

[669. Trim a Binary Search Tree](https://leetcode.com/problems/trim-a-binary-search-tree/)

```py
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        """
        站在一个节点：通过自己和range的大小比较，知道自己是否要被trim。还要只要子树是否被trim
        需要子树信息->递归->返回满足条件的root
        
        Time: O(N)
        Space: O(H)
        """
        
        
        # trim()返回满足条件的root
        def trim(node):
            if not node:
                return None
            
            if node.val > high: # 自己要被trim，返回比自己小的左子树
                return trim(node.left)
            elif node.val < low:
                return trim(node.right)
            else:
                node.left = trim(node.left) # 左边接上满足条件的节点
                node.right = trim(node.right) # 右边接上满足条件的节点
                return node
        
        return trim(root)
```

[333. Largest BST Subtree](https://leetcode.com/problems/largest-bst-subtree/)













































[100. Same Tree](https://www.youtube.com/watch?v=vRbbcKXCxOw)
每到一个节点，看这个节点的数是否相同

时间：O(p+q)
空间：P(p+q)

```py
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



[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) 上面有





[124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

```py
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        """
        站在每个节点：更新最大sum，返回从这个节点开始的max path sum
        更新最大sum: 自己 + 左节点max path sum + 右节点max path sum

        Time: O(N)
        Space: O(H)
        """
        res = [root.val]
        
        # return max path sum without split
        def dfs(root):
            if not root:
                return 0
            
            leftMax = dfs(root.left)
            rightMax = dfs(root.right)
            leftMax = max(leftMax, 0) # for those negative, doesn't need to look
            rightMax = max(rightMax, 0)
            
            # compute max path sum WITH split
            res[0] = max(res[0], root.val + leftMax + rightMax)
            return root.val + max(leftMax, rightMax)
        
        dfs(root)
        return res[0]
```

[113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)
```py
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """
        execute the dfs and maintain the running sum of node traversed and the list of those nodes
        
        Time: O(N^2)
        Space: O(N)
        """
        res = []
        def dfs(node, cur_sum, cur_path):
            if not node:
                return
            cur_sum += node.val
            cur_path.append(node.val)
            
            if cur_sum == targetSum and not node.left and not node.right: # 同时满足大小和位置关系
                res.append(cur_path.copy())
            else:
                dfs(node.left, cur_sum, cur_path)
                dfs(node.right, cur_sum, cur_path)
            
            cur_path.pop()
        
        dfs(root, 0, [])
        return res
```

多叉树的遍历
```py
class Node:
    def __init__(self, val, children=None):
        if children is None:
            children = []
        self.val = val
        self.children = children

def ternary_tree_paths(root: Node) -> List[str]:
    res = []
    def dfs(node, cur_path):
        if not node:
            return 
        cur_path.append(str(node.val)) 
        # cur_path.append(node.val)       
        
        if all(c is None for c in node.children):
            res.append("->".join(cur_path))
            # res.append(cur_path.copy())
            cur_path.pop()
            return
        for c in node.children:
            if c is not None:
                dfs(c, cur_path)
        cur_path.pop()
              
        
    dfs(root, [])
    return res
```


## other

[1367. Linked List in Binary Tree](https://leetcode.com/problems/linked-list-in-binary-tree/)
```py
class Solution:
    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        """
        遍历：遍历这棵树，看该节点是否和head相同，再在满足的每个节点遍历剩下的值，看能否嵌入链表
        
        Time: O(N * min(N, H))
        Space: O(H)
        """
        if not head:
            return True
        if not root:
            return False
        
        if head.val == root.val:
            if self.checkPath(head, root):
                return True
        
        return self.isSubPath(head, root.left) or self.isSubPath(head, root.right)

    def checkPath(self, head, root):
        if not head:
            return True
        if not root:
            return False
        
        if head.val == root.val:
            return self.checkPath(head.next, root.left) or self.checkPath(head.next, root.right)
        
        return False
```

# DFS + Memorization




[91. Decode Ways](https://leetcode.com/problems/decode-ways/)

```py
class Solution:
    def numDecodings(self, s: str) -> int:
        """
        BF: when str has more than two digits: draw a desicion tree
        Example: "121" can only branch to 1-26 -> O(2^N)
                 121
             /          \
            1            12
          /   \         /
         2    21       1
        /
        1

        subproblem: once solve 21, the subproblem is 1, solve from right to left
        dp[i] = dp[i + 1] + dp[i + 2]

        Time: O(N)
        Space: O(N), O(1) if only use two variables
        """
        dp = [1] * (len(s) + 1)

        for i in range(len(s) - 1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]

            if ((i + 1) < len(s)) and ((s[i] == "1") or s[i] == "2" and s[i + 1] in "0123456"): # double digit
            # if 10 <= int(s[i:i+2]) <= 26:
                dp[i] += dp[i + 2]
        
        return dp[0]
```

```py
class Solution:
    def numDecodings(self, s: str) -> int:
        """
        Time: O(N)
        Space: O(N)
        """
        memo = {}
        
        def dfs(idx):
            if idx in memo:
                return memo[idx]
            
            # 走到头了
            if idx == len(s):
                return 1
            
            # 这个string以0开头
            if s[idx] == "0":
                return 0
            
            # 走到前一位：只有1种方式了
            if idx == len(s) - 1:
                return 1
            
            res = dfs(idx + 1)
            if int(s[idx: idx + 2]) <= 26:
                res += dfs(idx + 2)
            
            memo[idx] = res       
                 
            return res
        
        return dfs(0)        
```



