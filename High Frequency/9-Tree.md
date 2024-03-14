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

## Think like a node

two things needed to think of when writing dfs()

1. Return value (Passing value up from child to parent)
   - Ask what information we need at the current node to make a decision
2. Identify states (Passing value down from parent to child)
   - what states do we need to maintain to compute the return value for the current node

## 解题过程

首先思考二叉树的每一个节点需要做什么，需要在什么时候（前中后序）做。
作为一个node，只知道两件事：

1. 自己的值
2. 到自己子节点的方式

接下来二选一：

1. 是否可以通过遍历一遍二叉树得到答案？如果可以，用一个 traverse 函数配合外部变量来实现。
2. 是否可以定义一个递归函数，通过子问题（子树）的答案推导出原问题的答案？如果可以，写出这个递归函数的定，思考
   - return value: 站在这个节点，需要返回给父节点什么信息
   - states: 站在这个节点，需要父节点提供什么信息来做决策计算
   - （如果发现题目和子树有关，那大概率要给函数设置合理的定义和返回值，在后序位置写代码了 -> 后序位置才能收到子树的信息）

## 模板

queue放元素：queue = deque([root])

```python
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order_traversal(root: Node) -> List[List[int]]:
    res = [] 
    if root is None:
        return res
    
    queue = deque([root])
    
    while queue:
        oneRes = []
        size = len(queue)
        for _ in range(size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            oneRes.append(node.val)
        res.append(oneRes)
                      
    return res
```

### Pre-order, In-order, and Post-order

1. Pre-order: make the decision before looking at your children
2. Post-order: make the dicision after collecting information on children

遍历

```py



```

## 题目

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

[103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

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

[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

```py
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        分治：从下到上返回当前节点的最大深度
        """
        if not root:
            return 0
        left_max = self.maxDepth(root.left)
        right_max = self.maxDepth(root.right)
        
        return 1 + max(left_max, right_max)
```

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

[199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)

```py
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """
        dfs遍历的时候同时传进来level，每次level+1，先看right再看left，添加的条件就是level == len(res)，这样就总是先加入右边的点了
        """
        if not root:
            return []
        
        res = []
        
        def dfs(node, level):
            if not node: # dfs总是不要忘记base case
                return
            if level == len(res):
                res.append(node.val)
            if node.right:
                dfs(node.right, level + 1) # 如果先加入左边，就是左视图
            if node.left:
                dfs(node.left, level + 1)
        
        dfs(root, 0)
        return res
```

```py
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        queue = collections.deque([root])
        res = []
        
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                if i == size - 1:
                    res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return res
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

[105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

先找到根节点位置，然后分治左右

![前序中序遍历效果图](https://labuladong.github.io/algo/images/%E4%BA%8C%E5%8F%89%E6%A0%91%E7%B3%BB%E5%88%972/1.jpeg)

```py
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
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
        mid = inorder.index(root.val) # 找到root在inorder的index

        # preorder：根据左子树的数量，root之后[1:mid+1]左闭右开都是左子树，[mid+1:]都是右子树
        # inorder的root左边都是左子树，右边都是右子树   
        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid]) # 右开
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])

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

[298. Binary Tree Longest Consecutive Sequence](https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/)

```py
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        """
        在每个节点，需要和上个节点比较大小，同时更新length和最终的res。因为不需要之后自己自节点的情况，所以不需要分治，直接遍历就可以了
        
        时间：O(N)
        空间：O(N)
        """
        
        res = [0]
        
        def dfs(node, parent, length):
            if not node:
                return
            
            if parent and node.val == parent.val + 1:
                length += 1
            else:
                length = 1
            
            res[0] = max(length, res[0])
            
            dfs(node.left, node, length)
            dfs(node.right, node, length)            
        
        dfs(root, None, 0)
        return res[0]
```

[1130. Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)

[1485. Clone Binary Tree With Random Pointer](https://leetcode.com/problems/clone-binary-tree-with-random-pointer/)

```py
class Solution:
    def copyRandomBinaryTree(self, root: 'Node') -> 'NodeCopy':
        nodeArr = {}

        def dfs(root):
            if not root: 
                return None
            if root in nodeArr: 
                return nodeArr[root]
            nRoot = NodeCopy(root.val)
            nodeArr[root] = nRoot
            nRoot.left = dfs(root.left)
            nRoot.right = dfs(root.right)
            nRoot.random = dfs(root.random)
            return nRoot

        return dfs(root)
```

[863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)

```py
class Solution:
    def distanceK(self, root, target, K):
        """
        Convert to a graph, then dfs

        """
        adj, res, visited = collections.defaultdict(list), [], set()
        def dfs(node):
            if node.left:
                adj[node].append(node.left)
                adj[node.left].append(node)
                dfs(node.left)
            if node.right:
                adj[node].append(node.right)
                adj[node.right].append(node)
                dfs(node.right)
        dfs(root)
        def dfs2(node, d):
            if d < K:
                visited.add(node)
                for v in adj[node]:
                    if v not in visited:
                        dfs2(v, d + 1)
            else:
                res.append(node.val)
        dfs2(target, 0)
        return res
```

[114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)
LC官方讲解好

```py
class Solution:
    
    def flattenTree(self, node):
        
        # Handle the null scenario
        if not node:
            return None
        
        # For a leaf node, we simply return the
        # node as is.
        if not node.left and not node.right:
            return node
        
        # Recursively flatten the left subtree
        leftTail = self.flattenTree(node.left)
        
        # Recursively flatten the right subtree
        rightTail = self.flattenTree(node.right)
        
        # If there was a left subtree, we shuffle the connections
        # around so that there is nothing on the left side
        # anymore.
        if leftTail:
            leftTail.right = node.right
            node.right = node.left
            node.left = None
        
        # We need to return the "rightmost" node after we are
        # done wiring the new connections. 
        return rightTail if rightTail else leftTail
    
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        
        self.flattenTree(root)
```

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
            
            if not left or not right or left.val != right.val:
                return False
            
            return dfs(left.right, right.left) and dfs(left.left, right.right)
        
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

[100. Same Tree](https://www.youtube.com/watch?v=vRbbcKXCxOw)

```py
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """
        每到一个节点，看这个节点的数是否相同

        时间：O(p+q)
        空间：P(p+q)
        """
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

[1457. Pseudo-Palindromic Paths in a Binary Tree](https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/)

```py
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        """
        最普通的方法，记录root-leaf的路径，到达leaf之后计算元素的奇偶
        """
        
        res = [0]
        
        def dfs(node, path):
            if not node:
                return
            
            path[node.val] += 1
            
            if not node.left and not node.right: # reach the leaf
                odd = even = 0
                for val in path.values():
                    if val % 2 == 1:
                        odd += 1
                    else:
                        even += 1
                if odd <= 1:
                    res[0] += 1
                return
            dfs(node.left, path.copy()) # 需要用path.copy()来分割
            dfs(node.right, path.copy())
            
        path = collections.defaultdict(int)
        
        dfs(root, path)
        return res[0]
```

```py
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        """
        更好的方法，用set来解决奇偶性：第二次见到就删掉，第一次见到就加进来
        """
        
        res = [0]
        
        def dfs(node, path):
            
            if not node:
                return
            
            if node.val in path:
                path.remove(node.val)
            else:
                path.add(node.val)
            
            if not node.left and not node.right: # reach the leaf
                if len(path) <= 1:
                    res[0] += 1
                return
            
            dfs(node.left, path.copy())
            dfs(node.right, path.copy())
            
        path = set()
        
        dfs(root, path)
        return res[0]
```

```py
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        """
        O(1) space complexity solution
        """
        
        res = [0]
        
        def dfs(node, path):
            if not node:
                return
            
            path = path ^ (1 << node.val) # left shift operator to define the bit. XOR to compute the digit frequency
            
            if not node.left and not node.right: # reach the leaf
                if path & (path - 1) == 0: # path & (path - 1) set the rightmost 1 to 0, if is equals 0, means there's only one 1 in path
                    res[0] += 1
                return
            
            dfs(node.left, path)
            dfs(node.right, path)
        
        dfs(root, 0)
        return res[0]
```

[872. Leaf-Similar Trees](https://leetcode.com/problems/leaf-similar-trees/description/)

```py
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def find_leaf(node, nodes):
            if not node:
                return
            if not node.left and not node.right:
                nodes.append(node.val)
            find_leaf(node.left, nodes)
            find_leaf(node.right, nodes)
        
        leaf1 = []
        leaf2 = []
        find_leaf(root1, leaf1)
        find_leaf(root2, leaf2)

        return leaf1 == leaf2
```

### Path系列

[257. Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)

```py
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """
        string manipulation: "1" + "->" + str(node.val)
        """
        res = []
        
        def dfs(node, cur_path):
            if not node:
                return
            
            cur_path += str(node.val)
            
            if not node.left and not node.right:
                res.append(cur_path)
            
            dfs(node.left, cur_path + "->")
            dfs(node.right, cur_path + "->")
        
        dfs(root, "")
        
        return res
```

[988. Smallest String Starting From Leaf](https://leetcode.com/problems/smallest-string-starting-from-leaf/)

```py
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        res = [None]

        def dfs(node, cur):
            if not node:
                return
            
            cur = cur + chr(ord("a") + node.val) # ord("a") returns ASCII of a, chr(97) return "a" from ASCII
            
            if not node.left and not node.right:
                if res[0] == None:
                    res[0] = cur[::-1] # in a reversed way
                else:
                    res[0] = min(cur[::-1], res[0]) # compare strings
            
            dfs(node.left, cur)
            dfs(node.right, cur)
            
        dfs(root, "")
        return res[0]
```

[1022. Sum of Root To Leaf Binary Numbers](https://leetcode.com/problems/sum-of-root-to-leaf-binary-numbers/)

```py
class Solution:
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        """
        int(value, base [optional]) return an integer representation of a number with a given base
        """
        
        res = [0]
        
        def dfs(node, num):
            if not node:
                return
            
            num = num * 10 + node.val
            
            if not node.left and not node.right:
                res[0] += int(str(num), 2) # (value, base) where value should be str
            
            dfs(node.left, num)
            dfs(node.right, num)
        
        dfs(root, 0)
        return res[0]
```

[112. Path Sum](https://leetcode.com/problems/path-sum/)

```py
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """
        遍历时候更新targetSum的值，返回这个点是否满足要求
        """
        if not root:
            return False
        
        targetSum -= root.val
        
        if not root.left and not root.right: # reach a leaf
            return targetSum == 0 # only check when reaches a leaf
        
        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)
```

[113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)

```py
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """
        execute the dfs and maintain the running sum of node traversed and the list of those nodes
        
        Time: O(N^2)
        Space: O(N)
        需要注意，如果是问root to node的话，就不需要满足位置关系。
        如果不全是positive value的话，不能提前break，一定要找到底
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

[437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)

```py
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
        Time: O(N^2)
        """
        res = [0]
        
        def dfs(node, cur_sum):
            if not node:
                return
            
            cur_sum += node.val
            if cur_sum == targetSum:
                res[0] += 1
                
            dfs(node.left, cur_sum)
            dfs(node.right, cur_sum)
        
        def d(node): # traverse the tree
            if not node:
                return
            dfs(node, 0) # dfs each node
            d(node.left)
            d(node.right)
        
        d(root)
        return res[0]
```

更好的方法Prefix_sum

```py
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
        remove the last cur_sum from dict before processing the parallel subtree
        """
        prefix_freq = collections.defaultdict(int)
        res = [0]
        def dfs(node, cur_sum): # cur_sum is the prefix_sum from the previous node
            if not node:
                return
            
            cur_sum += node.val
            
            if cur_sum == targetSum:
                res[0] += 1
            res[0] += prefix_freq[cur_sum - targetSum]
            prefix_freq[cur_sum] += 1
            
            dfs(node.left, cur_sum)
            dfs(node.right, cur_sum)
            prefix_freq[cur_sum] -= 1
        
        dfs(root, 0)
        return res[0]
```

[129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)

```py
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        res = [0]
        
        def dfs(node, cur_sum):
            if not node:
                return
            cur_sum = 10 * cur_sum + node.val
            
            if not node.left and not node.right:
                res[0] += cur_sum
            dfs(node.left, cur_sum)
            dfs(node.right, cur_sum)
        
        dfs(root, 0)
        
        return res[0]
```

[687. Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/)

```py
class Solution:
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        res = [0]
        
        def dfs(node): # 返回node为根且满足大小相同的最大深度
            if not node:
                return 0
            
            left_len = dfs(node.left)
            right_len = dfs(node.right)
            left_a = right_a = 0
            
            if node.left and node.left.val == node.val:
                left_a = left_len + 1 # 满足条件的深度 += 1
            if node.right and node.right.val == node.val:
                right_a = right_len + 1
            res[0] = max(res[0], left_a + right_a) # node为根的longest path
            
            return max(left_a, right_a) # 大小相同
    
        dfs(root)
        return res[0]
```

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
        res = [root.val] # may have negative number, so add root.val at the beginning
        
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

### 公共祖先系列

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

[623. Add One Row to Tree](https://leetcode.com/problems/add-one-row-to-tree/)

```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        if depth == 1:
            new_root = TreeNode(val)
            new_root.left = root
            return new_root
        
        def dfs(root, d): # traverse while keep track of cur_depth d
            if not root:
                return
            if d < depth - 1:
                dfs(root.left, d + 1)
                dfs(root.right, d + 1)
            else:
                left_node = TreeNode(val)
                left_node.left = root.left
                right_node = TreeNode(val)
                right_node.right = root.right
                root.left = left_node
                root.right = right_node
        
        dfs(root, 1)
        return root
```

# BST

BST is often used to look up the existence of certain objects. Compared to sorted arrays, the insertion has way lower time complexity, so it's good for dynamic insertion of items. If you don't need to dynamically insert new items, then you can simply sort the collection first and use binary search to look up.

However, most modern languages offers hash tables, which is another way of looking up the existence of an object in a collection. Most implementations are dynamically sized, which can cause the lookup and insertion of items to approach O(1), so usually hash tables are preferred over BST. Nevertheless, there are some advantages to using a BST over a hash table.

Hash tables are unsorted, while BSTs are. If you want to constantly maintain a sorted order while inserting, using a BST is more efficient than a hash table or a sorted list.
It's easy to look up the first element in the BST that is greater/smaller than a lookup value than a hash table.
It's easy to find the k-th largest/smallest element.
Dynamic hash tables usually have a lot of unused memory in order to make the insertion/deletion time approach O(1), whereas BST uses all the memory they requested.

基本性质：对于每个node，左子树节点的值都更小，右子树节点的值都更大；中序遍历结果是有序的

[653. Two Sum IV - Input is a BST](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/)

```py
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        """
        inorder traversal of BST is a sorted array
        """
        arr = []
        
        def dfs(node):
            if not node:
                return
            
            dfs(node.left)
            arr.append(node.val)
            dfs(node.right)
            
        dfs(root)
        
        l, r = 0, len(arr) - 1
        while l < r:
            sum = arr[l] + arr[r]
            if sum == k:
                return True
            elif sum < k:
                l += 1
            else:
                r -= 1
        
        return False
```

[938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/description/)

```py
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root:
            return 0
        res = 0

        def inorder(node):
            nonlocal res
            if not node:
                return 0

            if low <= node.val <= high: # 在范围以内，所以直接加
                res += node.val
            if low < node.val: # 在范围内，但是太大了，所以往左边走 
                inorder(node.left)
            if node.val < high: # 太小了，往右边走
                inorder(node.right)
        inorder(root)

        return res
```

[285. Inorder Successor in BST](https://leetcode.com/problems/inorder-successor-in-bst/)

```py
class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
        """
        从root开始往右一直走，直到走到左子树
        然后往左边走直到遇到这个点
        """
        successor = None
        
        while root:
            if p.val >= root.val:
                root = root.right
            else:
                successor = root
                root = root.left
        
        return successor
```

[510. Inorder Successor in BST II](https://leetcode.com/problems/inorder-successor-in-bst-ii/)

```py

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

[1214. Two Sum BSTs](https://leetcode.com/problems/two-sum-bsts/)

```py
class Solution:
    def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
        arr1 = arr2 = []
        
        def dfs(node, arr):
            if not node:
                return
            
            dfs(node.left, arr)
            arr.append(node.val)
            dfs(node.right, arr)
        
        dfs(root1, arr1)
        dfs(root2, arr2)
        
        l, r = 0, len(arr2) - 1
        
        while l < r:
            sum = arr1[l] + arr2[r]
            
            if sum == target:
                return True
            elif sum < target:
                l += 1
            else:
                r -= 1
        
        return False
```

```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
        """
        更好的方法，直接用predecessor和successor来移动遍历
        """
        
        stack1, stack2 = [], []
        
        while True:
            while root1:
                stack1.append(root1)
                root1 = root1.left
            while root2:
                stack2.append(root2)
                root2 = root2.right
                
            if not stack1 or not stack2:
                break
            
            sum = stack1[-1].val + stack2[-1].val
            
            if sum == target:
                return True
            elif sum < target: # move stack1
                root1 = stack1.pop().right
            else:
                root2 = stack2.pop().left
        
        return False
```

[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

```py
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        """
        二叉搜索树的中序遍历是递增序列
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

[108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

```py
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:    
        """
        always choose the left middle node as root
        
        """
        def helper(left, right):
            if left > right:
                return None

            # always choose left middle node as a root
            p = (left + right) // 2

            # preorder traversal: node -> left -> right
            root = TreeNode(nums[p])
            root.left = helper(left, p - 1)
            root.right = helper(p + 1, right)
            return root
        
        return helper(0, len(nums) - 1)
```

[333. Largest BST Subtree](https://leetcode.com/problems/largest-bst-subtree/) 类似LC98

```py
class SubTree(object):
    def __init__(self, largest, n, min, max):
        self.largest = largest  # largest BST
        self.n = n              # number of nodes in this ST
        self.min = min          # min val in this ST
        self.max = max          # max val in this ST

class Solution(object):
    def largestBSTSubtree(self, root):
        res = self.dfs(root)
        return res.largest
    
    def dfs(self, root):
        if not root:
            return SubTree(0, 0, float('inf'), float('-inf'))
        left = self.dfs(root.left)
        right = self.dfs(root.right)
        
        if root.val > left.max and root.val < right.min:  # valid BST
            n = left.n + right.n + 1
        else:
            n = float('-inf')
            
        largest = max(left.largest, right.largest, n)
        return SubTree(largest, n, min(left.min, root.val), max(right.max, root.val))
```

# 多叉树

## 模板

多叉树遍历模板

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

## 题目

[1490. Clone N-ary Tree](https://leetcode.com/problems/clone-n-ary-tree/)

```py
class Solution:
    def cloneTree(self, root: 'Node') -> 'Node':
        """
        traverse a tree: for each node, make a copy and then link together
        """
        if not root:
            return root
        
        new = Node(root.val)
        
        for child in root.children:
            new.children.append(self.cloneTree(child))
        
        return new
```

[559. Maximum Depth of N-ary Tree](https://leetcode.com/problems/maximum-depth-of-n-ary-tree/)

```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        """
        参考LC104
        分治的思路：从下往上返回高度，每个点返回目前为止的最大深度
        """
        if not root:
            return 0
            
        height = 0
        for child in root.children:
            height = max(height, self.maxDepth(child))
        
        return 1 + height
```

# Trie Tree

[208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

从root开始构建trie tree：每个letter是一个node，叶子节点要另外注明是一个单词的结束。

时间：O(N), N is len(word)
空间：O(N)

```python
class TrieNode:
    def __init__(self): # constructor
        self.children = {} # {letter: TrieNode}
        self.end_of_word = False


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root

        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        
        cur.end_of_word = True

    def search(self, word: str) -> bool:
        cur = self.root

        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        
        return cur.end_of_word

    def startsWith(self, prefix: str) -> bool:
        cur = self.root

        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        
        return True
        
```

[211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

回溯的方式解决.的情况

时间：O(M) M is len(key)
空间：O(M) to keep the recursion stack

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        
        cur.end_of_word = True

    def search(self, word: str) -> bool:
        
        def dfs(j, root):
            cur = root
            
            for i in range(j, (len(word))):
                c = word[i]

                if c == ".":
                    for child in cur.children.values():
                        if dfs(i + 1, child):
                            return True
                    return False

                else:
                    if c not in cur.children:
                        return False
                    cur = cur.children[c]
            
            return cur.end_of_word
        
        return dfs(0, self.root)
```

[212. Word Search II](https://www.youtube.com/watch?v=asbcE9mZz_U)

给目标的words构建一个Trie，从board每一个点走做DFS backtracking

时间：
空间：

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False
    
    def addWord(self, word):
        cur = self
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.end_of_word = True

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()

        for w in words:
            root.addWord(w)
        
        rows, cols = len(board), len(board[0])
        res = set()
        visit = set()

        def dfs(r, c, node, word):
            if r < 0 or r == rows or c < 0 or c == cols or (r, c) in visit or board[r][c] not in node.children:
                return
            
            visit.add((r, c))
            node = node.children[board[r][c]]
            word += board[r][c]
            
            if node.end_of_word:
                res.add(word)

            dfs(r - 1, c, node, word)
            dfs(r + 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)

            visit.remove((r, c))
        
        for r in range(rows):
            for c in range(cols):
                dfs(r, c, root, "")
        
        return list(res)
```

[1268. Search Suggestions System](https://leetcode.com/problems/search-suggestions-system/)

以上是必须掌握题

[820. Short Encoding of Words](https://leetcode.com/problems/short-encoding-of-words/)

```py
class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        """
        倒着构建Trie Tree，构建好之后可以直接返回深度，也可以把叶子放进一个array里，然后就
        """
        root = dict()
        leaves = []
        for word in set(words):
            cur = root
            for i in word[::-1]:
                next_node = cur.get(i, dict()) 
                cur[i] = next_node
                cur = next_node
            leaves.append((cur, len(word) + 1))
        return sum(depth for node, depth in leaves if len(node) == 0)
```

[2135. Count Words Obtained After Adding a Letter](https://leetcode.com/problems/count-words-obtained-after-adding-a-letter/)

```py
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

class Solution:
    
    def __init__(self):
        self.root = TrieNode()
    
    def add(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.end = True
    
    def find(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.end
    
    
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        """
        build trie from the startwords
        sort the word in targetWords, and try to find the potentional word in trie, to reduce the search place
        """
        for word in startWords:
            self.add(sorted(list(word)))
            
        res = 0
        for word in targetWords:
            target = sorted(list(word))
            for i in range(len(target)):
                w = target[:i] + target[i + 1:]
                if self.find(w):
                    res += 1
                    break
        
        return res
```

也可以用bitmask

[f20. Longest Word in Dictionary](https://leetcode.com/problems/longest-word-in-dictionary/)

```py
class TrieNode(object):
    def __init__(self):
        self.children=collections.defaultdict(TrieNode)
        self.isEnd=False
        self.word =''
        
class Trie(object):
    def __init__(self):
        self.root=TrieNode()
        
    def insert(self, word):
        node=self.root
        for c in word:
            node =node.children[c]
        node.isEnd=True
        node.word=word
    
    def bfs(self):
        q=collections.deque([self.root])
        res=''
        while q:
            cur=q.popleft()
            for n in cur.children.values():
                if n.isEnd:
                    q.append(n)
                    if len(n.word)>len(res) or n.word<res:
                        res=n.word
        return res 
    
class Solution(object):
    def longestWord(self, words):
        trie = Trie()
        for w in words: trie.insert(w)
        return trie.bfs()
```

[745. Prefix and Suffix Search](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

# Segment Tree

## what is

Segment Trees allow us to quickly perform range queries as well as range updates。

update: O(logN)
query: O(logN) <-> Comparing with a basic array, this can be O(N)

## how to use

[2, 4, 5, 7]
segment tree:
            18
           /    \
        6          12
      /  \        /   \
    2       4   5       7
leaves are numbers: nums[0] to nums[-1]

each node contains: left, sum, right

## when to use

range query problems like finding minimum, maximum, sum, greatest common divisor, least common denominator in array in logarithmic time.

<https://leetcode.com/articles/a-recursive-approach-to-segment-trees-range-sum-queries-lazy-propagation/>

## 例题

[307. Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)

```py
class Node(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.total = 0
        self.left = None
        self.right = None
        

class NumArray(object):
    def __init__(self, nums):
        """
        initialize your data structure here.
        :type nums: List[int]
        """
        #helper function to create the tree from input array
        def createTree(nums, l, r):
            
            #base case
            if l > r:
                return None
                
            #leaf node
            if l == r:
                n = Node(l, r)
                n.total = nums[l]
                return n
            
            mid = (l + r) // 2
            
            root = Node(l, r)
            
            #recursively build the Segment tree
            root.left = createTree(nums, l, mid)
            root.right = createTree(nums, mid+1, r)
            
            #Total stores the sum of all leaves under root
            #i.e. those elements lying between (start, end)
            root.total = root.left.total + root.right.total
                
            return root
        
        self.root = createTree(nums, 0, len(nums)-1)
            
    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: int
        """
        #Helper function to update a value
        def updateVal(root, i, val):
            
            #Base case. The actual value will be updated in a leaf.
            #The total is then propogated upwards
            if root.start == root.end:
                root.total = val
                return val
        
            mid = (root.start + root.end) // 2
            
            #If the index is less than the mid, that leaf must be in the left subtree
            if i <= mid:
                updateVal(root.left, i, val)
                
            #Otherwise, the right subtree
            else:
                updateVal(root.right, i, val)
            
            #Propogate the changes after recursive call returns
            root.total = root.left.total + root.right.total
            
            return root.total
        
        return updateVal(self.root, i, val)

    def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        #Helper function to calculate range sum
        def rangeSum(root, i, j):
            
            #If the range exactly matches the root, we already have the sum
            if root.start == i and root.end == j:
                return root.total
            
            mid = (root.start + root.end) // 2
            
            #If end of the range is less than the mid, the entire interval lies
            #in the left subtree
            if j <= mid:
                return rangeSum(root.left, i, j)
            
            #If start of the interval is greater than mid, the entire inteval lies
            #in the right subtree
            elif i >= mid + 1:
                return rangeSum(root.right, i, j)
            
            #Otherwise, the interval is split. So we calculate the sum recursively,
            #by splitting the interval
            else:
                return rangeSum(root.left, i, mid) + rangeSum(root.right, mid+1, j)
        
        return rangeSum(self.root, i, j)
```

# MST

It is a graph that connects all the vertices together, withoug cycles and with the minimum total edge weight

## Kruskal's Algo

Kruskal's algorithm generates the Minimum Spanning Tree by always choosing the smallest weigthed edge in the graph and consistently growing the tree by one edge.

1. Sort the edge besed on weights
2. Try every edge, add the edge to res as long as they are not connected -> Disjoint Sets
3. Repeat until have connected n-1 edges

Time: O(ElogE) union find is logE, do it E times. we also sort the graph

```py
class UnionFind:
    def __init__(self):
        self.id = {}

    def find(self, x):
        y = self.id.get(x, x)
        if y != x:
            self.id[x] = y = self.find(y)
        return y

    def union(self, x, y):
        self.id[self.find(x)] = self.find(y)

class Edge:
    def __init__(self, weight, a, b):
        self.weight = weight
        self.a = a
        self.b = b
def cmp():
    def compare(x, y):
        return x.weight < y.weight
    return compare
def minimum_spanning_tree(n : int, edges : List[edge]) -> int:
    # sort list, make sure to define custom comparator class cmp to sort edge based on weight from lowest to highest
    edges.sort(key = cmp)
    dsu = UnionFind()
    ret, cnt = 0, 0
    for edge in edges:
      # check if edges belong to same set before merging and adding edge to mst
      if dsu.find(edge.a) != dsu.find(edge.b):
        dsu.union(edge.a, edge.b)
        ret = ret + edge.weight
        cnt += 1
        if cnt == n - 1:
          break
    return ret
```

## 例题

[1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)

```py
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        """
        step1: create edges
        step2: Prim's algo
        
        Time: O(N^2logN)
        """
        N = len(points)
        adj = {i: [] for i in range(N)} # i : list of [cost, node]
        
        for i in range(N):
            x1, y1 = points[i]
            for j in range(i + 1, N):
                x2, y2 = points[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                adj[i].append([dist, j])
                adj[j].append([dist, i])
        
        # Prim's
        res = 0
        visited = set()
        minH = [[0, 0]] # [cost, point]

        while len(visited) < N:
            cost, i = heapq.heappop(minH)
            if i in visited:
                continue
            res += cost
            visited.add(i)
            for nei_cost, nei in adj[i]:
                if nei not in visited:
                    heapq.heappush(minH, [nei_cost, nei])
        
        return res
```

[1135. Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/)

```py
class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        """
        step1: build adj_list
        """
        adj = {i:[] for i in range(1, n + 1)} # 这些点都从1开始
        for x, y, cost in connections:
            adj[x].append([cost, y]) # i: [cost, node]
            adj[y].append([cost, x])
        
        minH = [(0, 1)] # (cost, node) start with node 1
        visited = set()
        res = 0
        
        while minH:
            cost, node = heapq.heappop(minH) # always deal with the mimimum node first
            if node in visited:
                continue
                
            visited.add(node)
            res += cost
            for nei_cost, nei in adj[node]: # add all unvisited nodes
                if nei not in visited:
                    heapq.heappush(minH, (nei_cost, nei))
        
        return res if len(visited) == n else -1
```
