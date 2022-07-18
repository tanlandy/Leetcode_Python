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



[1130. Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)

# BST
BST is often used to look up the existence of certain objects. Compared to sorted arrays, the insertion has way lower time complexity, so it's good for dynamic insertion of items. If you don't need to dynamically insert new items, then you can simply sort the collection first and use binary search to look up.

However, most modern languages offers hash tables, which is another way of looking up the existence of an object in a collection. Most implementations are dynamically sized, which can cause the lookup and insertion of items to approach O(1), so usually hash tables are preferred over BST. Nevertheless, there are some advantages to using a BST over a hash table.

Hash tables are unsorted, while BSTs are. If you want to constantly maintain a sorted order while inserting, using a BST is more efficient than a hash table or a sorted list.
It's easy to look up the first element in the BST that is greater/smaller than a lookup value than a hash table.
It's easy to find the k-th largest/smallest element.
Dynamic hash tables usually have a lot of unused memory in order to make the insertion/deletion time approach O(1), whereas BST uses all the memory they requested.



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




# 多叉树



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
            
            for i in range(j, (len(word)):
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

# Segment Tree
Segment Trees allow us to quickly perform range queries as well as range updates。

update: O(logN)
query: O(logN) <-> Comparing with a basic array, this can be O(N)

## when to use
range query problems like finding minimum, maximum, sum, greatest common divisor, least common denominator in array in logarithmic time.

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