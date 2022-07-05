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