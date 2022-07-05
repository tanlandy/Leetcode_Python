[323. Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/discuss/319459/Python3-UnionFindDFSBFS-solution)

```py
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        """
        并查集的模板写法，记下来find()和union()模板
        """
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        
        for x, y in edges: # 将每个节点连通
            union(x, y)
        
        return len({find(i) for i in range(n)}) # 返回连通个数
```


```py
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        """
        DFS：建图，然后对每个节点遍历，同时用visit来记录已经走过的
        """
        graph = collections.defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)
        
        def dfs(node):
            visit.add(node)
            for nei in graph[node]:
                if nei not in visit:
                    dfs(nei)
        
        count = 0
        visit = set()
        for node in range(n):
            if node not in visit:
                dfs(node)
                count += 1
        
        return count

```