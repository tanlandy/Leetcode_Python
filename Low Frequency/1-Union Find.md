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

[721. Accounts Merge](https://leetcode.com/problems/accounts-merge/)

[547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/)



[305. Number of Islands II](https://leetcode.com/problems/number-of-islands-ii/)

```py
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        def find(x):
            while x in pa:
                if pa[x] in pa:#path compress
                    pa[x]=pa[pa[x]]
                x=pa[x]
            return x    
        def union(x,y):
            pax,pay=find(x),find(y)
            if pax==pay:#union fail,has been unioned.
                return False
            pa[pax]=pay
            return True
        seen,pa,res,count=set(),{},[],0
        for x,y in positions:#connect with neighbor val==1,if union success,means one island disappear.
            if (x,y) not in seen:
                seen.add((x,y))
                count+=1
                for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                    if (i,j) in seen and union((i,j),(x,y)):
                        count-=1
            res.append(count)
        return res
```