1. 写一个可以insert, get, addToKey(all keys + certain number), addToValue(all values + certain number)的dict
-> 不用每次都加到key或者value上，只在insert和get的之后加上offset


https://leetcode.com/discuss/interview-question/933426/oa-uber 

```py
def solution(queryType, query):
    hm = {}
    res = 0
    ck, cv = 0, 0
    for i in range(len(queryType)):
        if queryType[i] == "insert":
            hm[query[i][0] - ck] = query[i][1] - cv
        elif queryType[i] == "addToKey":
            ck += query[i][0]
        elif queryType[i] == "addToValue":
            cv += query[i][0]
        elif queryType[i] == "get":
            res += hm[query[i][0] - ck] + cv
    return res
```



2. valid sodoku LC36

```py
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        N = 9

        # Use hash set to record the status
        rows = [set() for _ in range(N)]
        cols = [set() for _ in range(N)]
        boxes = [set() for _ in range(N)]

        for r in range(N):
            for c in range(N):

                val = board[r][c]
                # Check the row
                if val in rows[r]:
                    return False
                rows[r].add(val)

                # Check the column
                if val in cols[c]:
                    return False
                cols[c].add(val)

                # Check the box
                idx = (r // 3) * 3 + c // 3
                if val in boxes[idx]:
                    return False
                boxes[idx].add(val)

        return True
```

