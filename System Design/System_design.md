# System design

[355. Design Twitter](https://leetcode.com/problems/design-twitter/)

```py
class Twitter:

    def __init__(self):
        self.count = 0
        self.tweetMap = defaultdict(list)  # userId -> list of [count, tweetIds]
        self.followMap = defaultdict(set)  # userId -> set of followeeId
         
    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweetMap[userId].append([self.count, tweetId])
        self.count -= 1

    def getNewsFeed(self, userId: int) -> List[int]:
        res = []
        minHeap = [] 
        
        self.followMap[userId].add(userId)
        for followeeId in self.followMap[userId]:
            if followeeId in self.tweetMap:
                index = len(self.tweetMap[followeeId]) - 1
                count, tweetId = self.tweetMap[followeeId][index]
                heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])

        while minHeap and len(res) < 10:
            count, tweetId, followeeId, index = heapq.heappop(minHeap)
            res.append(tweetId)
            if index >= 0:
                count, tweetId = self.tweetMap[followeeId][index]
                heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        self.followMap[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.followMap[followerId]:
            self.followMap[followerId].remove(followeeId)
```

[1472. Design Browser History](https://leetcode.com/problems/design-browser-history/)

```py
class BrowserHistory:

    def __init__(self, homepage: str):
        """
        用2个stack，分别存history和future
        """
        self.his = []
        self.fut = []
        self.his.append(homepage)      

    def visit(self, url: str) -> None:
        self.his.append(url)
        self.fut = []

    def back(self, steps: int) -> str:
        """
        把对应步数的网站从his存到fut
        """
        while steps > 0 and len(self.his) > 1: # always let it bigger than 1, so that return homepage whenever steps < 0
            self.fut.append(self.his.pop())
            steps -= 1
        return self.his[-1]

    def forward(self, steps: int) -> str:
        while steps > 0 and self.fut:
            self.his.append(self.fut.pop())
            steps -= 1
        return self.his[-1]
```

```py
class BrowserHistory:

    def __init__(self, homepage: str):
        """
        更好的办法，只用一个array，同时用一个指针来指向现在正在访问的网站，每次visit的时候要保证cur处在最右边
        """
        self.history = [homepage]
        self.curr = 0
        
    def visit(self, url: str) -> None:
        self.curr += 1
        while self.curr < len(self.history):  # 往前面走过，所以要把后面的删掉
            self.history.pop()

        self.history.append(url)
        
    def back(self, steps: int) -> str:
        self.curr = max(0, self.curr-steps)
        return self.history[self.curr]
      
    def forward(self, steps: int) -> str:
        self.curr = min(len(self.history)-1, self.curr+steps)
        return self.history[self.curr]

```
