
[973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

方法一：minHeap
先形成一个minHeap, key是dist，另外还要保存x,y用来之后导出，然后根据要求取数字
```py
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        minHeap = []
        
        for x, y in points:
            dist = x * x + y * y
            minHeap.append([dist, x, y])
        
        heapq.heapify(minHeap)
        
        res = []
        while k > 0:
            dist, x, y = heapq.heappop(minHeap)
            res.append([x, y])
            k -= 1
        
        return res
```

方法二：quick select

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        
        def get_dist(point):
            return point[0] ** 2 + point[1] ** 2
        
        # return p, p is pth smallest in points
        def partition(l, r):
            pivot = points[r]
            p = l
            pivot_dist = get_dist(pivot)
            
            for i in range(l, r):
                if get_dist(points[i]) <= pivot_dist:
                    points[i], points[p] = points[p], points[i]
                    p += 1
            
            # before < get_dist(points[r]) < after
            points[r], points[p] = points[p], points[r]
            return p
        
        def select(l, r):
            if l > r:
                return points
            p = partition(l, r)

            if k < p:
                return select(l, p - 1)
            elif k > p:
                return select(p + 1, r)
            else:
                return points[:k]
        
        return select(0, len(points) - 1)
```


[347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        bucket sort
        i(count)  0 |  1  | 2  | 3 | 4 | 5 | ... | len(input) 
        values       [100]     [1,2]

        用map来记录num: count的数量，之后构建一个count:values的array，；最后array从后往前往res里加，直到len(res) == k；构建array: freq = [[] for i in range(len(nums) + 1)]; 从后往前遍历: for i in range(len(freq) -1, 0, -1)
        时间：O(N)
        空间：O(N)
        """
        # count: {each num:freq}
        count = collections.Counter(nums)
        # freq: [freq:[nums have same freq]]
        freq = [[] for i in range(len(nums) + 1)] # 大小是len(nums) + 1，注意如何构建values是list的list

        for n, count in count.items():
            freq[count].append(n)

        res = []
        for i in range(len(freq) -1, 0, -1): # 注意如何从后往前遍历
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res 
```


```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        maxHeap: {count: key}
        O(KlogN)
        """
        # count: {each num:freq}
        count = collections.Counter(nums)
        
        maxHeap = []
        
        for key in count:
            heapq.heappush(maxHeap, (-count[key], key))
        
        res = []
        while k > 0:
            value, key = heapq.heappop(maxHeap)
            res.append(key)
            k -= 1
        return res
```

[23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

```py
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Heap
        """
        dummy = ListNode(-1)
        cur = dummy
        minHeap = []
        
        # add the first node of each list into the minHeap
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(minHeap, (lists[i].val, i))
                lists[i] = lists[i].next # 每个list指向第二个节点
        
        while minHeap: 
            val, i = heapq.heappop(minHeap) # 取出来最小值
            cur.next = ListNode(val) # 新建node并连接到当前的node
            cur = cur.next # 移动ptr
            if lists[i]: # 把更新后的点加进minHeap
                heapq.heappush(minHeap, (lists[i].val, i))
                lists[i] = lists[i].next
        
        return dummy.next
```

```py
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Divide and conquer

        Time: O(NlogK) N is len(one_list), K is num of lists
        Space: O(N) for one_merge, O(1) for merge_two()
        """
        # edge case
        if not lists or len(lists) == 0:
            return None
        
        while len(lists) > 1:
            one_merge = []
            
            for i in range(0, len(lists), 2): # each time the step is 2
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None # check for the odd condition
                one_merge.append(self.merge_two(l1, l2))
                
            lists = one_merge
        
        return lists[0]
    
    def merge_two(self, l1, l2):
        """
        Same as Leetcode Q21
        """
        dummy = ListNode(-1) # dummy node to avoid empty ptr
        pre = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                pre.next = l1
                l1 = l1.next
            else:
                pre.next = l2
                l2 = l2.next
            pre = pre.next # don't forget to move ptr
        
        # append the remaining nodes in l1 or l2
        if l1:
            pre.next = l1
        elif l2:
            pre.next = l2
        
        return dummy.next
```

[264. Ugly Number II](https://leetcode.com/problems/ugly-number-ii/)



[1086. High Five](https://leetcode.com/problems/high-five/)






















# Others

[703. Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)
minHeap of size K
add/pop: O(logN)
get min: O(1)


```py
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        """
        先形成一个minHeap(nums) -> O(n)
        然后一直pop直到len(minHeap)为k -> O((n-k)logN)
        最后Kth largest就是minHeap里最小的值 -> O(1)
        """
        self.minHeap, self.k  = nums, k
        heapq.heapify(self.minHeap)
        while len(self.minHeap) > k:
            heapq.heappop(self.minHeap)

    def add(self, val: int) -> int:
        """
        加进来，然后一直pop直到len(minHeap)为k -> O(logN)
        最后返回minHeap[0]
        """
        heapq.heappush(self.minHeap, val)
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)
        return self.minHeap[0]

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```

[1046. Last Stone Weight](https://leetcode.com/problems/last-stone-weight/)
用一个maxHeap。每次先取出来2个数，然后如果不同就加进来他们的差，最后要注意为空的情况

```py
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        maxHeap = [-x for x in stones]
        heapq.heapify(maxHeap)
        
        while len(maxHeap) > 1:
            largest = heapq.heappop(maxHeap)
            second = heapq.heappop(maxHeap)
            if largest < second:
                diff = largest - second
                heapq.heappush(maxHeap, diff)
                
        maxHeap.append(0) # 处理input=[2,2]的情况
        return abs(maxHeap[0])  
```


[215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

用maxHeap
时间 O(N) + O(kLogN)
空间 O(N)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        maxHeap = [-x for x in nums]
        heapq.heapify(maxHeap)
        
        while k > 0:
            res = heapq.heappop(maxHeap)
            k -= 1
        
        return -res
```

Quickselect
partition: cut to two halves，左边的数都比右边的小，pivot就选最右的数，这个数字就是左右两边数的分界: p从最左index开始一直往右走，如果这个数比pivot小，那就放进来，然后p+=1，最后把p和pivot互换，效果就是pivot左边的数都比pivot小

时间 O(N)；如果每次的pivot都刚好是最大值，那每次都需要走一遍，所以那就是O(N^2)
空间 O(1)
```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        k = len(nums) - k # 把k变成sorted情况下的第k位

        # return p, p is pth smallest in nums     
        def partition(l, r):
            pivot, p = nums[r], l

            # nums before < nums[p] < nums after, based on pivot
            for i in range(l, r):
                if nums[i] <= pivot: # 如果当前这个数<=pivot，就放到左边
                    nums[p], nums[i] = nums[i], nums[p] # python不用一个swap()
                    p += 1
                    
            # nums before < nums[r] < nums after
            nums[p], nums[r] = nums[r], nums[p]
            return p

        def select(l, r): # l, r告诉跑quickSelect的范围
            p = partition(l, r)
            
            if k < p: 
                return select(l, p - 1)
            elif k > p:
                return select(p + 1, r)
            else:
                return nums[p]

        return select(0, len(nums) - 1)
```

[621. Task Scheduler](https://leetcode.com/problems/task-scheduler/)
每次都先处理max_freq的task：每次都要最大值->maxHeap；另外用一个queue存

时间：O(N*M) N is len(tasks), M is idleTime
空间：
```py
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        count = collections.Counter(tasks)
        maxHeap = [-cnt for cnt in count.values()]
        heapq.heapify(maxHeap)

        time = 0
        queue = deque() # pairs of [-cnt, idleTime]

        while maxHeap or queue:
            time += 1

            if maxHeap:
                cnt = heapq.heappop(maxHeap) + 1
                if cnt:
                    queue.append([cnt, time + n])
            
            if queue and queue[0][1] == time:
                heapq.heappush(maxHeap, queue.popleft()[0])

        return time

```

[295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
用两个size最多差1的Heap，每次新数字先加到minHeap，然后pop minHeap到maxHeap，最后永远保证minHeap的大小比maxHeap大于等于1；找数的时候，要么直接看minHeap，要么就是看二者的平均数

时间：addNum: O(logN)，找数O(1)
空间：O(N)
```py
class MedianFinder:

    def __init__(self):
        # all nums in small < all nums in large
        # small is a maxHeap, large is a minHeap
        self.small, self.large = [], []

    def addNum(self, num: int) -> None:
        # add each element to minHeap first
        # pop minHeap and add it to maxHeap 
        # balance the size 
        # In this case, all nums in small < large
        heapq.heappush(self.small, -heapq.heappushpop(self.large, num))
        
        if len(self.small) > len(self.large):
            heapq.heappush(self.large, -heapq.heappop(self.small))
        
    def findMedian(self) -> float:
        if len(self.small) < len(self.large):
            return self.large[0]
        else:
            return (self.large[0] - self.small[0]) / 2


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```


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