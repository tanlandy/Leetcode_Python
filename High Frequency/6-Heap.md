# Recap

Priority Queue is an Abstract Data Type, and Heap is the concrete data structure we use to implement a priority queue.

## Operation

Criteria: Complete Binary Tree + parent nodes are smaller or larger than children
Insertion of an element into the Heap has a time complexity of O(logN);
Deletion of an element from the Heap has a time complexity of O(logN);
The maximum/minimum value in the Heap can be obtained with O(1) time complexity.

The number in each node is *key*, not value(similar to tree node). We use the *key* to sort the nodes, and values are the data we want the heap to store
There is no comparable relationsihp across a level of a heap

because only nodes in a root-to-leaf path are sorted (nodes in the same level are not sorted), when we add/remove a node, we only have to fix the order in the vertical path the node is in. This makes inserting and deleting O(log(N)) too.

```py
import heapq

# Construct a Heap -> O(N)
minHeap = [12,3,132]
heapq.heapify(minHeap)
maxHeap = [-x for x in minHeap]
heapq.heapify(maxHeap)

# Insert an element: 5 -> O(logN)
heapq.heappush(minHeap, 5)
heapq.heappush(maxHeap, -1 * 5) 

# Get the top element -> O(1)
top = minHeap[0]
top = -1 * maxHeap[0]

# Delete the top element -> O(logN)
heapq.heappop(minHeap)
heapq.heappop(maxHeap)

# Get the size -> O(1)
len(minHeap)
```

# Categorized

[973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

```py
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        max_heap with a size of k. key是dist，另外还要保存x,y用来之后导出
        
        Time: O(NlogK)
        Space: O(N)
        """
        # k smallest points
        max_heap = []
        for x, y in points:
            dist = x * x + y * y
            heapq.heappush(max_heap, (-dist, x, y))
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        
        res = []
        while max_heap:
            dist, x, y = heapq.heappop(max_heap)
            res.append([x, y])
        
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
        [1,1,1,2,2,2,100]
        bucket sort
        i(freq)  0 |  1  | 2  | 3 | 4 | 5 | ... | len(input) 
        values       [100]     [1,2]

        用map来记录num: freq的数量，之后构建freq :values的array.最后array从后往前往res里加，直到len(res) == k；
        构建array: freq = [[] for i in range(len(nums) + 1)]
        从后往前遍历: for i in range(len(freq) -1, 0, -1)
        时间：O(N)
        空间：O(N)
        """
        # counter: {num: freq}
        counter = collections.Counter(nums)
        # freqs: [freq:[nums have same freq]]
        freqs = [[] for i in range(len(nums) + 1)] # 大小是len(nums) + 1，因为可能[1,1,1]的时候，freq=3

        for num, count in counter.items():
            freqs[count].append(num)

        res = []
        for i in range(len(freqs) -1, 0, -1): # 注意如何从后往前遍历
            for num in freqs[i]:
                res.append(num)
                if len(res) == k:
                    return res 
```

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        use counter:{num:freq}
        then push them to a maxHeap, based on freq
        finally pop heap according to k

        Time: O(NlogN)
        Space: O(N)
        """
        # counter: {each num:freq}
        counter = collections.Counter(nums)

        maxHeap = []

        for num, freq in counter.items():
            heapq.heappush(maxHeap, (-freq, num))

        res = []
        while k > 0:
            freq, num = heapq.heappop(maxHeap)
            res.append(num)
            k -= 1
        return res
```

```py
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        as the res can be in any order: maintain a min_heap with a size of k, that always contains the top k largest elem.

        Time: O(NlogK)
        Space: O(N)
        """
        counter = collections.Counter(nums)
        
        min_heap = []
        
        for num, freq in counter.items():
            heapq.heappush(min_heap, (freq, num))
            if len(min_heap) > k: # always pop the smallest element
                heapq.heappop(min_heap)
        res = []
        
        while k > 0:
            freq, num = heapq.heappop(min_heap)
            res.append(num)
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

```py
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        """
        starting from 1, in each cycle, pop the top of the heap and insert back that number multiplied by 2, 3, 5 into the heap (if that number wasn't in the heap already)
        """
        primes = (2, 3, 5)
        minHeap = [1]
        used = set([1])

        for i in range(n - 1):
            val = heapq.heappop(minHeap)
            for prime in primes:
                new_ugly = prime * val
                if new_ugly not in used:
                    heapq.heappush(minHeap, new_ugly)
                    used.add(new_ugly)

        return minHeap[0]
```

```py
class Solution:

    def nthUglyNumber(self, n):
        """
        use three pointers, each time only moves the smallest one, generate all ugly nunmbers

        Time: O(N)
        Space: O(N)
        """
        ugly = [1]
        i2, i3, i5 = 0, 0, 0
        while n > 1:
            u2, u3, u5 = 2 * ugly[i2], 3 * ugly[i3], 5 * ugly[i5]
            umin = min((u2, u3, u5))
            if umin == u2:
                i2 += 1
            if umin == u3:
                i3 += 1
            if umin == u5:
                i5 += 1
            ugly.append(umin)
            n -= 1
        return ugly[-1]
```

[1086. High Five](https://leetcode.com/problems/high-five)

```py
class Solution:
    def highFive(self, items: List[List[int]]) -> List[List[int]]:
        """
        for each student, create a maxHeap of their score
        then, calculate their average score and add it to result

        Time: O(NlogN) sort at the end
        Space: O(N)
        """
        students = collections.defaultdict(list)

        for idx, val in items:
            heapq.heappush(students[idx], val)

            # keep the size of heap to 5: pop the smallest when size is larger than 5
            if len(students[idx]) > 5:
                heapq.heappop(students[idx])

        res = [[i, sum(students[i]) // len(students[i])] for i in sorted(students)]

        return res
```

[88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

```py
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        edit nums1 in-place using two pointers
        at the end, what left in nums2 should be added to nums1

        Time: O(M+N)
        Space: O(1)
        """
        while m > 0 and n > 0:
            if nums1[m - 1] > nums2[n - 1]:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1

        nums1[:n] = nums2[:n]
```

[692. Top K Frequent Words]([Loading...](https://leetcode.com/problems/top-k-frequent-words/))

```py
class Solution:
    def topKFrequent(self, words: 'List[str]', k: 'int') -> 'List[str]':
        class Solution:
    def topKFrequent(self, words: 'List[str]', k: 'int') -> 'List[str]':
        """
        heapq.nsmallest(n, iterable, key=None): Return a list with the n smallest elements from the dataset defined by iterable. key, if provided 

        Time: O(NlogK) -> Heapq will build the heap for the first t elements, then later on it will iterate over the remaining elements by pushing and popping the elements from the heap (maintaining the t elements in the heap).
        Space: O(N)
        """
        counts = collections.Counter(words)
        return heapq.nsmallest(k, counts,
            key=lambda word:(-counts[word], word)
        )
# -Freqs[words] means 'rank the frequencies of words in descending order' (the first sorting criteria in lambda function);
# word means 'rank the words with the highest frequencies in their alphabetical order' (the second sorting criteria in lambda function).
# Finally, nsmallest() returns the [:k] of the result.
```

```py
class Element:
    def __init__(self, count, word):
        self.count = count
        self.word = word

    def __lt__(self, other): # less than
        if self.count == other.count:
            return self.word > other.word
        return self.count < other.count

    def __eq__(self, other): # equal
        return self.count == other.count and self.word == other.word

class Solution(object):
    def topKFrequent(self, words, k):
        counts = collections.Counter(words)   

        freqs = []
        heapq.heapify(freqs)
        for word, count in counts.items():
            # presonal comparator
            heapq.heappush(freqs, (Element(count, word), word))
            if len(freqs) > k:
                heapq.heappop(freqs)

        res = []
        for _ in range(k):
            res.append(heapq.heappop(freqs)[1])
        return res[::-1]
```

```py
class Solution:
    def topKFrequent(self, words, k):
        """
        Time: O(NlogN)
        Space: O(N)
        """
        counts = collections.Counter(words)
        items = list(counts.items())
        # sort first based on pref, then on words
        items.sort(key=lambda item:(-item[1],item[0]))
        return [item[0] for item in items[0:k]]
```

[378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/) 

```py
class Solution:  # 204 ms, faster than 54.32%
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        """
        add the first col into the heap, then process one by one. once pop a new item out of heap, push the neighbor into it.

        Time: Klog(min(K, N))
        Space: O(min(K, N))
        """
        rows, cols = len(matrix), len(matrix[0])  # For general, the matrix need not be a square

        minHeap = []  # val, r, c
        # add first col
        for r in range(min(k, rows)):
            heappush(minHeap, (matrix[r][0], r, 0))

        res = 0  # any dummy value
        for i in range(k):
            res, r, c = heappop(minHeap)
            # once get the number, add the next one in the row
            if c + 1 < cols: 
                heappush(minHeap, (matrix[r][c + 1], r, c + 1))
        return res
```

[295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) 

```py
class MedianFinder:
"""
用两个size最多差1的Heap，每次新数字先加到minHeap，然后pop minHeap到maxHeap，最后永远保证minHeap的大小比maxHeap大于等于1；找数的时候，要么直接看minHeap，要么就是看二者的平均数

always maintain a min_heap and a max_heap, where elems in min_heap > max_heap: to make this come true, each new num should be put into min_heap first, then pop the smallest in min_heap to max_heap.
in order to find the median, each time, push the largest in max_heap back to min_heap

时间：addNum: O(logN)，找数O(1)
空间：O(N)
"""

    def __init__(self):
        # all nums in max_heap < all nums in min_heap
        self.max_heap, self.min_heap = [], []

    def addNum(self, num: int) -> None:
        # add each element to minHeap first
        # pop minHeap and add it to maxHeap 
        # balance the size 
        # In this case, all nums in max_heap < min_heap
        heapq.heappush(self.min_heap, num)
        elem = heapq.heappop(self.min_heap)
        heapq.heappush(self.max_heap, -elem)
        
        if len(self.max_heap) > len(self.min_heap):
            elem = heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, -elem)

    def findMedian(self) -> float:
        if len(self.maxHeap) < len(self.minHeap):
            return self.min_heap[0]
        else:
            return (self.min_heap[0] - self.max_heap[0]) / 2


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

[767. Reorganize String]([Loading...](https://leetcode.com/problems/reorganize-string/))

```py
class Solution:
    def reorganizeString(self, s: str) -> str:
        """
        repeatedly select the most frequent character that isn't the one previously placed -> max_heap based on the count

        Time: O(Nlogk), N is characters in the string, k is the total [unique] characters in the string
        Space: O(k)
        """
        res = []
        heap = [(-count, ch) for ch, count in Counter(s).items()]
        heapify(heap)

        while heap:
            count_first, ch_first = heappop(heap)
            if not res or ch_first != res[-1]:  # add ch_first to the result
                res.append(ch_first)
                if count_first + 1 != 0:  # as used, push back if it's not 0
                    heappush(heap, (count_first + 1, ch_first))
            else:  # add ch_second to the result
                if not heap:  # there's no second can be used
                    return ''
                count_second, ch_second = heappop(heap)
                res.append(ch_second)
                if count_second + 1 != 0:  # as used, push back if it's not 0
                    heappush(heap, (count_second + 1, ch_second))
                heappush(heap, (count_first, ch_first))  # push back ch_first always

        return ''.join(res)
```

[1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit]([Loading...](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/))

```py
class Solution:
    def longestSubarray(self, A, limit):
        maxq, minq = [], []
        res = i = 0
        for j, a in enumerate(A):
            heapq.heappush(maxq, [-a, j])
            heapq.heappush(minq, [a, j])
            while -maxq[0][0] - minq[0][0] > limit:
                i = min(maxq[0][1], minq[0][1]) + 1
                while maxq[0][1] < i: heapq.heappop(maxq)
                while minq[0][1] < i: heapq.heappop(minq)
            res = max(res, j - i + 1)
        return res
```

[895. Maximum Frequency Stack]([Loading...](https://leetcode.com/problems/maximum-frequency-stack/))

```py
class FreqStack:
    """
    use two hashmap: 
    1. counter: {one num: its freq}
    2. group: {1 to maxCount: a list of all nums with that freq}
    """

    def __init__(self):
        self.count = collections.defaultdict(int)
        self.maxCount = 0
        self.group = collections.defaultdict(list)

    def push(self, val: int) -> None:
        valCount = self.count[val] + 1 # update one nums's count
        self.count[val] = valCount
        if valCount > self.maxCount: # update the maxCount
            self.maxCount = valCount
        self.group[valCount].append(val) # update the group

    def pop(self) -> int:
        num = self.group[self.maxCount].pop()
        self.count[num] -= 1
        if not self.group[self.maxCount]:
            self.maxCount -= 1

        return num
```

# Others

[703. Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)

```py
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        """
        先形成一个minHeap(nums) -> O(n)
        然后一直pop直到len(minHeap)为k -> O((n-k)logN)
        最后Kth largest就是minHeap里最小的值 -> O(1)

        minHeap of size K
        add/pop: O(logN)
        get min: O(1)
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

[1337. The K Weakest Rows in a Matrix]([Loading...](https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/))

```py

```

[1642. Furthest Building You Can Reach](https://leetcode.com/problems/furthest-building-you-can-reach/)

```py
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        尽可能让gap大的时候用ladder，其他时候用bricks
        思路：用minHeap，每次有gap的时候就放进来，当len(minHeap) > gap的时候就pop出来最小的消耗bricks，直到bricks < 0

        Time: O(NlogN)
        Space: O(N)
        """
        minHeap = []
        
        for i in range(len(heights) - 1):
            gap = heights[i + 1] - heights[i]
            
            if gap <= 0:
                continue
            
            heapq.heappush(minHeap, gap)
            
            if len(minHeap) <= ladders:
                continue
            
            bricks -= heapq.heappop(minHeap)
            
            if bricks < 0:
                return i
        
        return len(heights) - 1
```

```py
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        类似第一个方法，只是先尽可能用bricks，然后当有梯子并且bricks<0的时候，把用过的最大的gap加到bricks

        """
        maxHeap = []
        
        for i in range(len(heights) - 1):
            gap = heights[i + 1] - heights[i]
            
            if gap < 0:
                continue
            
            heapq.heappush(maxHeap, -gap)
            
            bricks -= gap

            if bricks < 0 and ladders == 0:
                return i
            
            if bricks < 0:
                for_ladder = -heapq.heappop(maxHeap)
                bricks += for_ladder
                ladders -= 1
        
        return len(heights) - 1
```

```py
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        二分查找：找最右满足的边界
        每次判断是否满足：穷举出来所有gaps，然后排序，把小的给bricks用，大的给ladders

        """
        def is_reachable(mid):
            gaps = []
            for h1, h2 in zip(heights[:mid], heights[1:mid + 1]):
                if h2 - h1 > 0:
                    gaps.append(h2 - h1)
            gaps.sort()
            b_remain = bricks
            l_remain = ladders
            for gap in gaps:
                if gap <= b_remain:
                    b_remain -= gap
                elif l_remain > 0:
                    l_remain -= 1
                else:
                    return False
            return True
        
        l, r = 0, len(heights) - 1
        while l <= r:
            mid = (l + r) // 2
            if is_reachable(mid):
                l = mid + 1
            else:
                r = mid - 1
        
        return r
```

[1229. Meeting Scheduler](https://leetcode.com/problems/meeting-scheduler/)

```py
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        """
        sort input arrays and apply two pointers
        always move the pointer that ends earlier

        Time: O(MlogM + NlogN)
        Space: O(M + N) as sort in python takes O(N) for the worst case
        """
        
        slots1.sort()
        slots2.sort()
        
        ptr1 = ptr2 = 0
        
        while ptr1 < len(slots1) and ptr2 < len(slots2):
            # find the common slot
            slot_end = min(slots1[ptr1][1], slots2[ptr2][1])
            slot_begin = max(slots1[ptr1][0], slots2[ptr2][0])
            if slot_end - slot_begin >= duration:
                return [slot_begin, slot_begin + duration]
            
            # mvoe the one ends earlier
            if slots1[ptr1][1] < slots2[ptr2][1]:
                ptr1 += 1
            else:
                ptr2 += 1
        return []
```

```py
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        # build up a heap containing time slots last longer than duration
        timeslots = list(filter(lambda x: x[1] - x[0] >= duration, slots1 + slots2))
        heapq.heapify(timeslots)

        while len(timeslots) > 1:
            start1, end1 = heapq.heappop(timeslots)
            start2, end2 = timeslots[0]
            if end1 >= start2 + duration:
                return [start2, start2 + duration]
        return []
```

# 总结

经典pq题目：767
