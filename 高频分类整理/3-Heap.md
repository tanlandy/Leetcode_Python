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
```$$