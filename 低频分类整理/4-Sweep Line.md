Line-sweep algorithms give one a good way to geometirc problems when it comes to update and query efficiently. 

伪代码
Variables: events -> list, active -> data structure to store active points
for every rectangle {
    add left endpoint to events
    add right endpoint to events
}
sort(events)
for every event in events {
  compute total area currently covered
  update events based on the points
}

[253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
```py
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        """
        想象有根线，沿着时间轴扫描，count遇到start就+1，遇到end就-1，count的最大值就是需要的最大值

        Time: O(NlogN)
        Space: O(N)
        """
        start = sorted([i[0] for i in intervals])
        end = sorted([i[1] for i in intervals])        
        
        res, count = 0, 0
        s, e = 0, 0
        
        while s < len(start):
            if start[s] < end[e]:
                count += 1
                s += 1
                res = max(res, count)
            else:
                count -= 1
                e += 1
                
        return res
```

Heap解法待续

[218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)
```py
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        """
        a vertical line to scan from left to right. If max height changes, add it to res

        Time: O(NlogN)
        Space: O(N)
        """
        events = []
        for L, R, H in buildings:
            events.append((L, -H, R)) # 轮廓升高, -H是为了maxHeap
            events.append((R, 0, 0)) # 轮廓降低
        
        events.sort()
        
        res = [[0, 0]]
        # maxHeap记录高为h的skyline结束的点pos: [(h, pos)]
        maxHeap = [(0, float("inf"))] # 返回最高的skyline
        
        for L, H, R in events:
            while L >= maxHeap[0][1]: # 找到升高的终点
                heapq.heappop(maxHeap)
            
            # 轮廓升高
            if H:
                heapq.heappush(maxHeap, (H, R))
            
            # 如果轮廓变化了，就记录
            if res[-1][1] != -maxHeap[0][0]:
                res += [[L, -maxHeap[0][0]]]
        
        return res[1:]
```

分治解法待续


[759. Employee Free Time](https://leetcode.com/problems/employee-free-time/)
sweep line
```py
class Solution(object):
    def employeeFreeTime(self, schedule):
        """
        :type schedule: [[Interval]]
        :rtype: [Interval]
        """
        if not schedule or not schedule[0]:
            return []
        List = [i for l in schedule for i in l]
        List.sort(key = lambda a:a.start)
        res, prev_end = [], List[0].end
        for interval in List:
            if interval.start>prev_end:
                new_i = Interval(prev_end, interval.start)
                res.append(new_i)
            prev_end= max(prev_end, interval.end)
        return res
```

minHeap
```py
class Solution:
    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        """
        sort all the intervals based on the starting time. this gives a set of busy intervals
        """
        heap = []
        for emp in schedule:
            for iv in emp:
                heap.append((iv.start, iv.end))
        heapify(heap)
        
        s, e = heappop(heap)
        free = e
        res = []
        while heap:
            s, e = heappop(heap)
            if s > free:
                res.append(Interval(free, s))
                free = e
            else:
                free = max(free, e)
        return res
```