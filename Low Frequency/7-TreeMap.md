# 基础知识
<https://grantjenks.com/docs/sortedcontainers/>

# 例题

[981. Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)

```py
class TimeMap:

    def __init__(self):
        """
        dic: {key: list of [timestamp, value]}
        """
        self.dic = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.dic[key].append([timestamp, value])

    def get(self, key: str, timestamp: int) -> str:
        """
        as return the pre_time <= timestamp: has to use binary search to find the pre_time to return

        Time: O(logN)
        Space: O(logN)
        """
        arr = self.dic[key]
        l, r = 0, len(arr) - 1
        while l <= r:
            mid = (l + r) // 2
            if arr[mid][0] <= timestamp:
                l = mid + 1
            else:
                r = mid - 1
        
        return "" if r == -1 else arr[r][1]


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)
```

follow-up

```py
"""
Implement a Map interface, allowing us to put a key-value pair, and retieve it with get(Key) to get lastest, and get(key, time) to get the value at the exact time it was inserted.

example:
put("fruit", "banana") // at time 1
put("fruit", "apple") // at time 2
put("fruit", "orange") // at time 3

get("fruit") => "orange"
get("fruit", "2") => apple
"""

import collections
class TimeMap:
    def __init__(self):
        self.last_dic = {}
        self.timed_dic = {}
        self.time = 0
    
    def put(self, key, value):
        """
        use a map of map, map the time to the value, map the key to the dict that stores the time:value pair
        """
        self.last_dic[key] = value
        self.time += 1
        if key not in self.timed_dic:
            self.timed_dic[key] = {}
        self.timed_dic[key][self.time] = value
    
    def get(self, key, timestamp=None):
        """
        Time: O(1)
        """
        if timestamp is None:
            print(self.last_dic[key])
            return
        if timestamp < 1:
            print("")
            return
        print(self.timed_dic[key][timestamp])

obj = TimeMap()

obj.put("fruit", "banana") # at time 1
obj.put("fruit", "apple") # at time 2
obj.put("fruit", "orange") # at time 3
obj.put("fruit", "pear") # at time 4
obj.put("fruit", "melon") # at time 5
obj.put("fruit", "jerry") # at time 6

obj.get("fruit") #=> "jerry"
obj.get("fruit", 2) # => apple
obj.get("fruit", -2) # => ""
obj.get("fruit", 4) # => "pear"
obj.get("fruit", 3) # => "orange"
obj.get("fruit", 5) # => "melon"

```

[729. My Calendar I](https://leetcode.com/problems/my-calendar-i/)

```py
from sortedcontainers import SortedList

class MyCalendar:
    """
    use SortedList() to store the (start, end) pair
    """

    def __init__(self):
        self.calendar = SortedList()

    def book(self, start: int, end: int) -> bool:
        if not self.calendar:
            self.calendar.add((start, end))
            return True
        
        pre = self.calendar.bisect_left((start, end)) - 1
        nxt = self.calendar.bisect_right((start, end))
        
        if (start, end) in self.calendar:
            return False
        
        if (pre == -1 or self.calendar[pre][1] <= start) and (nxt == len(self.calendar) or self.calendar[nxt][0] >= end):
            self.calendar.add((start, end))
            return True
        
        return False


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)
```

```py
import bisect

class MyCalendar:
    """
    Use two lists that are sorted using bisect
    each time find the starting and ending index of them and see if they are the same
    only if they are the same, means that is a valid insert
    """

    def __init__(self):
        self.starts = []
        self.ends = []

    def book(self, start: int, end: int) -> bool:
        start_idx = bisect.bisect_left(self.starts, end)
        end_idx = bisect.bisect_right(self.ends, start)
        if start_idx == end_idx:
            self.starts.insert(start_idx, start)
            self.ends.insert(end_idx, end)
            return True
        return False


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)
```

[731. My Calendar II](https://leetcode.com/problems/my-calendar-ii/)

```py
class MyCalendarTwo:
    """
    use two lists: overlaps and calendar
    for new (start, end), if intersect with the overlaps, return false
    if not: find the correct (new_start, new_end) to add to overlaps, and add(start, end) to calendar
    """
    def __init__(self):
        self.overlaps = []
        self.calendar = []

    def book(self, start, end):
        for i, j in self.overlaps:
            if start < j and end > i:
                return False
        for i, j in self.calendar:
            if start < j and end > i:
                self.overlaps.append((max(start, i), min(end, j)))
        self.calendar.append((start, end))
        return True
```

[846. Hand of Straights](https://leetcode.com/problems/hand-of-straights/)

[218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)

[480. Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)

[315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
