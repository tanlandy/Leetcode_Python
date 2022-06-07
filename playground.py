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
        # print("called put()")
        self.last_dic[key] = value
        self.time += 1
        if key not in self.timed_dic:
            self.timed_dic[key] = {}
        self.timed_dic[key][self.time] = value
    
    def get(self, key, timestamp=None):
        # print("called get()")
        if timestamp is None:
            print(self.last_dic[key])
            return
        if timestamp < 1:
            print("")
            return
        print(self.timed_dic[key])

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

