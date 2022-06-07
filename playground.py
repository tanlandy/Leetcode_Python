import collections
from multiprocessing.sharedctypes import Value


class TimeMap:
    last_dict = {}
    timed_dict = {}

    def put(self, key, value):
        self.last_dict[key] = value
        if key not in self.timed_dict:
            self.timed_dict[key] = {}

        self.map[key].append(value)

    def get(self, key, time = 0):
        arr = map[key]
        idx = int(time) - 1
        return arr[idx]


obj = TimeMap()

obj.put("fruit", "banana") # at time 1
obj.put("fruit", "apple") # at time 2
obj.put("fruit", "orange") # at time 3

print(obj.get("fruit")) #=> "orange"
print(obj.get("fruit", "2")) # => apple