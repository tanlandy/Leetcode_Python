
import collections


d = collections.defaultdict(int)
d["df"] = 9
d["df2"] = 9
d["df3"] = 9
d["df4"] = 9
d["d5"] = 9
d["d6"] = 9
d["df7"] = 9
d["df9"] = 9
d["df9"] = 9

print(max(d.values())==min(d.values()))