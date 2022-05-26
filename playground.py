import collections


h = collections.defaultdict(int)
h[9] = 5
bulls = 0
bulls += int(h[9] > 0)
res = int(True)
print(res)