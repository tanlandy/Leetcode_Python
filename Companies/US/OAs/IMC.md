# OA

类似
[780. Reaching Points](https://leetcode.com/problems/reaching-points/)

一个点(x, y)可以走到(x + c, y + c), (x + y, y), (x, x + y)，在(x + y) == sqrt的位置有障碍物，问最后能否走到(x2, y2)

动态规划求解，dict: {(x, y): can_reach}

```py
import math


def canReach(c, x1, y1, x2, y2):
    # memory. -1 not reachable. 1 reachable
    dp = {}

    def gonext(x, y):
        # reuse memory
        if (x, y) in dp:
            return dp[(x, y)]
        # overshoot
        if x > x2 or y > y2:
            dp[(x, y)] = -1
            return -1
        # blocked
        if math.isqrt(x + y) ** 2 == (x + y):
            dp[(x, y)] = -1
            return -1
        # reach target
        if x == x2 and y == y2:
            dp[(x, y)] = 1
            return 1

        # go right
        if gonext(x + y, y) == 1:
            dp[(x, y)] = 1
            return 1
        # go upper
        if gonext(x, x + y) == 1:
            dp[(x, y)] = 1
            return 1
        # go upper right
        if gonext(x + c, y + c) == 1:
            dp[(x, y)] = 1
            return 1

        dp[(x, y)] = -1
        return -1
    gonext(x1, y1)
    if (x2, y2) in dp and dp[(x2, y2)] == 1:
        return "Yes"
    return "No"


if __name__ == '__main__':
    print(canReach(1, 1, 4, 7, 6))

```

车流交汇问题

```py
import collections


def getResult(arrival, street):
    # car waiting queue
    q_main = collections.deque()
    q_ave = collections.deque()

    time = 0
    res = [0] * len(arrival)

    # init queue
    for i, s in enumerate(street):
        if s == 1:
            # ave car
            q_ave.append((i, arrival[i]))
        else:
            # main car
            q_main.append((i, arrival[i]))

    while q_main or q_ave:
        # 1st Avenue queue
        while q_ave:
            # ave car is waiting
            if q_ave[0][1] <= time:
                ave = q_ave.popleft()
                res[ave[0]] = time
                time += 1
            else:
                # no ave car is waiting
                break

        # Main St queue
        while q_main:
            # main car is waiting
            if q_main[0][1] <= time:
                main = q_main.popleft()
                res[main[0]] = time
                time += 1
            else:
                # no main car is waiting
                break

        min_time = None
        if q_ave:
            min_time = q_ave[0][1]
        if q_main:
            min_time = min(q_main[0][1], min_time) if min_time else q_main[0][1]
        # skip empty time span
        if min_time and min_time > time:
            time = min_time
    return res


if __name__ == '__main__':
    res = getResult([0, 1, 1, 3, 3], [0, 1, 0, 0, 1])
    print(res)
```
