1. 第一题 alternate sum of digit
题目：给一个数，依次交换每个位置的符号，最后求和
思路：先把数字转换成str，然后根据index来加或减

```py
def solution(n):
    n = str(n)
    res = 0
    
    for idx, c in enumerate(n):
        if idx % 2: # at even index
            res -= int(c)
        else:
            res += int(c)
    
    return res
```

2. 第二题 replace consonant letter with the next consonant letter
题目：给一个string，要求把string里的辅音字母替换成对应的下一个辅音字母。
注意：

- 不用替换其他字母，只需要修改辅音字母。
- 如果一个辅音字母的下一个字母是元音字母，比如"d"，那就把"d"替换成"f"
思路：
- 先通过元音字母和pytho的isalnum()函数，判断当前是否是辅音字母。
- "z"或"Z"是特例，遇到之后直接替换成题目要求的"b"或“B”
- 如果辅音字母的下一个字母是元音字母，就要移动2位；否则只需要移动1位

```py
def solution(message):
    res = []
    
    for ch in message:
        if ch in "aeiouAEIOU" or not ch.isalnum():
            res.append(ch)
            continue

        if ch == "z":
            res.append("b")
            continue
        if ch == "Z":
            res.append("B")
            continue
        
        pos = ord(ch)
        if ch in "dDhHnNtT": # "eiou"
            pos += 2
        else:
            pos += 1
        new_ch = chr(pos)
        res.append(new_ch)
    
    return "".join(res)
```

3. 第三题 count ids
题目：给一串msgs，对于每个msg，里面包含了@id1,id3,id5这样的内容，求所有msgs中不同id出现的次数

input: msgs = [
"Hi @id1, id3, id8, today is a good day @id1",
"oh id2",
"see ya @id8"
],
members = ["id1", "id2", "id8"]

output: ["id8=2", "id1=1", "id2=0"]

解释：id1在第一个msg出现，同一个msg重复出现只计1次
id8在第一和第三msg出现，一共2次
id2格式不对，没有@，所以是0次
最后输出按照出现次数从大到小排序，如果相同则按照字母序排序

思路：维护一个nameToCount的字典，同一个msg用set来记录出现的名字，把nameToCount的内容都加到maxHeap里，然后根据count来一个一个行程最终的result
注意！这个方法最后是210/300，出现的问题是在["id300=0","id30=0"]这种情况，要求输出的顺序是["id30=0","id300=0"]，而这个方法的顺序是["id300=0","id30=0"]，不知道大家有没有什么更好的办法来解决这个排序的问题

```py
import heapq
def solution(members, messages):
    name_count = dict()
    
    for member in members:
        name_count[member] = 0
        
    for msg in messages:
        words = msg.split(" ")
        seen = set()
        for word in words:
            if word[0] != "@":
                continue
            new_word = word[1:]
            nums = new_word.split(",")
            for num in nums:
                if num in name_count:
                    seen.add(num)
        for num in seen:
            name_count[num] += 1
    
    res = []
    
    max_heap = []
    for name, count in name_count.items():
        heapq.heappush(max_heap, (-count, str(name) + "=" + str(count)))
    
    while max_heap:
        count, val = heapq.heappop(max_heap)
        res.append(val)
    
    return res
```

4. 第四题 count coverage
题目：坐标系中给很多点，每个点以自己为中心辐射2*2的范围，求发生重叠的点有多少对
思路：BF方法是排序后每个点和下一个点来看是否重叠了。我后来想办法跳过了当前重复的点，但是没有跳过重复比较下一个点，所以最后TLE只有140/300分

```py
def solution(centers):
    centers.sort()
    res = 0
    size = len(centers)
    idx = 0
    while idx < size:
        same = 1
        while idx < size - 1 and centers[idx] == centers[idx + 1]:
            idx += 1
            same += 1
            
        point1 = centers[idx]
        x1, y1 = point1[0], point1[1]

        for i in range(idx + 1, size):
            point2 = centers[i]

            x2, y2 = point2[0], point2[1]
            if abs(x1 - x2) <= 2 and abs(y1 - y2) <= 2:
                res += same
        idx += 1
        same -= 1
        while same > 0:
            res += same
            same -= 1
    
    return res
```
