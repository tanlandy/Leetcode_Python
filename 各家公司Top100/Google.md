# OA

[2178. Maximum Split of Positive Even Integers](https://leetcode.com/problems/maximum-split-of-positive-even-integers/)

从2，4，6开始加同时finalSum-=246，直到curRes>finalSum，这个时候就res[-1]+=finalSum就可以了

```python
class Solution:
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        res = []
        curRes = 2
        
        if finalSum %2 == 0:
            while curRes <= finalSum:
                res.append(curRes)
                finalSum -= curRes
                curRes += 2
            res[-1] += finalSum
        
        return res

```

given a string S consisting of lowercase letters of the English alphabet, returns the longest consistent fragment of S which begins and ends with the same letter. If there are many possible answers you should return the one starting at the earliest position.
Examples:

1. Given S = "cbaabaab", your function should return "baabaab".

2. Given S = "performance", your function should return "erformance".

3. Given S = "cat", your function should return "c".


两个字典分别从前后来存字母，如果出现了就直接返回

```python
def findLS(s):
    di = {}
    dj = {}
    
    i, j = 0, len(s) - 1
    
    while i < j:
        if s[j] not in di and s[j] not in dj:
            dj[s[j]] = j
        if s[j] in di:
            return s[di[s[j]]:j + 1]
        if s[i] not in di and s[i] not in dj:
            di[s[i]] = i
        if s[i] in dj:
            return s[i:dj[s[i]] + 1]
        i += 1
        j -= 1
        
    if i == j:
        return s[0]

```


```python
def findLS(S):
    d = {}

    # store the furthest index of the letters in S
    for i in range(len(S) - 1, -1, -1):
        if S[i] not in d:
            d[S[i]] = i

    max_length = float("-infinity")
    best_index = 0

    # loop from the beginning of S
    for i, let in enumerate(S):
        # calculate the distance from current instance of the letter to the last
        sub_length = d[let] + 1 - i
        # only update if the distance is greater than max
        # this means we always start our answer from the earliest index possible. 
        if sub_length > max_length:
            best_index = i
            max_length = sub_length
            
    return S[best_index:d[S[best_index]] + 1]

print(findLS("performance") == "erformance")
print(findLS("adsaas") == "adsaa")
print(findLS("adsaass") == "adsaa")
print(findLS("adsaasss") == "saasss")

```


given an array consisting of N integers, returns the maximum possible number of pairs with the same sum. each array may belong to one pair only. (focus on the correctness, not the performance)

A = [1,9,8,100,2] output; 2 (A[0],A[1]) and (A[2], A[4])

A = [2,2,2,3] output; 1 (A[0], A[1])

排序之后，计算和为1-2001的每种可能的数量，取最大的
时间：O(NlogN)
空间：O(N)

```python
arr = [2,2,2,3]

a = arr
a.sort()
def kaafi_bekar(target_sum):
    i, j = 0, len(arr)-1
    counter = 0
    while(i<j):
        curr_sum = a[i]+a[j]
        if curr_sum<target_sum:
            i+=1
        elif curr_sum>target_sum:
            j-=1
        else:
            counter+=1
            j-=1
            i+=1
    return counter

max_counter = 0
for i in range(2001):
    curr = kaafi_bekar(i)
    max_counter = max(max_counter, curr)

print(max_counter)
```


Find the length of the longest substring that every character h‍as the same occurrences: input s="ababbcbc",出现次数相同的且最长的是"abab","bcbc"，长度是4

暴力解，把所有的substring都看一下，从中找出满足条件且最长的;字典：{letter: counter}，字典里面存每个字母出现的次数;看条件是否满足可以通过查看是否字典里的最大值和最小值相等

时间：O(N^2)
空间：O(N^2)

```python
def solution(s: str) -> int:
    """
    暴力解，把所有的substring都看一下，从中找出满足条件且最长的
    字典：{letter: counter}，字典里面存每个字母出现的次数
    看条件是否满足可以通过查看是否字典里的最大值和最小值相等
    
    时间：O(N^2)
    空间：O(N^2)
    """
    n = len(s)
    if n <= 1:
        return n
    counter = collections.defaultdict(int)
    curRes = 0
    res = 0
    
    for i in range(n - 1):
        counter.clear()
        for j in range(i, n):
            counter[s[j]] += 1
            if min(counter.values()) == max(counter.values()):
                curRes = j + 1 - i
                res = max(res, curRes)
            
    return res
 
 
if __name__ == "__main__":
    # assert solution("") == 0
    print(solution("ababbcbc") == 4)
    print(solution("aabcde") == 5)
    print(solution("aaaa") == 4)
    print(solution("beeebbbccc") == 9)
    print(solution("daababbd") == 6)
    print(solution("abcabcabcabcabcabcabcabcabcpabcabcabcabcabcabcabcabcabcabczabcabc") == 30)
    
    

```


# VO

[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)


时间：O(sqrt(N))
空间：O(N)

```python
class Solution:
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        res = []
        curRes = 2
        
        if finalSum %2 == 0:
            while curRes <= finalSum:
                res.append(curRes)
                finalSum -= curRes
                curRes += 2
            res[-1] += finalSum
        
        return res
```

