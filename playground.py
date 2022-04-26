import collections


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

    ccounter = collections.Counter(s)
    res = min(ccounter.values())
    
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
    print(solution("aa") == 2)
    