1. random writer
要求实现一个function take two input （String paragraph, int length）返回 string.  这个paragraph 由一些单词和空格组成，随机选择paragraph中个一个单词， 下一个单词从这个随机选择单词右边接邻单词选，这个右边的单词可能在paragraph中出现不只一次，要求从这些相同的单词里面随机选个单词作为下个单词, 直到单词个数和给定length一样长。. .и
example:
String paragraph = “this is a sentence it is not a good one and it is also bad”
Int length = 5
如果随机选择了sentence作为第一个单词， it 是选择的下一个单词，但是it出现两次，从两个it中随机选择任意一个单词作为下一个单词，直到单词的长度达到length。
Output: sentence it is also bad

```py
def random_idx(idxs: list) -> int:
    """
    randomly return an element from a list idxs
    """
    idx = randint(0, len(idxs) - 1) # left, right both are inclusive
    return idxs[idx]

def random_generator(msg, size):
    msg = msg.split() #['hello', 'this', 'is', 'a', 'flexport', 'interview', 'and', 'this', 'is', 'a', 'hello', 'and', 'that', 'was', 'cool']
    
    # word_idxs: {"this": [2, 7], "flexport": [4]}
    word_idxs = collections.defaultdict(list)
    for idx, word in enumerate(msg):
        word_idxs[word].append(idx)
        
    res = []
    idx = randint(0, len(msg) - 1) # generate first item
    res.append(msg[idx])
    count = 1

    while count < size:
        next_word = msg[(idx + 1) % len(msg)] # get next_word based on the previous one
        idxs = word_idxs[next_word] # get the idxs
        idx = random_idx(idxs)
        res.append(msg[idx])
        count += 1
    
    return " ".join(res)

msg = "hello this is a flexport interview and this is a hello and that was cool"
size = 4
print(random_generator(msg, size)) # "a flexport interview and"
```

We'll pick k random start words and then random successive words and output n words in total
text = "this is a sentence it is not a good one and it is also bad"
input: Integer, Integer, String
output: String
e.g. input: k = 2, n = 5, text
     possible output: a good one and it
     possible output: it is not a good
     wrong output: it is a sentence it
说明：首次随机选择连续的 k 个单词，然后每次随机选择这 k 个单词的 successive word
举例：k = 2，首次选择 it is，然后随机选择 not 或者 also，假设选择 also，然后再选择 is also 的 successive word
思路：hashmap: String -‍‌‌‌‌‍‍‍‍‍‌‍‌‍‌‍‌‍‌‌> List<String> 记录每 k 个单词的所有successive words
用StringBuilder作为队列，每次删掉最前面的单词，再加入新单词


2. Valid IP Addresses
一串IP检测是不是valid, valid指是否有四个数字部分，数字部分要在[0-255]范围eg:
“12.123.1.213” true
"0..12.324" false
第二道
给一个数字string，它可以组成多少个valid的 IP, 输出
eg:
input
"00123"
output:
"0.0.1.23"
"0.0.12.3"

[468. Validate IP Address](https://leetcode.com/problems/validate-ip-address/)
```py
class Solution:
    def validIPAddress(self, queryIP: str) -> str:
        def valid_v4(queryIP):
            nums = queryIP.split(".")
            for n in nums:
                if len(n) == 0 or len(n) > 3:
                    return False
                if n[0] == "0" and len(n) != 1 or not n.isdigit() or int(n) > 255 :
                    return False
            return True

        def valid_v6(queryIP):
            nums = queryIP.split(":")
            hexdigits = "0123456789abcdefABCDEF"
            for n in nums:
                if len(n) == 0 or len(n) > 4 or not all(c in hexdigits for c in n):
                    return False
            return True
        
        if queryIP.count(".") == 3:
            return "IPv4" if valid_v4(queryIP) else "Neither"
        elif queryIP.count(":") == 7:
            return "IPv6" if valid_v6(queryIP) else "Neither"
        else:
            return "Neither"
```

[93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)

```py
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        self.backtrack(res, "", 0, s)
        return res
    
    def backtrack(self, res, path, idx, s):
        if idx > 4:
            return
        
        if idx == 4 and not s:
            res.append(path[:-1])
        
        for i in range(1, len(s) + 1):
            if s[:i] == "0" or (s[0] != "0" and 0 < int(s[:i]) <= 255):
                self.backtrack(res, path + s[:i] + ".", idx + 1, s[i:])
```


[943. Find the Shortest Superstring](https://leetcode.com/problems/find-the-shortest-superstring/)

[706. Design HashMap](https://leetcode.com/problems/design-hashmap/description/)

```py
class Bucket:
    def __init__(self):
        self.bucket = []

    def get(self, key):
        for (k, v) in self.bucket:
            if k == key:
                return v
        return -1

    def update(self, key, value):
        found = False
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                self.bucket[i] = (key, value)
                found = True
                break

        if not found:
            self.bucket.append((key, value))

    def remove(self, key):
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                del self.bucket[i]


class MyHashMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # better to be a prime number, less collision
        self.key_space = 2069
        self.hash_table = [Bucket() for i in range(self.key_space)]


    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: None
        """
        hash_key = key % self.key_space
        self.hash_table[hash_key].update(key, value)


    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        hash_key = key % self.key_space
        return self.hash_table[hash_key].get(key)


    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: None
        """
        hash_key = key % self.key_space
        self.hash_table[hash_key].remove(key)

```

[17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/)

```py
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        d_ch = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "qprs",
            "8": "tuv",
            "9": "wxyz" 
        }
        
        res = []

        def backtrack(one_res, idx):
            if len(one_res) == len(digits):
                res.append("".join(one_res))
                return
            
            for ch in d_ch[digits[idx]]:
                one_res.append(ch)
                backtrack(one_res, idx + 1)
                one_res.pop()
        
        backtrack([], 0)
        return res
```

[348. Design Tic-Tac-Toe](https://leetcode.com/problems/design-tic-tac-toe/description/)

```py
class TicTacToe:

    def __init__(self, n: int):
        self.n = n
        self.hori = [0] * n
        self.ver = [0] * n
        self.diag1 = 0
        self.diag2 = 0
        self.board = [["." for i in range(n)] for j in range(n)]        

    def move(self, row: int, col: int, player: int) -> int:
        n = self.n
        move = 1
        if player == 2:
            move = -1
        
        self.hori[col] += move
        self.ver[row] += move

        if row == col:
            self.diag1 += move
        if row + col == (n - 1):
            self.diag2 += move
        
        if abs(self.hori[col]) == n or abs(self.ver[row]) == n or abs(self.diag1) == n or abs(self.diag2) == n:
            return player
        self.show_status(row, col, player)
        return 0

    def show_status(self, row, col, player):
        if player == 1:
            self.board[row][col] = "X"
        else:
            self.board[row][col] = "O"
        print(self.board)
```

[62. Unique Paths](https://leetcode.com/problems/unique-paths/)
```py
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """
        number of path to a cell = number of path to its left + to its tops
        dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        base case: dp[r][c]的第一行和第一列都是1
        最后返回dp[-1][-1]
        """

        dp = [[0 for _ in range(n)] for _ in range(m)]

        for c in range(n):
            dp[0][c] = 1
        for r in rnage(m):
            dp[r][0] = 1
        
        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        
        return dp[-1][-1]
```

[91. Decode Ways](https://leetcode.com/problems/decode-ways/)

```py
class Solution:
    def numDecodings(self, s: str) -> int:
        """
        BF: when str has more than two digits: draw a desicion tree
        Example: "121" can only branch to 1-26 -> O(2^N)
                 121
             /          \
            1            12
          /   \         /
         2    21       1
        /
        1

        subproblem: once solve 21, the subproblem is 1, solve from right to left
        dp[i] = dp[i + 1] + dp[i + 2]

        Time: O(N)
        Space: O(N), O(1) if only use two variables
        """
        dp = [1] * (len(s) + 1)

        for i in range(len(s) - 1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]

            if ((i + 1) < len(s)) and ((s[i] == "1") or s[i] == "2" and s[i + 1] in "0123456"): # double digit
            # if 10 <= int(s[i:i+2]) <= 26:
                dp[i] += dp[i + 2]
        
        return dp[0]
```


第一题：假设有些order还有一些flight，根据order选flight。
每个order会有出发日期和到达日期，flight有出发日期和到达日期，每个flight有限载量
第二题：
类似于这样
map=
1："a" "b"
2: "c"‍‌‌‌‌‍‍‍‍‍‌‍‌‍‌‍‌‍‌‌
get_nums=
11: "aa" "ab" "ba" "bb"
12: "ac" "bc"

题目是面经题 checker board
part1: given a state of the board and a player, get all possible next moves of all checkers of the player
part2: given a move, make a movement and update the board

1. 给你一个webpage和里面有的link (link是另一个webpage), 让你求出通过第一个webpage能访问到的所有webpage的size, webpage里有webpage的size信息Follow up: 给一个有所有node的set, 如何求root webpage
2。check 版本号 输入是两个string num followed by dot followed by int 看哪个更新。follow up：如何validate string


1. 给一个byte[] read（） 让你implement byte read(int size)来read给定size的数据  让你考虑各种可能性follow up: 如果数据是  1 3 2 4 0 1要你decode 成 3 4 4 （要考虑数据很大的情况，不能直接copy，要使用iterator）follow up反过来encode题都不难 但每个题都要分析时间复杂度空间复杂度以及各种corner case和优化

3 opentable 设计，就是设计data model
4 http协议，网站访问过程，bq



电面是 这里一道题 checker game
vo四轮
1. project deep dive 需要画 diagram说明
2. coding 是一道oop 这里没有出现过 有一个shipment class有weight这个filed 然后不同weight区间对应不同rate 需要计算一个 shipment的价格这样。。
3.debug  基本上就是string里的特殊字 replace掉 并不难但我脑子仿佛在这里卡住了= = 场面一度十分尴尬


VO:
1. checker game 没有要求写连吃的operation 提前10分钟就结束了
2. design open table 万年不变



中高难度
沟通max
讲中文
coderpad


Token card games
Given card with cost in terms of tokens. For eg to buy some Card A, you need 3 Blue tokens and 2 Green tokens. Tokens can be of Red, Green, Blue, Black or White color.
Now there is player who is holding some tokens. For eg player has 4 Blue tokens and 2 Green tokens, then player can buy above Card A. Lets say if player only has 2 Blue tokens and 2 Green tokens, then player can not buy Card A above as player is short of 1 Blue token.
Write a method that returns true if player can buy the card or false otherwise.
More examples :
Cost of Card : 2 White, 1 Black and 4 Blue.--
If Player has : 2 White, 2 Black and 4 Blue, method will return true
If Player has : 2 White, 2 Green and 4 Blue, method will return false
1、Implement canPurchase and purchase
2、Discount with card owned
(比如手里有三张 Red Card，下一张待购买的卡片 cost 需要 N 个 Red Money，实际购买时 只需要支付 N-3 个 Red Money 就行)
cost: {"Red":4}
player:{"Red":7,"Bule":5}
purchase()
player:{"Red":3,"Bule":5}
playerCard:{"Red":1}
purchase()
player:{"Red":0,"Bule":5}
playerCard:{"Red":2}


每个项目都过了一遍

英语自我介绍


reverse linklist