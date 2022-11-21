华为自有考试平台“时习知”练习地址https://shixizhi.huawei.com/iexam/1366212223726481409/open/examInfo?examId=1401820605984784386&tenant_id=1366212223726481409
牛客网练习地址：https://www.nowcoder.com/ta/huawei

第一个是模拟地址，第二个是可以刷题的

💯华为23届校招机考
1⃣️考试有效时间段:逢国内周三晚19:00--周四晚19:00
2⃣️考试时长:2小时

【机考安排】
【短信/邮箱】发送机考链接:华为自有“时习知平
台3道编程题【练习地址】
牛客网练习，所有题型都有
www.nowcoder.com/ta/huawe
leetcode也可以自己找题练>
【机考时间】
120分钟，满分600分，100合格【卷面3道题】
第1道题100分;第2道题200分;第3道题300分。考试语言自行选择，注:可使用本地IDE调试(建议在线编译)，但需提前准备好调试语言。

请注意:
1.软件编程题考试为ACM模式，需要处理输入输
出，请提前练习熟悉该模式。
2.练习/正式考试时务必点击【保存并调试】，同时
也可以多次点击【保存并调试】随时查看通过率。

1. 字符串最后一个单词的长度

```py
last = input().split()[-1]
print(len(last))
```

2. 计算某字符出现次数

```py
from collections import Counter

msg = input().upper()
freq = Counter(msg)
need = input().upper()
print(freq[need])
```


3. 明明的随机数

```py
while True:
    try:
        n = int(input())
        nums = []
        for i in range(n):
            nums.append(int(input()))
        uni_nums = set(nums)
        nums = list(uni_nums)
        nums.sort()
        for n in nums:
            print(n)
    except:
        break
```

4. 字符串分隔
```py
msg = input()

while len(msg) %8 != 0:
    msg += "0"

for i in range(0, len(msg) - 7, 8):
    print(msg[i: i + 8])
```

5. 进制转换

16进制转为10进制

原理是res = res * BASE + hexToDec(ch)

```java
import java.util.*;

public class Main {

    private final static int BASE = 16;

    private static Map<Character, Integer> hexToDec = new HashMap<Character, Integer>() {{
        put('0', 0);
        put('1', 1);
        put('2', 2);
        put('3', 3);
        put('4', 4);
        put('5', 5);
        put('6', 6);
        put('7', 7);
        put('8', 8);
        put('9', 9);
        put('A', 10);
        put('B', 11);
        put('C', 12);
        put('D', 13);
        put('E', 14);
        put('F', 15);
        put('a', 10);
        put('b', 11);
        put('c', 12);
        put('d', 13);
        put('e', 14);
        put('f', 15);
    }}; 

    public static int toDec(String number) {
        int res = 0;
        for (char ch : number.toCharArray()) {
            res = res * BASE + hexToDec.get(ch);
        }
        return res;
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNext()) {
            String number = in.next();
            int res = toDec(number.substring(2)); // 因为16进制在Java，Python表示时使用字首"0x"，所以从第三位开始转换
            System.out.println(res);
        }
    }
}

```

Python可以用内置函数进行转换
16 -> 10
res = int('ff', 16)

10 -> 16
res = hex(16)

10 转为 16进制的原理：
1999 % 16 余数就是位置上的数

6. 质数因子

prime从2开始直到sqrt(num) + 1，分别往上除，不要忘了最后剩下的那个数

```py
import math
num = int(input())

for i in range(2, int(math.sqrt(num)) + 1):
    while num % i == 0:
        print(i, end = ' ') # end是结束的符号，默认是"\n"，即自动换行
        num //= i # 这样最后一个数就是不含小数的整数

if num > 2:
    print(num)
```

7. 取近似值

```py
num = float(input())
print(int(num + 0.5)) # int()中参数是float时，直接去尾
```

```py
num = float(input())
res = num // 1
tep = num - res
if tep >= 0.5:
    print(int(res + 1))
else:
    print(int(res))

```

8. 合并记录

```py

size = int(input())
dict = dict()

for i in range(size):
    line = input().split()

    key = int(line[0])
    val = int(line[1])
    dict[key] = dict.get(key, 0) + val

for key in sorted(dict):
    print(key, dict[key])

```

9. 提取不重复的整数

```py
num = list(input())

seen = set()
res = []
for i in range(len(num) - 1, -1, -1):
    n = num[i]
    if n not in seen:
        res.append(n)
        seen.add(n)
    else:
        continue        

res = "".join(res)
print(res)
```


10. 字符个数统计

```py
def count_ch(str):
    string = "".join(set(str))
    count = 0
    for ch in string:
        if 0 <= ord(ch) <= 127:
            count += 1
    return count

str = input()
print(count_ch(str))
```

11. 数字颠倒
```py
num = list(input())

print("".join(reversed(num))) # 或者直接 print(input()[:: -1])
```

12. 字符串反转
同11题

13. 句子逆序

```py
s = input().split()
s = s[::-1]

for word in s:
    print(word, end = ' ')
```

14. 字符串排序

```py
size = int(input())
res = []
for i in range(size):
    word = input()
    res.append(word)

res.sort()

for i in range(size):
    print(res[i])
```

15. 求2进制下1的个数

```py
num = int(input())

print(bin(num).count('1')) # bin(num) 转换成二进制数
```

16. 购物车

```py
n, m = map(int,input().split())
primary, annex = {}, {}
for i in range(1,m+1):
    x, y, z = map(int, input().split())
    if z==0:#主件
        primary[i] = [x, y]
    else:#附件
        if z in annex:#第二个附件
            annex[z].append([x, y])
        else:#第一个附件
            annex[z] = [[x,y]]
m = len(primary)#主件个数转化为物品个数
dp = [[0]*(n+1) for _ in range(m+1)]
w, v= [[]], [[]]
for key in primary:
    w_temp, v_temp = [], []
    w_temp.append(primary[key][0])#1、主件
    v_temp.append(primary[key][0]*primary[key][1])
    if key in annex:#存在主件
        w_temp.append(w_temp[0]+annex[key][0][0])#2、主件+附件1
        v_temp.append(v_temp[0]+annex[key][0][0]*annex[key][0][1])
        if len(annex[key])>1:#存在两主件
            w_temp.append(w_temp[0]+annex[key][1][0])#3、主件+附件2
            v_temp.append(v_temp[0]+annex[key][1][0]*annex[key][1][1])
            w_temp.append(w_temp[0]+annex[key][0][0]+annex[key][1][0])#3、主件+附件1+附件2
            v_temp.append(v_temp[0]+annex[key][0][0]*annex[key][0][1]+annex[key][1][0]*annex[key][1][1])
    w.append(w_temp)
    v.append(v_temp)
for i in range(1,m+1):
    for j in range(10,n+1,10):#物品的价格是10的整数倍
        max_i = dp[i-1][j]
        for k in range(len(w[i])):
            if j-w[i][k]>=0:
                max_i = max(max_i, dp[i-1][j-w[i][k]]+v[i][k])
        dp[i][j] = max_i
print(dp[m][n])
```

17. 坐标移动

```py
cmds = input().split(';')
x, y = 0, 0
for cmd in cmds:
    if not 2 <= len(cmd) <= 3:
        continue
    dir = cmd[0]
    move = cmd[1:]
    if not dir.isalpha() or not move.isdigit():
        continue
    move = int(move)
    if dir == "A":
        x -= move
    elif dir == "D":
        x += move
    elif dir == "S":
        y -= move
    elif dir == "W":
        y += move
print(str(x) + ',' + str(y))
```

18. 识别有效ip地址和子网掩码并统计

```py
import sys

res = [0,0,0,0,0,0,0]

def puip(ip):
    if 1 <= ip[0] <= 126:				# A类地址判断条件
        res[0] += 1
    elif 128 <= ip[0] <= 191:			# B类地址判断条件
        res[1] += 1
    elif 192 <= ip[0] <= 223:			# C类地址判断条件
        res[2] += 1
    elif 224 <= ip[0] <= 239:			# D类地址判断条件
        res[3] += 1
    elif 240 <= ip[0] <= 255:			# E类地址判断条件
        res[4] += 1
    return

def prip(ip):			# 私有IP地址判断条件
    if (ip[0] == 10) or (ip[0] == 172 and 16 <= ip[1] <= 32) or (ip[0] == 192 and ip[1] == 168):
        res[6] += 1
    return

def ym(msk):
    val = (msk[0] << 24) + (msk[1] << 16) + (msk[2] << 8) + msk[3]	# 获取32位的掩码表示
    s = bin(val)[2:]												  # 去除“0b”字符，并转换成字符串
    pos0 = s.find('0')												  # 从左往右找到0第一次出现的位置 
    pos1 = s.rfind('1')												  # 从右往左找到1第一次出现的位置
    if pos0 != -1 and pos1 != -1 and pos0 - pos1 == 1:				  # 判断两个位置是否相差1，且是否找不到
        return True
    return False
    

def judge(line):
    ip, msk = line.strip().split('~')
    ips = [int(x) for x in filter(None, ip.split('.'))]				# 获得表示IP的列表，理论上应该包含四个元素
    msks = [int(x) for x in filter(None, msk.split('.'))]			# 获得表示掩码的列表，理论上应该包含四个元素
    if ips[0] == 0 or ips[0] == 127:								# 排除非法IP不计数
        return
    if len(ips) < 4 or len(msks) < 4:								  # 判断错误掩码或错误IP
        res[5] += 1
        return
    if ym(msks) == True:											# 通过掩码判断的可以进行IP判断
        puip(ips)
        prip(ips)
    else:
        res[5] += 1
    return

for line in sys.stdin:
    judge(line)
# judge("192.168.0.2~255.255.255.0")

res = [str(x) for x in res]
print(" ".join(res))
```

19. 简单错误记录

```py
ls = []   # 储存键
dic = {}  # 储存键-值对
while True:
    try:
        msg = input().split()    
        msg[0] = msg[0].split('\\')[-1]        # 路径\分割，只取最后一个
        msg = ' '.join([msg[0][-16:], msg[1]]) # 取后16位及行号（str[-16],num）-> (str[-16] num)  此时属性为字符串
        if msg not in dic.keys():              # 将msg记为字典的key值并判断是否存在
            ls.append(msg)                     # 不存在就将其计入列表ls
            dic[msg] = 1                       	  # 将msg为key的value记录为1
        else:
            dic[msg] += 1                         # 存在msg就在字典中对应值增加计数
    except:
        break
for item in ls[-8:]:                           # 正序遍历后八个存储的键
    print(item, dic[item])                     # 打印键-值对
```

20. 密码验证合格程序

```py
def check(s):
        if len(s) <= 8:
            return 0

        upper_has = 0
        lower_has = 0
        num_has = 0
        other_has = 0
        dup_req = False

        for ch in s:
            if ord('a') <= ord(ch) <= ord('z'):
                upper_has = 1
            elif ord('A') <= ord(ch) <= ord('Z'):
                lower_has = 1
            elif ord('0') <= ord(ch) <= ord('9'):
                num_has = 1
            else:
                other_has = 1
        
        if upper_has + lower_has + num_has + other_has < 3:
            return 0
        
        dc = {}
        for i in range(len(s) - 2):
            if s[i:i + 3] in dc:
                return 0
            else:
                dc[s[i:i + 3]] = 1
        
        return 1

while True:
    try:
        s = input()
        if check(s):
            print("OK")
        else:
            print("NG")

    except:
        break
```

21. 简单密码

```py
string = list(input())

res = []

for ch in string:
    if ord("A") <= ord(ch) <= ord("Z"):
        ch = ch.lower()
        if ch == "z":
            res.append("a")
        else:
            res.append(chr(ord(ch) + 1))
    elif ord("a") <= ord(ch) <= ord("z"):
        if ch in 'abc':
            res.append('2')
        elif ch in 'def':
            res.append('3')
        elif ch in 'ghi':
            res.append('4')
        elif ch in 'jkl':
            res.append('5')
        elif ch in 'mno':
            res.append('6')
        elif ch in 'pqrs':
            res.append('7')
        elif ch in 'tuv':
            res.append('8')
        else:
            res.append('9')
    else:
        res.append(ch)

print("".join(res))
```

22. 汽水瓶

```py
def max_bottles(num):
    res = num // 3
    bottles = num // 3 + num % 3
    if bottles == 2:
        res += 1
    elif bottles < 2: # base case
        res += 0
    else:
        res += max_bottles(bottles)
    return res


while True:
    try:
        num = int(input())
        if num == 0:
            break
        else:
            print(max_bottles(num))
    except:
        break
```

也可以直接//2得到结果

23. 删除字符串中出现最少的字符

```py
while True:
    try:
        word = input()
        freq = dict()
        for ch in word:
            if ch in freq:
                freq[ch] += 1
            else:
                freq[ch] = 1
        to_del = min(freq.values())
        res = ''
        
        for ch in word:
            if freq[ch] != to_del:
                res += ch
        print(res)

    except:
        break
```

26. 字符串排序

```py
while True:
    try:
        s = input()
        a = ''
        for i in s:
            if i.isalpha():
                a += i
        b = sorted(a, key = str.upper) # 字符由A到Z排序，且同字符输入先后排序
        idx = 0
        res = ''
        for i in range(len(s)):
            if s[i].isalpha():
                res += b[idx]
                idx += 1
            else:
                res += s[i]
        print(res)

    except:
        break
```

27. 查找兄弟单词

```py
while True:
    try:
        msg = input().split()
        size = msg[0]
        pos = int(msg[-1])
        strs = msg[1:-2]
        std = msg[-2]
        count = 0
        valid_msg = []

        for word in strs:
            if word == std:
                continue
            elif sorted(word) == sorted(std):
                count += 1
                valid_msg.append(word)
        print(count)
        valid_msg.sort()
        print(valid_msg[pos - 1])
    except:
        break
```


29. 字符串加解密

```py
while True:
    try:
        a = input()
        a = list(a) #需要加密的字符串
        b = input()
        b = list(b) #需要解密的字符串
        for i in range(len(a)):#加密过程
            if(a[i].isupper()): #如果字符是大写字母
                if(a[i] == 'Z'): #首先如果是Z的话变为a，若z的ascll+1的值是‘{’
                    a[i] = 'a'
                else:
                    a[i] = a[i].lower() #先变为小写
                    c = ord(a[i]) + 1 #ascll码+1
                    a[i] = chr(c) #转为字符
            elif(a[i].islower()): #小写同理
                if(a[i] == 'z'):
                    a[i] = 'A'
                else:
                    a[i] = a[i].upper()
                    c = ord(a[i]) + 1
                    a[i] = chr(c)
            elif(a[i].isdigit()): #若是数字则+1，9需要单独处理
                if(a[i] == '9'):
                    a[i] = '0'
                else:
                    a[i] = int(a[i]) + 1
                    a[i] = str(a[i])
            else: #若是其他字符则保持不变
                a[i] = a[i]
                
                
        for i in range(len(b)): #解密过程
            if(b[i].isupper()): #若为大写则先变为小写，再ascll减一，A要单独处理
                if(b[i] == 'A'):
                    b[i] = 'z'
                else:
                    b[i] = b[i].lower()
                    c = ord(b[i]) - 1
                    b[i] = chr(c)
            elif(b[i].islower()): #若是小写要先变为大写，再ascll减一，a要单独处理
                if(b[i] == 'a'):
                    b[i] = 'Z'
                else: 
                    b[i] = b[i].upper()
                    c = ord(b[i]) - 1
                    b[i] = chr(c)
            elif(b[i].isdigit()):#若是数字需要减一，0要单独处理
                if(b[i] == '0'):
                    b[i] = '9'
                else:
                    b[i] = int(b[i]) - 1
                    b[i] = str(b[i])
            else:
                b[i] = b[i]
        print(''.join(a)) #按要求输出
        print(''.join(b))
    except:
        break             
```



31. 单次倒排

```py
msg = input() # I am a student
s = []
for ch in msg:
    if ch.isalpha():
        s.append(ch)
    else:
        s.append(' ')
s = ''.join(s)
s = s.split(' ')
print(s) # s = ['I', 'am', 'a', 'student']
s = s[::-1]
print(' '.join(s)) # student a am I
```


34. 图片整理

```py
msg = list(input())

msg.sort()
print("".join(msg))
```
也可以把每个ch转为ascii，排序之后，再转换回来输出


35. 蛇形矩阵

input = 4
output = 
1 3 6 10
2 5 9
4 8
7

先普通设置成
1
2 3 
4 5 6
7 8 9 10
然后每一行取最后一个元素，形成输出的行

```py
while 1:
    try:
        n = int(input())
        list1 = []
        for i in range(1, n + 1):
            list1.append([0] * i)
        a = 0
        for i in range(n):
            for j in range(i + 1):
                a += 1
                list1[i][j] = a
        
        list2, res = [], []
        for i in range(1, n + 1):
            for line in list1:
                if line:
                    list2.append(line.pop())
            res.append(' '.join(map(str, list2)))
            list2 = []
        for i in res:
            print(i)

    except:
        break
```

37. 统计每个月兔子的总数

```py
def count(month):
    if month == 1 or month == 2:
        return 1
    month = count(month - 1) + count(month - 2)
    return month

month = int(input())
print(count(month))
```

38. 小球经历的路程

```py
while True:
    try:
        H = int(input())
        S = 0 - H
        for i in range(5):
            S += H * 2 # 以落地为准，每次都走了2个H的距离（除了一开始）
            H /= 2
        print(float(S))
        print(float(H))

    except:
        break
```

40. 统计字符

```py
while True:
    try:
        s = input()
        letter, space, num, others = 0, 0, 0, 0
        for ch in s:
            if ch.isalpha():
                letter += 1
            elif ch.isalnum():
                num += 1
            elif ch == ' ':
                space += 1
            else:
                others += 1
        print(letter)
        print(space)
        print(num)
        print(others)

    except:
        break
```

46. 截取字符串

```py
while True:
    try:
        string = input()
        size = int(input())
        print(string[:size])
    except:
        break
```

51. 输出单向链表中倒数第k个结点

```py
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        node = Node(val)
        if self.head is None:
            self.head = node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = node
    
    def remove_nth_from_last(self, n):
        dummy = Node(-1)
        dummy.next = self.head
        ptr1 = ptr2 = dummy

        while n > 0:
            ptr1 = ptr1.next
            n -= 1
        ptr1 = ptr1.next
        while ptr1:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        res = ptr2.next.val
        ptr2.next = ptr2.next.next
        return res

while True:
    try:
        size = int(input())
        vals = input().split()
        ll = LinkedList()
        pos = int(input())
        for i in range(size):
            ll.append(vals[i])
        print(ll.remove_nth_from_last(pos))
        
    except:
        break
```


53. 杨辉三角变形
从第三行开始，第一个偶数位置依次出现在2、3、2、4位
```py
alt = [2, 3, 2, 4]
while True:
    try:
        n = int(input())
        if n < 3:
            print(-1)
        else:
            print(alt[(n - 3) % 4])
    except:
        break
```

54. 表达式求值

```py
while True:
    try:
        print(int(eval(input())))
    except:
        break
```
正经面试还是要用栈


55. 挑7

```py
def count_seven(n):
    count = 0
    for i in range(7, n + 1):
        if i % 7 == 0:
            count += 1
        elif '7' in str(i):
            count += 1
    return count

while True:
    try:
        n = int(input())
        print(count_seven(n))

    except:
        break
```

56. 完全数计算

simulate
```py
n = int(input())
count = 0
for i in range(1, n):
    # 查看i是否是完全数
    sum_factor = 0
    for factor in range(1, i):
        if i % factor == 0:
            sum_factor += factor 
    if sum_factor == i:
        count += 1

print(count)

```


57. 高精度整数加法

```py
def add_string(s1, s2):

        res = []
        i = 0
        carry = 0
        val = 0
        while i < max(len(s1), len(s2)):
            a = 0 if i >= len(s1) else s1[i]
            b = 0 if i >= len(s2) else s2[i]
            val = (carry + a + b) % 10
            carry = (carry + a + b) // 10
            res.append(val)
            i += 1
        if carry:
            res.append(carry)
        return "".join(str(x) for x in res[::-1])


while True:
    try:
        s1 = list(map(int, input()))[::-1]
        s2 = list(map(int, input()))[::-1]
        print(add_string(s1, s2))

    except:
        break
```

58. 输出最小k个数

```py
size, k = list(map(int, input().split()))
num = list(map(int, input().split()))

num.sort()
for i in range(k):
    print(num[i], end = ' ')
```

60. 查找组成一个偶数最接近的两个素数

```py
import math
def is_prime(num):
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

while True:
    try:
        n = int(input())
        for i in range(2, n // 2 + 1):
            if is_prime(i) and is_prime(n - i):
                a, b = i, n - i
        print(a)
        print(b)

    except:
        break
```

61. 放苹果

```py
def count(m, n): # 返回(m, n)情况下苹果个数
    """
    分两种情况：
    1. 其中一个盘子是空的
    2. 所有盘子都有苹果
    """
    if m < 0 or n < 0:
        return 0
    elif m == 1 or n == 1:
        return 1
    else:
        return count(m, n - 1) + count(m - n, n)

while True:
    try:
        m, n = map(int, input().split())
        print(count(m, n))

    except:
        break
```

62. 查找二进制中1的个数

```py
while True:
    try:
        n = int(input())
        count = 0
        while n:
            count += (n & 1) # 看最后一位是不是1，如果是的话就+= 1
            n >>= 1 # 最后一位处理过了
        print(count)
    except:
        break
```

72. 百钱买百鸡

```py
while True:
    try:
        n = input()
        for a in range(21):
            for b in range(34):
                for c in range(101):
                    if a + b + c == 100 and 5 * a + 3 * b + 1 * c / 3 == 100:
                        print(a, b, c)
    except:
        break
```

73. 计算日期到天数转换

```py
while True:
    try:
        y, m, d = map(int, input().split())
        month = [31, 28, 31, 30, 31, 30, 31, 31,30, 31, 30, 31]
        if y % 400 == 0 or (y % 100 != 0 and y % 4 == 0): # 计算闰年的方法
            month[1] = 29
        print(sum(month[:m - 1]) + d)
    except:
        break
```

76. 尼科彻斯定理

```py
# 开始和结束的数分别就是下面的start和end
n = int(input())
start = n * (n - 1) + 1
end = n * (n + 1) - 1
arr = []
for i in range(start, end + 1, 2):
    arr.append(i)
print('+'.join(map(str, arr)))
```

80. 整型数组合并

```py
while True:
    try:
        size1, nums1, size2, nums2 = input(), list(map(int, input().split())), input(), list(map(int, input().split()))
        nums = nums1 + nums2
        nums = list(set(nums))
        nums.sort()
        print("".join(list(map(str, nums))))
    except:
        break
```

81. 字符串字符匹配

```py
while True:
    try:
        s, t = input(), input()
        if set(s) & set(t) == set(s):
            print("true")
        else:
            print("false")
    except:
        break
```

83. 二维数组操作

```py
while True:
    try:
        m, n = map(int, input().split())
        x1, y1, x2, y2 = map(int, input().split())
        i_m, i_n = int(input()), int(input())
        x, y = map(int, input().split())
        # 1，数据表行列范围都是[0,9]，若满足输出'0'，否则输出'-1'
        print('0' if (0 <= m <= 9) and (0 <= n <= 9) else '-1')
        # 2，交换的坐标行列数要在输入的表格大小行列数范围[0, m)x[0, n)内
        print('0' if (0 <= x1 < m) and (0 <= y1 < n) and (0 <= x2 < m) and (0 <= y2 < n) else '-1')
        # 3.1，插入的x坐标要在 [0, m) 范围内
        print('0' if (0 <= i_m < m) and (m < 9) else '-1')
        # 3.2，插入的y坐标要在 [0, n) 范围内
        print('0' if (0 <= i_n < n) and (n < 9) else '-1')
        # 4，要检查的位置 (x,y) 要在 [0, m)x[0, n) 内
        print('0' if (0 <= x < m) and (0 <= y < n) else '-1')
    except:
        break
```

84. 统计大写字母个数

```py
s = list(input())
res = 0
for ch in s:
    if ord("A") <= ord(ch) <= ord("Z"):
        res += 1
print(res)
```

85. 最长回文子串
把所有可能性都计算一次，每当找到一个回文字串就记录其长度
```py
while True:
    try:
        s = input()
        res = []

        for i in range(len(s)):
            for j in range(i + 1, len(s)+1):
                if s[i: j] == s[i: j][::-1]:
                    res.append(j - i)
        if res:
            print(max(res))
    except:
        break
```

86. 最大连续bit数

```py
while True:
    try:
        n = int(input())
        count = 0
        res = 0
        while n:
            if n & 1:
                count += 1
                res = max(count, res)
            else:
                count = 0
            n >>= 1
        print(res)

    except:
        break
```

87. 密码等级

```py

#密码长度得分计算
def lenscore(x):
    if 0<len(x)<=4:
        return 5
    elif 5<=len(x)<=7:
        return 10
    elif len(x)>=8:
        return 25
    else:
        return 0
    
#字母大小写得分计算
def zimuscore(x):
    x=str(x)
    a=0#计算小写个数
    b=0#计算大写个数
    for i in x:
        if i.islower():#计算小写
            a+=1
        if i.isupper():#计算大写
            b+=1
            
    if (a!=0 and b==0) or (b!=0 and a==0):#全是小写或者全是大写
        return 10
    if a!=0 and b!=0:#大小写混合
        return 20
    else:
        return 0
    
#数字得分计算：
def digtscore(x):
    x=str(x)
    a=0#计算数字个数
    for i in x:
        if i.isdigit():
            a+=1
    
    if a==1:
        return 10
    if a>1:
        return 20
    else:
        return 0
    
#符号得分计算
def fhscore(x):
    x=str(x)
    a=0#计算符号个数
    fsm="!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    for i in x:
        if i in fsm:
            a+=1
    
    if a==1:
        return 10
    if a>1:
        return 25
    else:
        return 0
    
#奖励得分计算
def jlscore(x):
    x=str(x)
    a=0#计算小写个数
    b=0#计算大写个数
    for i in x:
        if i.islower():#计算小写
            a+=1
        if i.isupper():#计算大写
            b+=1
            
    if ((a!=0 and b==0) or (b!=0 and a==0)) and digtscore(x)!=0:#字母加数字
        return 2
    if ((a!=0 and b==0) or (b!=0 and a==0)) and digtscore(x)!=0 and fhscore(x):#字母加数字加符号
        return 3
    if (a!=0 and b!=0) and digtscore(x)!=0 and fhscore(x):#大小写字母加数字加符号
        return 5
    else:
        return 0
    
while True:
    try:
        a=str(input())
        countscore=lenscore(a)+zimuscore(a)+digtscore(a)+fhscore(a)+jlscore(a)
        #print(countscore)
        if countscore>=90:
            print("VERY_SECURE")
        if 80<=countscore<90:
            print("SECURE")
        if 70<=countscore<80:
            print("VERY_STRONG")
        if 60<=countscore<70:
            print("STRONG")
        if 50<=countscore<60:
            print("AVERAGE")
        if 25<=countscore<50:
            print("WEAK")
        if 0<=countscore<25:
            print("VERY_WEAK")
            
    except:
        break
```

91. 走方格的方案数

```py
while True:
    try:
        n, m = map(int, input().split())
        dp = [[1 for i in range(n + 1)] for j in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        print(dp[m][n])
    except:
        break
```

94. 记票统计

```py
while True:
    try:
        size1, candidates, size2, polls = input(), input().split(), int(input()), input().split()
        valid_count = 0
        for i in candidates:
            valid_count += polls.count(i)
            print(i + " : " + str(polls.count(i)))
        print("Invalid : " + str(size2 - valid_count))

    except:
        break
```

95. 人民币转换

```py
def fun(n, s =''):
    if n < 20:  # 由于10应写作“拾”，所以第一前1-19进行查字典处理
        s += dic[n]
    elif n < 100 :  # 大于20小于100的数
        if n % 10 >= 1:  # 非整十
            s += fun(n//10) + '拾' + fun(n % 10)
        else:  # 整十
            s += fun(n//10) + '拾'
    elif n < 1000: # 大于100小于1000的数
        if n % 100 >= 10:  # 十位不为0
            s += fun(n//100) + '佰' + fun(n % 100)
        elif n % 100 > 0:  # 个位不为零
            s += fun(n//100) + '佰零' + fun(n % 100)
        else:  # 个位为零
            s += fun(n//100) + '佰'
    elif n < 10000:  # 大于1000小于10000的数
        if n % 1000 >= 100:  # 百位不为零
            s += fun(n//1000) + '仟' + fun(n % 1000)
        elif n % 1000 > 0:  # 个位不为0
            s += fun(n//1000) + '仟零' + fun(n % 1000)
        else:  # 个位为0
            s += fun(n//1000) + '仟'
    elif n < 100000000:  # 大于10000小于100000000的数
        if n % 10000 >= 1000:  # 千位不为0时
            s += fun(n//10000) + '万' + fun(n % 10000)
        elif n % 10000 > 0:  # 个位不为0
            s += fun(n//10000) + '万零' + fun(n % 10000)
        else:  # 个位为0
            s += fun(n//10000) + '万'
    else:  # 大于100000000的数
        if n % 100000000 >= 10000000:  # 千万位不为0
            s += fun(n//10000) + '亿' + fun(n % 100000000)
        elif n % 100000000 > 0:  # 个位不为0
            s += fun(n//100000000) + '亿零' + fun(n % 100000000)
        else:  # 个位为0
            s += fun(n//100000000) + '亿'
    return s

while True:
    try:
        dic = {1:'壹', 2:'贰', 3:'叁', 4:'肆', 5:'伍', 6:'陆', 7:'柒', 8:'捌', 9:'玖', 10:'拾', 11:'拾壹', 12:'拾贰', 13:'拾叁', 14:'拾肆', 15:'拾伍', 16:'拾陆', 17:'拾柒', 18:'拾捌', 19:'拾玖'}
        n, f = map(int,input().split('.'))
        if n > 0:
            s = '人民币' + fun(n) + '元'
        else:
            s = '人民币'
        if f == 0:
            s += '整'
        elif f < 10:
            s += dic[f] + '分'
        elif f % 10 == 0:
            s += dic[f//10] + '角'
        else:
            s += dic[f//10] + '角' + dic[f % 10] + '分'
        print(s)
    except:
        break
```

96. 表示数字

```py
"""
同时看当前和前面的ch
"""

while True:
    try:
        s = input()
        res = ""
        pre_ch = ""
        for ch in s:
            if ch.isdigit(): # 这个是数字
                if not pre_ch.isdigit(): # 如果之前的不是，就加*
                    res += "*"
            else:
                if pre_ch.isdigit(): # 如果之前是，就加*
                    res += "*"
            res += ch
            pre_ch = ch
        if pre_ch.isdigit(): # 最后一个是数字
            res += "*"
        print(res)

    except:
        break
```

97. 记负均正

```py
while True:
    try:
        size, nums = int(input()), list(map(int, input().split()))
        neg_count = 0
        pos_sum = 0
        pos_count = 0
        for n in nums:
            if n < 0:
                neg_count += 1
            elif n > 0:
                pos_sum += n
                pos_count += 1
        print(neg_count, end = ' ')
        if pos_count == 0: 
            print(0.0)
        else:
            print(format(pos_sum / pos_count, '.1f')) # 分母有0的可能性，一定要考虑到

    except:
        break
```


99. 自守数
按照题目意思一个一个确定
```py
while 1:
    try:
        n = int(input())
        count = 0
        for i in range(n + 1):
            if str(i) == str(i ** 2)[-len(str(i)):]:
                count += 1
        print(count)
    except:
        break
```

100. 等差数列

```py
n = int(input())
last = 2 + 3 * (n - 1)
total = (2 + last) * n // 2
print(total)
```

101. 排序后输出
ascending
descending
```py
size = int(input())
nums = list(map(int, input().split()))
descending = int(input())

if descending:
    nums.sort(reverse = True)
else:
    nums.sort()

for n in nums:
    print(n, end = ' ')
```

102. 字符统计

```py
while True:
    try:
        ori = input()
        s = sorted(set(ori))
        res = sorted(s, key = lambda x: ori.count(x), reverse = True)
        print("".join(res))

    except:
        break
```

103. Redraiment的走法

就是求最长上升子序列
```py
"""
dp[i]: 重点为i时，最大走法
"""
while True:
    try:
        n = int(input())
        s = [int(x) for x in input().split()]
        dp = [1] * n

        for i in range(n):
            for j in range(i):
                if s[i] > s[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        print(max(dp))
    except:
        break
```

105. 记负均正

```py
neg = []
non_neg = []

while True:
    try:
        n = int(input())
        if n < 0:
            neg.append(n)
        else:
            non_neg.append(n)
    except:
        print(len(neg))

        if len(non_neg) != 0:
            # format(num, '.1f;): 只保留一位小数
            print(format(sum(non_neg) / len(non_neg), '.1f'))
        else:
            print(0.0)
        break
```

106. 字符逆序

```py
msg = input()
msg = msg[::-1]
print(msg)
```

107. 求解立方根

```py
def find_cube(n):
    sign = 1
    if n < 0:
        n = -n
        sign = -1
    
    if n > 1: # 二分查找要注意起点终点的大小
        l = 0
        r = n
    else:
        l = n
        r = 1
    mid = (l + r) / 2
    while abs(mid ** 3 - n) >= 0.0001:
              
        if mid ** 3 > n:
            r = mid
        else:
            l = mid
        mid = (l + r) / 2
    res = sign * mid
    return(res)

while True:
    try:
        n = float(input().strip())
        res = find_cube(n)
        print(format(res, '.1f'))
    except:
        break
```

108. 最小公倍数

```py
# 在大的数的倍数里，找能被小的数整除的最小的数
a, b = map(int, input().split())
if a < b:
    a, b = b, a

for i in range(a, a * b + 1, a):
    if i % b == 0:
        print(i)
        break
```