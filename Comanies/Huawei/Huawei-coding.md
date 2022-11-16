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

58. 输出最小k个数

```py
size, k = list(map(int, input().split()))
num = list(map(int, input().split()))

num.sort()
for i in range(k):
    print(num[i], end = ' ')
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