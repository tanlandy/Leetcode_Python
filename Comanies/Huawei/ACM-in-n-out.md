# 输入一行
input: hello nowcoder
```py
last = input().split()[-1]
msg = input() # msg = "hello nowcoder"
```

# 输入两行
## 依次输入两行
input: ABCabc
       A
```py
msg = input() # 每次input()都是获得这次input的行数据
need = input()
```

## 第一行是第二行的信息
input: 5 2
       1 3 5 2 4 
```py
size, k = list(map(int, input().split())) # 使用map函数，分别实施int(input().split())
num = list(map(int, input().split())) # 输入的是string，所以同上处理得到数组
```


# 输入n行
## 第一行是说明接下里有几行
input: 4
       0 1
       0 2
       1 3
       3 4
```py
# 先把第一行的数据拿出来，然后用for循环分别处理每一行的数据
size = int(input()) # size = 4
for i in range(size):
    line = input().split() # line = ['0', '1']

```

## 最后一行说明结束
input: 3
       10
       2
       0
```py
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
