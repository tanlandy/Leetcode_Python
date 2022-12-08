"""
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

示例 1：

输入：x = 123
输出：321
示例 2：

输入：x = -123
输出：- 321
示例 3：

输入：x = 120
输出：21

示例 4：

输入：x = 0
输出：0
"""
import queue


def reverse_num2(num):
    if num == 0:
        return 0

    num = str(num)
    num = [x for x in num]  # ["1", "2", "3"]

    sign = 1
    i = 0
    if num[0] == "-":
        sign = -1
        i = 1

    res = []
    while i < len(num):
        if num[i] == "0":
            i += 1

    j = len(num) - 1
    while j > i:
        res.append(num[j])
        j -= 1

    # [3, 2, 1] -> 321
    # s_num = ["x" for x in res]  # ["3", "2", "1"]
    num = "".join(res)  # ["321"]
    num = int(num)  # 321
    return num if sign else -num


def reverse_num(num):
    num = str(num)
    num = list(num)
    sign = 1
    if num[0] == "-":
        sign = -1
    num = num[::-1]
    if sign == -1:
        num.pop()
        return "-" + "".join(num)
    return "".join(num)


print(reverse_num(-2314))
