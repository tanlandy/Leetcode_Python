"""

世界首富

《福布斯》杂志每年都会根据世界上最富有的人群的年度排名发布其亿万富翁名单。

现在，你需要模拟这项工作，但你只需要统计特定年龄段的富有人群。

也就是说，给定 N 个人的净资产，你必须找到在给定年龄范围内的 M 个最富有的人。



输入格式：

第一行包含一个整数 N，表示总人数。

接下来 N 行，每行包含一个人的姓名（长度不超过 8 且不包含空格的字符串），年龄（范围在(0,200] 的整数），净资产（范围在 [−10^6,10^6] 的整数）。

第二行包含三个整数，分别为：M、年龄范围最小值 和 年龄范围最大值



输出格式

接下来输出给定年龄段内最富有的 M 个人的信息，格式为：

Name Age Net_Worth



输入样例：

12

Zoe_Bill 35 2333

Bob_Volk 24 5888

Anny_Cin 95 999999

Williams 30 -22

Cindy 76 76000

Alice 18 88888

Joe_Mike 32 3222

Michael 5 300000

Rosemary 40 5888

Dobby 24 5888

Billy 24 5888

Nobody 5 0

4 15 45

输出样例：

Alice 18 88888

Billy 24 5888

Bob_Volk 24 5888

Dobby 24 5888

"""


def find_richest(queries, N, left, right):  # top N
    nums = []
    for name, age, money in queries:
        if left <= age <= right:
            nums.append((money, name, age))
    nums.sort()  # ascending

    i = len(nums) - 1
    while N > 0:
        money, name, age = nums[i]
        print(name, age, money)
        i -= 1
        N -= 1


queries = [["Alice", 18, 88888], ["Billy", 24, 5888],
           ["Bob_Volk", 24, 5888], ["Dobby", 24, 5888]]

N = 4
left, right = 14, 45

find_richest(queries, N, left, right)
