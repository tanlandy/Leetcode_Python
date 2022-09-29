# Practices
1. For statues = [6, 2, 3, 8], the output should be
solution(statues) = 3.

Ratiorg needs statues of sizes 4, 5 and 7.

```py
def solution(statues):
    return max(statues) - min(statues) - len(statues) + 1
```






# OAs

1. You are given an array of integers numbers and two integers left and right. You task is to calculate a boolean array result, where result[i] = true if there exists an integer x, such that numbers[i] = (i + 1) * x and left ≤ x ≤ right. Otherwise, result[i] should be set to false.

For numbers = [8, 5, 6, 16, 5], left = 1, and right = 3, the output should be solution(numbers, left, right) = [false, false, true, false, true].

```py
def solution(numbers, left, right):
    res = []
    
    for i, num in enumerate(numbers):
        a = i + 1
        x = numbers[i] // a
        if left <= x <= right and x * a == numbers[i]:
            res.append(True)
        else:
            res.append(False)
    
    return res
```

2. You are given an array of non-negative integers numbers. You are allowed to choose any number from this array and swap any two digits in it. If after the swap operation the number contains leading zeros, they can be omitted and not considered (eg: 010 will be considered just 10).

Your task is to check whether it is possible to apply the swap operation at most once, so that the elements of the resulting array are strictly increasing.

For numbers = [1, 3, 900, 10], the output should be solution(numbers) = true.

By choosing numbers[2] = 900 and swapping its first and third digits, the resulting number 009 is considered to be just 9. So the updated array will look like [1, 3, 9, 10], which is strictly increasing.



3. You are given an array of arrays a. Your task is to group the arrays a[i] by their mean values, so that arrays with equal mean values are in the same group, and arrays with different mean values are in different groups.

Each group should contain a set of indices (i, j, etc), such that the corresponding arrays (a[i], a[j], etc) all have the same mean. Return the set of groups as an array of arrays, where the indices within each group are sorted in ascending order, and the groups are sorted in ascending order of their minimum element.

a = [[3, 3, 4, 2],
     [4, 4],
     [4, 0, 3, 3],
     [2, 3],
     [3, 3, 3]]

solution(a) = [[0, 4],
                 [1],
                 [2, 3]]

```py
import collections
def solution(a):
    mean_to_idx = collections.defaultdict(list)
    
    for i in range(len(a)):
        l = a[i]
        m = sum(l) / len(l)
        mean_to_idx[m].append(i)
        
    res = []
    for val in mean_to_idx.values():
        res.append(val)
    
    return res
```

4. Given an array of integers a, your task is to count the number of pairs i and j (where 0 ≤ i < j < a.length), such that a[i] and a[j] are digit anagrams.

Two integers are considered to be digit anagrams if they contain the same digits. In other words, one can be obtained from the other by rearranging the digits (or trivially, if the numbers are equal). For example, 54275 and 45572 are digit anagrams, but 321 and 782 are not (since they don't contain the same digits). 220 and 22 are also not considered as digit anagrams, since they don't even have the same number of digits.

For a = [25, 35, 872, 228, 53, 278, 872], the output should be solution(a) = 4.

```py
import collections
def solution(a):
    digits_to_count = collections.defaultdict(int)
    
    # convert 25, 52 to 52
    for n in a:
        n = str(n) # 25 --> "25"
        digits = [int(x) for x in n] # [25]
        digits.sort(reverse = True) # [52]
        strings = [str(x) for x in digits] # ["5", "2"]
        a_string = "".join(strings) # "52"
        digits = int(a_string) # 52
        digits_to_count[digits]+= 1
    
    res = 0
    for val in digits_to_count.values():
        while val >= 2:
            val -= 1
            res += val
    
    return res    
```

