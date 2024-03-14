# 函数的类型注解

对于函数的参数，可以使用 `:` 来指定参数的类型，使用 `->` 来指定返回值的类型。

```py
def add_nums(a: int, b: int) -> int:
    return a + b
```

对于list的类型注解，可以使用 `list[int]` 来指定list中元素的类型。

```python

def sum_nums(nums: list[int]) -> int:
    total = 0
    for num in nums:
        total += num
    return total

# 相比list，Sequence更加通用，可以用于list、tuple、str等

from typing import Sequence

def sum_nums_all(nums: Sequence[int]) -> int:
    total = 0
    for num in nums:
        total += num
    return total


```

对于dict的类型注解，可以使用 `dict[str, int]` 来指定key和value的类型。

```python

def sum_nums_dict(nums: dict[str, int]) -> int:
    total = 0
    for num in nums.values():
        total += num
    return total

```

对于要处理None的情况，可以使用 `Optional` 来指定参数的类型。相当于Union[None, list[int]]。

```python

from typing import Optional

def sum_nums_optional(nums: Optional[list[int]]) -> int:
    total = 0
    if nums is None:
        return total
    for num in nums:
        total += num
    return total

```

# 给变量作类型注解type hint

```python

users: list[str] = []
users.append("user1")

```

当函数没有返回值的时候，使用 `None` 来指定返回值的类型。

```python

def print_hello(name: str) -> None:
    print(f"Hello {name}")

```

Literal的使用，使用可以指定参数的值。

```python

from typing import Literal

def print_color(color: Literal["red", "green", "blue"]) -> None:    
    print(f"Color is {color}")
    
print_color("red")  # ok
print_color("yellow")  # error

```

给类型增加一个变量名，容易维护和理解

```python

from typing import Literal

ColorType = Literal["red", "green", "blue"]

def print_color(color: ColorType) -> None:    
    print(f"Color is {color}")

c: ColorType = "red"
print_color(c)

```

使用 `NewType` 来定义一个新的类型，可以避免使用混淆。
