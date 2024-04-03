# Coding Style

函数的写法

# 1 - 函数名和变量名采用小写+下划线的形式

# 2 - 使用Type Hint，标注参数和返回值的类型。详细介绍见下文

# 3 - 使用Docstring，标注函数的功能、参数和返回值。详细介绍见下文

# Code Style for signatures w/ Type Hints

# 使用了Type Hint的参数

# 没有默认值的参数，格式为：参数名: 参数类型，其中“：”前面无空格后面有一个空格，如下方的a和b参数

# 有默认值的参数，格式为：参数名: 参数类型 = 默认值，其中“：”前面无空格后面有一个空格，“=”两边都有一个空格，如下方的c参数

# 未使用Type Hint的参数

# 格式为：参数名=默认值，其中“=”两边无空格。如下方的d参数

def add_nums(a: int, b: list, c: int = 4, d=5) -> int:
    """add numbers"""
    return a + sum(b) + c + d

def get_ner(c_word: str) -> str:
    """获取该词的NER标签

    Args:
        c_word: 比如"Today"

    Returns:
        pos: 如有标签，返回"place", "people", "object", "time"之一。否则返回None

    """
    if c_word in categorydict:
        return categorydict[c_word]

Type Hint

# 对于函数的参数，可以使用 `:` 来指定参数的类型，使用 `->` 来指定返回值的类型

def add_nums(a: int, b: list) -> int:
    return a + sum(b)

output1 = add_nums(3, [1, 2])
output1_err = add_nums('3', 1)  # 会提示报错

# 对于list的类型注解，可以使用 `list[int]` 来指定list中元素的类型

def sum_nums(nums: list[int]) -> int:
    for num in nums:
        total += num
    return total

output2 = sum_nums([3, 1, 2])
output2_err = sum_nums([3, '1', 2])  # 会提示报错

# 对于dict的类型注解，可以使用 `dict[str, int]` 来指定key和value的类型

def sum_nums_dict(nums: dict[str, int]) -> int:
    total = 0
    for num in nums.values():
        total += num
    return total

ouput3 = sum_nums_dict({'Mon': 3, 'Tue': 1, 'Wed': 2})
ouput3_err = sum_nums_dict({'Mon': '3', 'Tue': [1], 'Wed': 2})  # 会提示报错

其他更加详细介绍，可以参考
<https://docs.python.org/3/library/typing.html>
<https://www.bilibili.com/video/BV11Z4y1h79y/?share_source=copy_web&vd_source=1aea27c12a97d57f180ca22afea77cce>

Doctring

# Docstring

# Python uses docstrings to document code. A docstring is a string that is the first statement in a package, module, class or function

def fetch_smalltable_rows(
    table_handle: smalltable.Table,
    keys: Sequence[bytes | str],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """

# 推荐Docstring使用Google Style，Pycharm修改方式为

# 进入Settings -> Tools下方Python Integrated Tools -> 右侧Docstrings的Docstrings format选择Google -> 下方点击apply点击OK

更加详细介绍，可以参考
<https://peps.python.org/pep-0257/> 官方文档
<https://google.github.io/styleguide/pyguide.html> 谷歌Python Style Guide章节3.8.1
