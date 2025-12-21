def get_full_name(first_name, last_name):
    full_name = first_name.title() + " " + last_name.title()
    return full_name

print(get_full_name("john", "wick"))

"""
接收 first_name 和 last_name 参数。
通过 title() 将每个参数的第一个字母转换为大写形式。
中间用一个空格来拼接它们。
"""

def get_full_name(first_name: str, last_name: str):
    full_name = first_name.title() + " " + last_name.title()
    return full_name

print(get_full_name("john", "wick"))

def get_name_with_age(name: str, age: int):
    name_with_age = name.title() + " is " + str(age) + " years old"
    return name_with_age

print(get_name_with_age("John", 40))


def get_items(item_a: str, item_b: int, item_c: float, item_d: bool, item_e: bytes):
    return item_a, item_b, item_c, item_d, item_d, item_e

