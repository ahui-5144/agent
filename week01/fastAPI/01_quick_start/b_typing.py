from typing import List, Tuple, Set, Dict


def process_items(items: List[str]):
    for item in items:
        print(item.title())


process_items(['apple', 'banana', 'cherry'])


def process_items(items_t: Tuple[str, int, float]):
    for item in items_t:
        print(item)


process_items(("张三", 12, 13.5))

def process_items(prices: Dict[str, float]):
    for item_name, item_price in prices.items():
        print(item_name, item_price)

process_items({"苹果": 5.99, "香蕉": 3.50, "橙子": 4.25})


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

def get_person_info(one_person: Person):
    return f"{one_person.name} {one_person.age}"

one_person = Person("John", 20)
print(get_person_info(one_person))