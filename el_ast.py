from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, Iterable, TypeVar, Union


@dataclass(order=True)
class CodeLocation:
    firstPos: int = -1
    lastPos: int = -1

    def is_valid(self) -> bool:
        return self.firstPos >= 0 and self.lastPos >= 0 and self.firstPos <= self.lastPos

    def length(self) -> int:
        return self.lastPos - self.firstPos + 1


@dataclass
class CodeBlock:
    loc: CodeLocation
    code: str


class OrderDirection(Enum):
    LongEntry = 1
    ShortEntry = 2
    LongExit = 3
    ShortExit = 4


class SlPtDirective(Enum):
    SetStopLoss = 1
    SetProfitTarget = 2
    SetContractOrPosition = 3


T = TypeVar("T")


class _CodeLocMap(Generic[T]):
    """
    A location-ordered store of objects which have a CodeLocation memer `loc` and a uniquely identifying member variable (e.g. a name).
    - Individual items are accessed via their identifying member value. (Alternatively by an index into the map.)
    - Iteration and index-based access is automatically in order of the items' CodeLocation `loc`.
    """

    def __init__(self, key_getter: Callable[[T], str]):
        self._key_getter = key_getter
        self._items: list[T] = []
        self._key_to_index: dict[str, int] = {}

    def add(self, item: T, replace_existing_key: bool = True) -> bool:
        key = self._key_getter(item).lower()
        if replace_existing_key or key not in self._key_to_index:
            self._items.append(item)
            self._sort_and_index(erased=False)
            return True

        return False

    def add_many(self, items: Iterable[T], replace_existing_keys: bool = True) -> bool:
        ok = True
        for item in items:
            ok = self.add(item, replace_existing_key=replace_existing_keys) and ok

        return ok

    def clear(self) -> None:
        self._items.clear()
        self._key_to_index.clear()

    def empty(self) -> bool:
        return len(self._items) == 0

    def count(self) -> int:
        return len(self._items)

    def contains(self, key: str) -> bool:
        return key.lower() in self._key_to_index

    def keys(self) -> list[str]:
        return list(self._key_to_index.keys())

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: Union[int, str]) -> T:
        if isinstance(key, int):
            return self._items[key]

        return self._items[self._key_to_index[key.lower()]]

    def _sort_and_index(self, erased: bool) -> None:
        if erased:
            self._key_to_index.clear()

        self._items.sort(key=lambda x: getattr(x, "loc", CodeLocation()).firstPos)
        for i, item in enumerate(self._items):
            self._key_to_index[self._key_getter(item).lower()] = i


@dataclass
class Variable:
    dataType: str = ""
    name: str = ""
    value: str = ""
    loc: CodeLocation = field(default_factory=CodeLocation)


class VariableDeclBlock(_CodeLocMap[Variable]):
    def __init__(self):
        super().__init__(key_getter=lambda v: v.name)
        self.loc = CodeLocation()


@dataclass
class CaseBlock:
    loc: CodeLocation = field(default_factory=CodeLocation)
    caseValue: str = ""
    code: str = ""


class SwitchBlock(_CodeLocMap[CaseBlock]):
    def __init__(self):
        super().__init__(key_getter=lambda c: c.caseValue)
        self.loc = CodeLocation()
        self.switchVar = ""


class SwitchBlockContainer(_CodeLocMap[SwitchBlock]):
    def __init__(self):
        super().__init__(key_getter=lambda s: s.switchVar)


def join_code_blocks(blocks: list[CodeBlock], delim: str = "\n\n") -> str:
    return delim.join(b.code for b in blocks)


def merge_overlapping_code_locations(blocks: list[CodeBlock]) -> None:
    blocks.sort(key=lambda b: b.loc.firstPos)
    if len(blocks) <= 1:
        return

    merged = [blocks[0]]
    for next_item in blocks[1:]:
        current = merged[-1]
        if next_item.loc.firstPos <= current.loc.lastPos:
            current.loc.lastPos = max(current.loc.lastPos, next_item.loc.lastPos)
        else:
            merged.append(next_item)

    blocks[:] = merged
