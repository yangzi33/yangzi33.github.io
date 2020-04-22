from typing import List

# Problem 1
def valid(s: str, alphabet: str) -> bool:
    for _ in s:
        if _ not in alphabet:
            return False
    return True

# Problem 2
def duplicate_cut(enzyme_list: List[str]) -> bool:
    if len(enzyme_list) <= 1:
        return False
    if enzyme_list[0] == enzyme_list[1] or enzyme_list[-1] == enzyme_list[-2]:
        return True

    for i in range(1, len(enzyme_list) - 1):
        if enzyme_list[i] == enzyme_list[i + 1] or enzyme_list[i] == enzyme_list[i - 1]:
            return True
    return False

# Problem 3
def has_last_token(L: List[str], token: str) -> List[str]:
    lst = L.copy()
    ret = []
    for i in range(len(lst)):
        lst[i] = lst[i].split(",")
        if lst[i][-1] == token:
            ret += [lst[i][-1]]
    return ret
