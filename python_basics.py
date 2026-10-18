"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiou")
    return sum(1 for ch in text.lower() if ch in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number:
        count += number & 1
        number >>= 1
    return count


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        prod = 1
        for d in str(number):
            prod *= int(d)
        number = prod
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    d = 2
    while d * d <= number:
        if number % d == 0:
            power = 0
            while number % d == 0:
                number //= d
                power += 1
            factors.append((d, power))
        d += 1 if d == 2 else 2
    if number > 1:
        factors.append((number, 1))
    return "".join(f"({p})" if k == 1 else f"({p}**{k})" for p, k in factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    return k if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(c) for c in str(number)]
    L = len(digits)
    if L % 2 == 1:
        left = digits[: L // 2]
        right = digits[L // 2 + 1:]
    else:
        left = digits[: L // 2 - 1]
        right = digits[L // 2 + 1:]
    return sum(left) == sum(right)