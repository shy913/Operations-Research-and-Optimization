epsilon = 1e-5

cache = [-1] * 100
cache[0] = 0
cache[1] = 1


def fib(n):
    global cache
    if n <= 1:
        return n
    elif cache[n] == -1:
        cache[n] = fib(n - 1) + fib(n - 2)
        return cache[n]
    return cache[n]


def fib_search(f, a, b, n):
    global epsilon
    '''
    :param f: The function to be evaluated.
    :param a: Left node of the interval
    :param b: Right node of the interval
    :param n: Total number of evaluation
    :return: Interval
    '''
    if n <= 1:
        print(f"[{a}, {b}]")
        return a, b
    elif n == 2:
        mid = (a + b) / 2
        if f(mid - epsilon) < f(mid + epsilon):
            print(f"len: {(b - a)/2}")
            print(f"[{a}, {mid}]")
            return a, mid
        elif f(mid - epsilon) > f(mid + epsilon):
            print(f"len: {(b - a)/2}")
            print(f"[{mid}, {b}]")
            return mid, b
        else:
            print(f"[{mid - epsilon}, {mid + epsilon}]")
            return mid - epsilon, mid + epsilon
    else:
        print(f"len: {b-a}")
        left_eval = (a + (b - a) * fib(n - 1) / fib(n + 1))
        right_eval = (a + (b - a) * fib(n) / fib(n + 1))
        print(f"[{a},{left_eval},{right_eval},{b}]")
        if f(left_eval) < f(right_eval):
            # print(f"left_eval < right_eval!\n-----------------\n")
            fib_search(f, a, right_eval, n - 1)
        elif f(left_eval) == f(right_eval):
            # print(f"right_eval = left_eval!\n-----------------\n")
            fib_search(f, left_eval, right_eval, n - 1)
        else:
            # print(f"right_eval > left_eval!\n-----------------\n")
            fib_search(f, left_eval, b, n - 1)


def binary_search(f, a, b, n):
    global epsilon
    mid = (a + b) / 2
    if n == 0 or n == 1:
        print(f"len: {b-a}")
        print(f"[{a}, {b}]")
        return a, b
    elif f(mid - epsilon) < f(mid + epsilon):
        print(f"[{a}, {mid}]")
        return binary_search(f, a, mid, n-2)
    elif f(mid - epsilon) > f(mid + epsilon):
        print(f"[{mid}, {b}]")
        return binary_search(f, mid, b, n-2)


if __name__ == "__main__":
    def func(x):
        return x ** 2 - 6 * x + 2
    print("fibonacci search:")
    fib_search(func, 0, 10, 8)
    print("\nbinary search:")
    binary_search(func, 0, 10, 12)
