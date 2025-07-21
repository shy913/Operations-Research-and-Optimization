# Operations Research & Optimization

By shy

This is a part of my homework of Operations Research & Optimization.

## Simplex Method & Simplex Table
Contains a step-by-step solver to linear programming problem using simplex method while showing the simplex table. Also it can tell the nature of the solution. 
### How to use:
1. Transform the LP problem to standard form:



$$\text{max} \ c^T x$$
$$
\text{s.t. } 
\begin{cases}
A\textbf{x} = \textbf{b} \\
\textbf{x} \geq 0
\end{cases}
$$

  Make sure that an identity matrix is contained in A.

2. execute:
```python
import shypy
c = ...
A = ...
b = ...
Solution = maximize(c, A, b)
Solution().show()
```

## Fibonacci Search & Bisection Search
### How to use:
```python
fib_search(f, a, b, n):
    param f: The function to be evaluated.
    param a: Left node of the interval
    param b: Right node of the interval
    param n: Total number of evaluation
    return: Final Interval
```

example:


``` python
def function(x):
    return x ** 2 - 6 * x + 2

fib_search(function, 0, 10, 34)
```
