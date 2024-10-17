# Operations Research & Optimization
## Simplex Method & Simplex Table
Contains a solver to linear programming problem using simplex method while showing the simplex table. Also it can tell the nature of the solution. This is my homework of Operations Research & Optimization, so it may not be updated. 
### How to use:
1. Transform the LP problem to standard form:

  Maximize c^T x

  Subject to {A x = b, x >= 0

  Make sure that an identity matrix must contain in A！
  
2. Now we have A, b, c.
3. import shypy
4. Let Solution = maximize(c, A, b)
5. Solution().show()
Note that bugs may exsist! 
### Existing bugs: 
1. In some cases, especially when there're multiple solutions, the number of basic variables may be more than $m$.

## Fibonacci Search & BIsection Search
