import numpy as np
import tabulate


def maximize(c, A, b):
    """
        使用单纯形法求解线性规划问题：
        Maximize c.*x
        Subject to A x = b, x >= 0

        input: standard form
        :param c: numpy array/list,目标函数系数矩阵，行向量
        :param A: numpy array/list,限制条件矩阵
        :param b: numpy array/list,右侧项b，**行向量**
        :return: class solution：
            status: str
            optimalValue: np.float64 ?不确定
            solution: list: np.float64 ?也不确定
            solutionType: str
        """
    A = np.array(A)*1.0
    b = np.array(b)*1.0
    c = np.array(c)*1.0

    class result:
        status = 'undone'
        optimalValue = None
        solution = []
        solutionType = 'unknown'

        def show(self):
            print('\nsol_info:\n', tabulate.tabulate(
                [['status', self.status],
                 ['type', self.solutionType],
                 ['optimal', self.optimalValue]]),
                  sep='')
            print('\nbasic solution(s):\n',
                  tabulate.tabulate([sol for sol in self.solution],
                                    headers=[f'x{i+1}' for i in range(len(A[0]))],
                                    tablefmt='fancy_grid'), sep='')

    def showTable():
        tab_s = []
        sigma_s = [['', '', 'σ_j->'] + sigma]

        for i in range(m):
            tab_s.append([cb[i], f"x{base_index[i] + 1}", b[i]] + list(A[i]) + [theta[i]])

        tbp = [[f'({iter_count + 1})', '', 'c_j->'] + list(c),
               # ['', '', 'isBasis->'] + list(f'{bool(i)}' for i in bases),
               ['C_B', 'X_B', 'b'] + list(f'x{i + 1}' for i in range(len(bases))) + ['θ']
               ] + tab_s + sigma_s
        print(f'{tabulate.tabulate(tbp, tablefmt='fancy_grid')}')
        # print(f'{tabulate.tabulate(tbp)}')

    m, n = A.shape

    # simplex method
    iter_count = 0
    sigma = [None] * n
    theta = [None] * m

    bases = [0] * n
    base_index = []
    # choose n vectors as init solution
    count = 0
    id_m = np.identity(m)
    for i in range(n):
        for j in range(m):
            # print(f'A的第i列:{A[:, i]}\n'
            #       f'id_m的j列:{id_m[:, j]}')
            if np.all(A[:, i] == id_m[:, j]):
                count += 1
                # print(f"找到初解！{count}/{m}  j={i}, {A[:, i]}")
                bases[i] = 1
    if count != m:
        exit('你这standard form保熟吗？')

    for i in range(len(bases)):
        if bases[i] == 1:
            base_index.append(i)
    cb = []
    for i in range(n):
        if bases[i] == 1:
            cb.append(c[i])
    exit_flg = 0

    while True:
        # print(f'\niteration {iter_count+1}')
        if iter_count == 10:
            result.status = 'Tooooo many iterations!'
            exit('循环太多次啦')
        result.status = 'maximizing'
        # showTable()
        # 开始算参数
        # 算sigma
        for i in range(n):
            sigma[i] = c[i]
            for j in range(m):
                sigma[i] -= cb[j] * A[j, i]
        # print(f'sigma = {sigma}')
        enter_index = np.nanargmax(sigma)
        main_elem_j = np.nanargmax(sigma)
        # print(f'sigma计算结束')

        # 算theta
        np.seterr(divide='ignore', invalid='ignore') # 关闭除0警报
        for i in range(m):
            theta[i] = b[i] / A[i, enter_index]
        for i in range(len(theta)):
            if theta[i] < 0:
                theta[i] = np.nan
        exit_index = np.nanargmin(theta)
        main_elem_i = np.nanargmin(theta)
        # print(f'theta计算结束')
        np.seterr(divide='warn', invalid='warn') # 打开除0警报

        sigma = np.round(sigma, 5).tolist()

        # 如果sigma都<=0：结束！
        if all(s <= 0 for s in sigma):
            visited = bases
            for i in range(len(bases)):
                if visited[i] == 0 and sigma[i] == 0: # 多解了！
                    result.status = 'finding multiple solutions'
                    result.solutionType = 'bounded, multiple, optimal'
                    final_solution = np.zeros(n)
                    for i in range(m):
                        final_solution[base_index[i]] = b[i]
                    result.solution += [final_solution]
                    result.optimalValue = np.dot(c, final_solution)
                    # 现在开始找等高点
                    for i in range(len(visited)):
                        if visited[i] == 0 and sigma[i] == 0:
                            # print('可以过去',i,visited[i],visited)
                            visited[i] = 1
                            # 去那里看看
                            # 展示当前表格，我要开始变了
                            showTable()
                            enter_index = i
                            main_elem_j = i
                            # 重新计算theta
                            # 算theta
                            np.seterr(divide='ignore', invalid='ignore')  # 关闭除0警报
                            for i in range(m):
                                theta[i] = b[i] / A[i, enter_index]
                            for i in range(len(theta)):
                                if theta[i] < 0:
                                    theta[i] = np.nan

                            exit_index = np.nanargmin(theta)
                            main_elem_i = np.nanargmin(theta)
                            # print(f'theta计算结束')
                            np.seterr(divide='warn', invalid='warn')  # 打开除0警报
                            # print(f'主元在第{main_elem_i}行{main_elem_j}列,值为{A[main_elem_i,main_elem_j]}')
                            # 进进出出
                            bases[enter_index] = 1
                            bases[base_index[exit_index]] = 0
                            base_index[exit_index] = enter_index
                            for i in range(len(bases)):
                                if bases[i] == 1:
                                    base_index.append(i)
                            cb[exit_index] = c[enter_index]
                            # print(f'进出基变量更新结束')

                            # 初等行变换
                            b[main_elem_i] = b[main_elem_i] / A[main_elem_i, main_elem_j]
                            A[main_elem_i, :] = A[main_elem_i, :] / A[main_elem_i, main_elem_j]

                            for i in range(m):
                                if i != main_elem_i:
                                    b[i] = b[i] - b[main_elem_i] * A[i, main_elem_j]
                                    A[i, :] = A[i, :] - A[main_elem_i, :] * A[i, main_elem_j]

                            # print(f'初等行变换结束')
                            final_solution = np.zeros(n)
                            for i in range(m):
                                final_solution[base_index[i]] = b[i]
                            result.solution += [final_solution]
                            iter_count += 1


                    solutions = []
                    for arr in result.solution:
                        if not any(np.allclose(arr, unique, rtol=1e-05, atol=1e-08) for unique in solutions):
                            solutions.append(arr)
                    result.solution = solutions
                    result.status = 'Done'
                    return result

            # single obfs

            theta = [0] * m
            showTable()
            # print("找到最优解！")
            result.status = 'Done'
            result.solutionType = 'bounded, single, optimal'
            final_solution = np.zeros(n)
            for i in range(m):
                final_solution[base_index[i]] = b[i]
            result.solution = [final_solution]
            result.optimalValue = np.dot(c, final_solution)
            break
        # print(f'退出判断结束')

        # 如果unbounded，结束！
        if A[main_elem_i, main_elem_j] == 0:
            # print('最终解unbounded！')
            result.status = 'Done'
            result.solutionType = 'unbounded'
            result.solution = ''
            result.optimalValue = np.inf
            showTable()
            break

        # 画表格
        showTable()
        # print(f'表格绘制结束')

        # 找主元
        # print(f'主元在第{main_elem_i}行{main_elem_j}列,值为{A[main_elem_i,main_elem_j]}')

        # 进进出出
        bases[enter_index] = 1
        bases[base_index[exit_index]] = 0
        base_index[exit_index] = enter_index
        for i in range(len(bases)):
            if bases[i] == 1:
                base_index.append(i)
        cb[exit_index] = c[enter_index]
        # print(f'进出基变量更新结束')

        # 初等行变换
        b[main_elem_i] = b[main_elem_i] / A[main_elem_i, main_elem_j]
        A[main_elem_i, :] = A[main_elem_i, :] / A[main_elem_i, main_elem_j]

        for i in range(m):
            if i != main_elem_i:
                b[i] = b[i] - b[main_elem_i] * A[i, main_elem_j]
                A[i, :] = A[i, :] - A[main_elem_i, :] * A[i, main_elem_j]

        # print(f'初等行变换结束')
        iter_count += 1
    return result

def minimize(c,A,b):
    c = -np.array(c)
    Result = maximize(c,A,b)
    Result.optimalValue = -1 * Result.optimalValue
    return Result

if __name__ == '__main__':
    # c = np.array([3, 4, 0, 0])
    # A = np.array([[2, 1, 1, 0], [1, 3, 0, 1]])
    # b = np.array([40, 30]).T
    # result = maximize(c, A, b)
    # print(result.solution)

    c = [2, 4, 0, 0, 0]
    A = [[-1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [1, -1, 0, 0, 1]]
    b = [4, 10, 2]

    Solution = maximize(c, A, b)
    Solution().show()
