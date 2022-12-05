import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np



def knapsack_problem(items, capacity):
    """
    :param items: list of tuples (value, weight)
    :param capacity: int
    :return: list of tuples (value, weight)
    """

    # initialize a matrix of zeros
    K = [[0 for x in range(capacity + 1)] for x in range(len(items) + 1)]

    # build table K[][] in bottom up manner
    for i in range(len(items) + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif items[i - 1][1] <= w:
                K[i][w] = max(items[i - 1][0] + K[i - 1][w - items[i - 1][1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # stores the result of Knapsack
    res = K[len(items)][capacity]
    print(res)

    w = capacity
    result = []
    for i in range(len(items), 0, -1):
        if res <= 0:
            break
        # either the result comes from the top
        # (K[i-1][w]) or from (val[i-1] + K[i-1]
        # [w-wt[i-1]]) as in Knapsack table. If
        # it comes from the latter one/ it means
        # the item is included.
        if res == K[i - 1][w]:
            continue
        else:

            # This item is included.
            result.append(items[i - 1])
            res = res - items[i - 1][0]
            w = w - items[i - 1][1]

    return result

print(knapsack_problem([(1, 2), (4, 3), (5, 6), (6, 7)], 10))