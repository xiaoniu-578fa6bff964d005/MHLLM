#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
探索LLM上的MH采样
    问题
        已知一个分布
            \latex $p(x)$
        有一个能量函数
            \latex $E(x)$
        要从新分布中采样
            \latex $\frac{1}{Z}p(x)e^{E(x)}$
    用途
        保证输出具有某种性质
    方法
        MH采样
        定义迁移分布
            \latex $g(x'|x)$
    尝试实现
        例子1
            \latex $p(x)$
                生成一首诗
            \latex $E(x)$
                每一行开头必须是ABC
                首尾两行必须词序相反
                    ref
                        page 6=79
    扩展
        unbiased MCMC with maximum coupling
"""

from utils import *

#  p_prompt = """Instruction:
#  Write a short poem.
#
#  Poem:
#  """

#  p_prompt = """Instruction:
#  Write a short poem so the last sentence and the first sentence have the same words, but in reverse order.
#
#  Naive Example:
#  Morning is beautiful
#  The sun rises
#  The sun sets
#  Beautiful is morning
#
#  Poem:
#  """

p_prompt = """Instruction:
Write a short poem so the last sentence and the first sentence have the same words, but in reverse order.

Poem:
"""


#  g_prompt = """Instruction:
#  Write a short poem where the last sentence and the first sentence have the same words, but in reverse order.
#
#  Naive Example:
#  Morning is beautiful
#  The sun rises
#  The sun sets
#  Beautiful is morning
#
#  Poem:
#  """

g_prompt = """Reference Poem:
{old_x}

Instruction:
Write a short poem so the last sentence and the first sentence have the same words, but in reverse order.

Naive Example:
A is B in C
...
C in B is A

Amended Poem:
"""

#  g_prompt = """Reference Poem:
#  {old_x}
#
#  Instruction:
#  Write a short poem so the last sentence and the first sentence have the same words, but in reverse order.
#
#  Amended Poem:
#  """


def get_E(x):
    lines = x.lower().split("\n")
    lines = [line for line in lines if line != ""]
    line1 = lines[0]
    linen = lines[-1]
    words1 = line1.split(" ")
    wordn = linen.split(" ")
    rlinen = " ".join(wordn[::-1])

    from Levenshtein import distance

    return 200 * distance(line1, rlinen)


from test import *


def run_MH():
    import numpy as np

    old_x, log_p_old_x = sample(p_prompt)
    E_old_x = get_E(old_x)
    print("[init] ", old_x)
    while True:
        x, log_g_x = sample(g_prompt, {"old_x": old_x})
        log_p_x = get_log_p(x, p_prompt)
        E_x = get_E(x)
        log_g_old_x = get_log_p(old_x, g_prompt, {"old_x": x})
        log_pre_alpha = -E_x + log_p_x + log_g_old_x - log_p_old_x + E_old_x - log_g_x
        print(log_pre_alpha)
        print(log_p_x, log_g_old_x, log_p_old_x, log_g_x)
        print(E_x, E_old_x)
        alpha = min(1, np.exp(min(log_pre_alpha, 10)))
        if np.random.rand() < alpha:
            old_x = x
            log_p_old_x = log_p_x
            E_old_x = E_x
            print("[accepted!!!!!!!!] ", x)
        else:
            print("[rejected] ", x)


if __name__ == "__main__":
    #  test1()
    #  test2()
    #  test3()
    #  test4()
    #  test5()
    run_MH()
