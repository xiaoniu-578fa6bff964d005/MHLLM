#!/usr/bin/env python
# -*- coding: utf-8 -*-

from main import *


def test1():
    x, log_p = sample_p()
    print(x)
    print(log_p)


def test2():
    """
    verify logprobs is irrelevant to temperature
    """
    prefix = "Hello world"
    suffix = "How are you?"

    def get_logprobs(temperature):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prefix + suffix,
            temperature=temperature,
            logprobs=5,
            max_tokens=0,
            echo=True,
        )
        logprobs = response.choices[0].logprobs
        return logprobs

    logprobs1 = get_logprobs(0.0)
    logprobs2 = get_logprobs(1.0)
    logprobs3 = get_logprobs(2.0)
    import numpy as np

    lp1 = np.array(logprobs1["token_logprobs"][1:])
    lp2 = np.array(logprobs2["token_logprobs"][1:])
    lp3 = np.array(logprobs3["token_logprobs"][1:])
    #  print(lp1, lp2, lp3)
    #  print(lp1 - lp2, lp1 - lp3)
    #  assert np.allclose(lp1, lp2, atol=1e-2)
    #  assert np.allclose(lp1, lp3, atol=1e-2)
    assert np.allclose(lp1, lp2, atol=5e-2)
    assert np.allclose(lp1, lp3, atol=5e-2)


def test3():
    """
    verify logprobs is max at 0 temperature
    """
    prompt = "Generate a short poem.\n\nPoem:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        logprobs=5,
        max_tokens=100,
        echo=True,
    )
    logprobs = response.choices[0].logprobs
    out_pos = next(
        (i for i, x in enumerate(logprobs["text_offset"]) if x >= len(prompt)), None
    )

    def top_key(d):
        # d is like {"t1":0.1, "t2":0.2}
        return max(d, key=d.get)

    max_tokens = [top_key(d) for d in logprobs["top_logprobs"][out_pos:]]
    real_tokens = logprobs["tokens"][out_pos:]
    assert max_tokens == real_tokens


def test4():
    x, log_p = sample_p()
    log_p2 = get_p(x)

    import numpy as np

    assert np.allclose(log_p, log_p2, rtol=1e-2)

    old_x = "Hello world"
    x, log_p = sample_g(old_x)
    log_p2 = get_g(old_x, x)
    assert np.allclose(log_p, log_p2, rtol=1e-2)


def test5():
    x = """Hellow world
Hi
World hello"""
    print(get_E(x))
    x = """Hello world
Hi
World hello"""
    print(get_E(x))
    x = """Levenshtein has a some overlap with difflib (SequenceMatcher). It supports only strings, not arbitrary sequence types, but on the other hand it’s much faster.

It supports both normal and Unicode strings, but can’t mix them, all arguments to a function (method) have to be of the same type (or its subclasses)."""
    print(get_E(x))
