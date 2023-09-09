#!/usr/bin/env python
# -*- coding: utf-8 -*-


import openai
import json

openai.api_key = json.load(open("config.json"))["openai_api_key"]


def _real_prompt(prompt, prompt_dict):
    real_prompt = prompt
    for k, v in prompt_dict.items():
        if "{" + k + "}" in real_prompt:
            real_prompt = real_prompt.format(**{k: v})
    return real_prompt


def sample(prompt, prompt_dict={}, completion_kwargs={}):
    real_prompt = _real_prompt(prompt, prompt_dict)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=real_prompt,
        temperature=1.0,
        max_tokens=2048,
        logprobs=5,
        **completion_kwargs,
    )
    x = response.choices[0].text
    logprobs = response.choices[0].logprobs
    out_pos = next(
        (i for i, x in enumerate(logprobs["text_offset"]) if x >= len(real_prompt)),
        None,
    )
    log_p = sum(logprobs["token_logprobs"][out_pos:])
    return x, log_p


def get_log_p(x, prompt, prompt_dict={}, completion_kwargs={}):
    real_prompt = _real_prompt(prompt, prompt_dict)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=real_prompt + x,
        temperature=1.0,
        max_tokens=0,
        logprobs=5,
        echo=True,
        **completion_kwargs,
    )
    logprobs = response.choices[0].logprobs
    out_pos = next(
        (i for i, x in enumerate(logprobs["text_offset"]) if x >= len(real_prompt)),
        None,
    )
    log_p = sum(logprobs["token_logprobs"][out_pos:])
    return log_p
