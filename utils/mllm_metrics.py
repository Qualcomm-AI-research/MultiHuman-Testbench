# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
MLLM Visual Question Answering (VQA) Module

This module provides functionality to perform visual question answering using a 
Multimodal Large Language Model (MLLM) from Google's Gemini API. It constructs a 
prompt from a list of questions and an image, sends it to the model, and retrieves 
a digit-only response representing the answers.

Includes:
- Initialization of the Gemini client.
- A mapping for digit-based answers.
- A retry mechanism for handling temporary API failures.

Important:
Please add your Google API key in line 25. To generate this, navigate to 
https://aistudio.google.com/app/apikey and follow instructions to Get API key.
"""

import time

try:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key="") # Please put the API key here.
    MLLM_AWAKE = True
except:
    print("Can't load MLLM. VQA not available")
    MLLM_AWAKE = False


MAP = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
}


def mllm_vqa(questions, image):
    prompt = "Answer the following questions regarding the Image:"
    num_questions = len(questions)

    for i, que in enumerate(questions):
        prompt += f" Q{i+1}. {que}"

    prompt += f". Provide a {MAP[num_questions]}-digit answer. Each digit is the answer. Leave no spaces, no answer identifiers, no newline characters. No 'Q1'/'Q2' tags. Only digits"

    flag = False
    while not flag:
        # If Gemini is down, we re-try after 5 seconds. This is to bypass time-based limits.
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[image, prompt],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            flag = True
        except:
            time.sleep(5)
            print("Gemini down. Re-trying after 5 seconds ...")
    answers = response.text

    return answers
