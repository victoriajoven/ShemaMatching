import asyncio
import json
from typing import Any, Dict, List

from config import config
from models import Answer, Parameters, Prompt
from storage import (
    store_answer,
    store_chatcompletion,
    get_answers_by_prompt,
)


# ASK LOCAL QWEN MODEL

async def ask_local_qwen(
    prompt_text: str,
    parameters: Parameters
) -> List[str]:

    model = parameters.model
    tokenizer = parameters.tokenizer

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    ).to(model.device)

    cut = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=parameters.max_new_tokens,
        temperature=parameters.temperature,
        do_sample=parameters.do_sample,
        top_p=parameters.top_p,
        num_return_sequences=1,   # UNA SOLA RESPUESTA
    )

    generations = []

    for i in range(outputs.shape[0]):
        gen_ids = outputs[i, cut:]
        text = tokenizer.decode(
            gen_ids,
            skip_special_tokens=True
        )
        generations.append(text)

    return generations


# FEW-SHOT PROMPT BUILDER

def build_fewshot_prompt(prompt: Prompt) -> str:

    header = (
        'You are a schema-matching model. '
        'Output ONLY a JSON object {"match": true/false}.\n\n'
    )

    examples = ""

    for ex in prompt.fewshot_examples:
        examples += (
            f"Input: {ex['input']}\n"
            f"Output: {ex['output']}\n\n"
        )

    return header + examples + "Now solve:\n" + prompt.input_text


# VALIDACIÓN JSON

def extract_json(answer: Answer) -> Dict[str, Any]:

    start = answer.answer.rindex("{")
    end = answer.answer.index("}", start)

    raw = answer.answer[start:end + 1].replace("'", '"')

    return json.loads(raw)


def is_valid_answer(answer: Answer) -> bool:

    try:
        extract_json(answer)
        return True
    except Exception:
        return False


# PROCESS SINGLE PROMPT

async def process_and_store_prompt(
    parameters: Parameters,
    prompt: Prompt,
) -> List[Answer]:

    valid_answers = get_answers_by_prompt(
        prompt,
        filter_valid=True
    )

    missing = prompt.prompt["n"] - len(valid_answers)

    prompt_text = build_fewshot_prompt(prompt)

    if missing > 0:

        generations = await ask_local_qwen(
            prompt_text,
            parameters
        )

        store_chatcompletion(
            {
                "prompt": prompt_text,
                "generations": generations
            },
            prompt.meta["path"]
        )

        for idx, text in enumerate(generations):

            answer = Answer(
                prompt.attributes,
                index=idx,
                answer=text,
            )

            if is_valid_answer(answer):
                answer.valid = True
                valid_answers.append(answer)

            store_answer(
                answer,
                prompt.meta["path"],
                answer_id=str(idx)
            )

    return valid_answers[:1]


# PROCESS PROMPTS SECUENCIALMENTE

async def process_prompt_list(
    parameters: Parameters,
    prompts: List[Prompt]
) -> List[Answer]:

    answers = []

    for i, prompt in enumerate(prompts):

        print(f"Procesando prompt {i+1}/{len(prompts)}")

        result = await process_and_store_prompt(
            parameters,
            prompt
        )

        answers.extend(result)

    return answers


# ENTRY POINT

def send_prompts(
    parameters: Parameters,
    prompts: List[Prompt]
) -> List[Answer]:

    return asyncio.run(
        process_prompt_list(
            parameters,
            prompts
        )
    )
    