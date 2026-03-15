import asyncio
import copy
import json
from typing import Any, Dict, List
import tenacity
import torch

from .config import config
from .errors import NotDoneException
from .models import Answer, Parameters, Prompt
from .storage import store_answer, store_chatcompletion, get_answers_by_prompt



#  ASK LOCAL QWEN MODEL con los parametros y el prompt

async def ask_local_qwen(prompt_text: str, parameters: Parameters) -> List[str]:
    """
    Genera múltiples secuencias con Qwen local usando HuggingFace.
    Devuelve una lista de strings (num_return_sequences).
    """

    model = parameters.model
    tokenizer = parameters.tokenizer

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    cut = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=parameters.max_new_tokens,
        temperature=parameters.temperature,
        do_sample=parameters.do_sample,
        top_p=parameters.top_p,
        num_return_sequences=parameters.num_return_sequences,  # self-consistency
    )

    generations = []
    for i in range(outputs.shape[0]):
        gen_ids = outputs[i, cut:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generations.append(text)

    return generations



#  FEW-SHOT PROMPT BUILDER (ajustando a formato) 


def build_fewshot_prompt(prompt: Prompt) -> str:
    """
    Crea el texto final del prompt con tus ejemplos.
    Puedes adaptar esta función según tu estructura.
    """

    header = "You are a schema-matching model. Output ONLY a JSON object {\"match\": true/false}.\n\n"
    examples = ""

    for ex in prompt.fewshot_examples:
        examples += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"

    return header + examples + "Now solve:\n" + prompt.input_text



#  VALIDACIÓN JSON con True o False

def extract_json(answer: Answer) -> Dict[str, Any]:
    """Extrae JSON del texto de la respuesta."""
    start = answer.answer.rindex("{")
    end = answer.answer.index("}", start)
    raw = answer.answer[start:end+1].replace("'", '"')
    return json.loads(raw)


def is_valid_answer(answer: Answer) -> bool:
    """Verifica si la respuesta es JSON válido."""
    try:
        extract_json(answer)
        return True
    except:
        return False



#  PROCESS + STORE A SINGLE PROMPT con llamada al modelo local
async def process_and_store_prompt(
    parameters: Parameters,
    prompt: Prompt,
    semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
) -> List[Answer]:

    valid_answers = get_answers_by_prompt(prompt, filter_valid=True)
    missing = prompt.prompt["n"] - len(valid_answers)

    # construir few-shot
    prompt_text = build_fewshot_prompt(prompt)

    if missing > 0:
        try:
            for attempt in tenacity.Retrying(
                stop=tenacity.stop_after_attempt(5),               # máximo reintentos
                retry=tenacity.retry_if_exception_type(NotDoneException),
            ):
                with attempt:
                    async with semaphore:

                        # LLAMADA AL MODELO QWEN LOCAL ---------
                        
                        generations = await ask_local_qwen(
                            prompt_text,
                            parameters
                        )

                    # almacenar (opcional: similar al "store_chatcompletion")
                    store_chatcompletion(
                        {"prompt": prompt_text, "generations": generations},
                        prompt.meta["path"]
                    )

                    # procesar cada generación
                    for idx, text in enumerate(generations):

                        answer = Answer(
                            prompt.attributes,
                            index=idx,
                            answer=text,
                        )

                        if is_valid_answer(answer):
                            answer.valid = True
                            valid_answers.append(answer)
                            missing -= 1

                        # almacenar cada respuesta individual
                        store_answer(answer, prompt.meta["path"], answer_id=str(idx))

                    if missing > 0:
                        raise NotDoneException("Not enough valid answers provided.")

        except tenacity.RetryError:
            pass

    return valid_answers[: config["OPENAI_N"]]


#  PROCESS LIST OF PROMPTS (PARALLEL)

async def process_prompt_list(parameters: Parameters, prompts: List[Prompt]) -> List[Answer]:
    """Ejecuta varios prompts en paralelo con semáforo."""
    semaphore = asyncio.Semaphore(config["PARALLEL_OPENAI_REQUESTS"])

    tasks = []
    async with asyncio.TaskGroup() as tg:
        for prompt in prompts:
            tasks.append(
                tg.create_task(process_and_store_prompt(parameters, prompt, semaphore))
            )

    return [answer for task in tasks for answer in task.result()]



#  EXTERNAL ENTRY POINT

def send_prompts(parameters: Parameters, prompts: List[Prompt]) -> List[Answer]:
    return asyncio.run(process_prompt_list(parameters, prompts))