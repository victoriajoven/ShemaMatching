# models.py
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Parameters:
    model: Any
    tokenizer: Any
    max_new_tokens: int = 50
    temperature: float = 0.2
    do_sample: bool = True
    top_p: float = 0.9
    num_return_sequences: int = 3


@dataclass
class Prompt:
    input_text: str
    fewshot_examples: List[Dict]
    attributes: Dict
    prompt: Dict
    meta: Dict


@dataclass
class Answer:
    attributes: Dict
    index: int
    answer: str
    valid: bool = False