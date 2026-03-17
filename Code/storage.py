# storage.py
import os
import json

def store_chatcompletion(data, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "chat_log.jsonl"), "a", encoding="utf8") as f:
        f.write(json.dumps(data) + "\n")


def store_answer(answer, path, answer_id):
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f"answer_{answer_id}.json")
    with open(fname, "w", encoding="utf8") as f:
        json.dump({
            "attributes": answer.attributes,
            "index": answer.index,
            "answer": answer.answer,
            "valid": answer.valid
        }, f, indent=2)


def get_answers_by_prompt(prompt, filter_valid=False):
    # inicialmente no hay respuestas guardadas
    return []