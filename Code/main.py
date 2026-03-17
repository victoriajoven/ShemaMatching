# main.py

import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

from Send_LLM import send_prompts
from models import Parameters, Prompt

# Assumimos que ya tienes importado:
# Parameters, Prompt, Answer
# send_prompts, build_fewshot_prompt, process_prompt_list, etc.


# -------------------------------------
# 2 BASES DE DATOS DE EJEMPLO
# -------------------------------------

db1 = {
    "users": ["id", "email", "name"],
    "orders": ["order_id", "date", "amount"]
}

db2 = {
    "customers": ["customer_id", "email_address", "full_name"],
    "purchases": ["purchase_id", "timestamp", "total"]
}

# -------------------------------------
# FEW-SHOT EXAMPLES
# -------------------------------------

fewshot_examples = [
    {
        "input": "TableA: users, ColumnA: email | TableB: customers, ColumnB: email_address",
        "output": '{"match": true}'
    },
    {
        "input": "TableA: users, ColumnA: id | TableB: orders, ColumnB: order_date",
        "output": '{"match": false}'
    }
]

# -------------------------------------
# GENERADOR DE PROMPTS PARA TODAS LAS COMBINACIONES
# -------------------------------------

def build_schema_prompts(db1, db2):

    prompts = []
    
    for tableA, colsA in db1.items():
        for colA in colsA:

            for tableB, colsB in db2.items():
                for colB in colsB:

                    input_text = (
                        f"TableA: {tableA}, ColumnA: {colA}\n"
                        f"TableB: {tableB}, ColumnB: {colB}\n"
                    )

                    prompt = Prompt(
                        input_text=input_text,
                        fewshot_examples=fewshot_examples,
                        attributes={
                            "tableA": tableA,
                            "columnA": colA,
                            "tableB": tableB,
                            "columnB": colB
                        },
                        meta={"path": "./results"},
                        prompt={"n": 3}
                    )

                    prompts.append(prompt)

    return prompts


# -------------------------------------
# MAIN
# -------------------------------------

def main():

    # Cargar modelo Qwen local
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    parameters = Parameters(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=3
    )

    # Construir todos los prompts para DB1×DB2
    prompts = build_schema_prompts(db1, db2)

    # Ejecutar pipeline
    answers = send_prompts(parameters, prompts)

    # Mostrar resultados
    for ans in answers:
        print(f"{ans.attributes} → {ans.answer} → VALID={ans.valid}")


# -------------------------------------
# ENTRYPOINT
# -------------------------------------

if __name__ == "__main__":
    main()