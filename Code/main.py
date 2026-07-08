# main.py

from transformers import AutoModelForCausalLM, AutoTokenizer

from Send_LLM import send_prompts
from models import Parameters, Prompt

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


# -------------------------------------
# 2 BASES DE DATOS DE EJEMPLO
# -------------------------------------

db1 = {
    "users": ["id", "email", "name"],
    "orders": ["order_id", "date", "amount"],
    "products": ["product_id", "name", "price"],
    "payments": ["payment_id", "order_id", "payment_method", "status"]
}

db2 = {
    "customers": ["customer_id", "email_address", "full_name"],
    "purchases": ["purchase_id", "timestamp", "total"],
    "items": ["item_id", "item_name", "unit_price"],
    "transactions": ["transaction_id", "purchase_id", "method", "state"]
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


def build_schema_prompts(
    db1,
    db2,
    table_knn_index,
    embedding_model,
    table_names,
    k_tables=3
):

    prompts = []

    for tableA, colsA in db1.items():

        # Embedding de la tabla origen
        tableA_vector = embedding_model.encode([tableA])

        # Obtener tablas similares de db2
        distances, indices = table_knn_index.kneighbors(
            tableA_vector,
            n_neighbors=min(k_tables, len(table_names))
        )

        nearest_tables = [
            table_names[idx]
            for idx in indices[0]
        ]

        print(f"\nTabla origen: {tableA}")
        print(f"Tablas vecinas: {nearest_tables}")

        for colA in colsA:

            for tableB in nearest_tables:

                colsB = db2[tableB]

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

    # Modelo LLM
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    parameters = Parameters(
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=3
    )

    # ---------------------------------
    # Embeddings para tablas
    # ---------------------------------

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    table_names = list(db2.keys())

    table_vectors = embedding_model.encode(table_names)

    table_knn_index = NearestNeighbors(
        metric="cosine"
    )

    table_knn_index.fit(table_vectors)

    # ---------------------------------
    # Generar prompts
    # ---------------------------------

    prompts = build_schema_prompts(
        db1,
        db2,
        table_knn_index,
        embedding_model,
        table_names,
        k_tables=2  # db2 tiene solo 2 tablas
    )

    print(f"\nPrompts generados: {len(prompts)}")

    # ---------------------------------
    # Ejecutar modelo
    # ---------------------------------

    answers = send_prompts(
        parameters,
        prompts
    )

    # ---------------------------------
    # Mostrar resultados
    # ---------------------------------

    for ans in answers:
        print(
            f"{ans.attributes} -> "
            f"{ans.answer} -> "
            f"VALID={ans.valid}"
        )


# -------------------------------------
# ENTRYPOINT
# -------------------------------------

if __name__ == "__main__":
    main()