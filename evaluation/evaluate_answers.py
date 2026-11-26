import time
import csv
import evaluation.rag_baseline as baseline
import rag as hybrid

from evaluation.evaluate import test_cases


# Select three questions from each policy category
answerable_cases = [
    case
    for i, case in enumerate(test_cases)
    if i % 5 < 3
]

# Questions outside the knowledge base
out_of_scope = [
    "What is the capital of France?",
    "How do I cook pasta?",
    "What is the current price of Bitcoin?",
    "Write a Python program to sort a list.",
    "Who won the 2018 FIFA World Cup?"
]


def evaluate_answers(module, name):

    records = []

    questions = [
        {
            "question": case["question"],
            "expected_behavior": "Answer"
        }
        for case in answerable_cases
    ]

    questions += [
        {
            "question": question,
            "expected_behavior": "Refuse"
        }
        for question in out_of_scope
    ]

    for i, case in enumerate(questions, start=1):

        question = case["question"]

        print(
            f"[{i}/{len(questions)}] "
            f"{name} | {question}"
        )

        start = time.perf_counter()

        try:

            result = module.generate_answer(question)

            answer = result["answer"]
            sources = result["sources"]

            elapsed = time.perf_counter() - start

            source_details = " | ".join(
                f"{s['source']} (Page {s['page']})"
                for s in sources
            )

            records.append({
                "architecture": name,
                "question": question,
                "expected_behavior": case["expected_behavior"],
                "generated_answer": answer,
                "source_count": len(sources),
                "sources": source_details,
                "response_time": round(elapsed, 4),
                "accuracy": "",
                "groundedness": "",
                "refusal_correct": "",
                "error": ""
            })

            print(f"Response time: {elapsed:.2f}s")

        except Exception as e:

            print(f"Error: {e}")

            records.append({
                "architecture": name,
                "question": question,
                "expected_behavior": case["expected_behavior"],
                "generated_answer": "",
                "source_count": 0,
                "sources": "",
                "response_time": "",
                "accuracy": "",
                "groundedness": "",
                "refusal_correct": "",
                "error": str(e)
            })

        # Reduce risk of API rate limiting
        time.sleep(3)

    return records


if __name__ == "__main__":

    print("\nBASELINE ANSWER EVALUATION\n")

    baseline_results = evaluate_answers(
        baseline,
        "Baseline"
    )

    print("\nHYBRID ANSWER EVALUATION\n")

    hybrid_results = evaluate_answers(
        hybrid,
        "Hybrid"
    )

    results = baseline_results + hybrid_results

    with open(
        "evaluation/results/answer_evaluation.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation complete.")
    print("Results saved to answer_evaluation.csv")