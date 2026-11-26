import rag
import pandas as pd

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

valid_questions = [
    "What is the penalty for plagiarism?",
    "plagarism punishmnt?",
    "How can I meet my professor outside class?",
    "Is alcohol allowed on campus?",
    "What are the attendance requirements for BDS students?",
    "How do I submit a formal complaint?",
    "How can I book a student counsellor?",
    "Can I use my phone during lectures?",
    "Do I need to carry my student ID?",
    "What English language support is available?"
]

invalid_questions = [
    "What is the capital of France?",
    "How do I cook pasta?",
    "What is the current price of Bitcoin?",
    "How do I sort a list in Python?",
    "Who won the 2018 FIFA World Cup?"
]

# Retrieve once per question
records = []

for label, questions in [
    ("valid", valid_questions),
    ("invalid", invalid_questions)
]:
    for question in questions:
        results = rag.retrieve(question, top_k=20)

        records.append({
            "question": question,
            "label": label,
            "scores": [r["score"] for r in results]
        })

# Evaluate different thresholds
summary = []

for threshold in thresholds:

    accepted_valid = 0
    rejected_invalid = 0
    total_chunks = 0

    for record in records:

        retained = [
            s for s in record["scores"]
            if s >= threshold
        ]

        total_chunks += len(retained)

        if record["label"] == "valid" and retained:
            accepted_valid += 1

        if record["label"] == "invalid" and not retained:
            rejected_invalid += 1

    summary.append({
        "threshold": threshold,
        "valid_acceptance_%": (
            accepted_valid / len(valid_questions) * 100
        ),
        "invalid_rejection_%": (
            rejected_invalid / len(invalid_questions) * 100
        ),
        "avg_chunks": round(
            total_chunks / len(records), 2
        )
    })

df = pd.DataFrame(summary)

print("\nThreshold Evaluation:\n")
print(df.to_string(index=False))

df.to_csv("evaluation/results/threshold_evaluation.csv", index=False)

print("\nIndividual question maximum scores:\n")

for record in records:
    print(
        f"{record['label']:7} | "
        f"{max(record['scores'], default=0):.3f} | "
        f"{record['question']}"
    )