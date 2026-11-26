import time
import csv
import evaluation.rag_baseline as baseline
import rag as hybrid


# Document filenames
INTEGRITY = "Academic Integrity Policy_SPJ.pdf"
REGULATIONS = "Academic Regulations_SPJ.pdf"
APPEALS = "Complaints_and_Appeals_SPJ.pdf"
SUPPORT = "Student Support Information_SPJ.pdf"
CONDUCT = "Student Code of Conduct_SPJ.pdf"


def make_case(question, category, source, pages):
    return {
        "question": question,
        "category": category,
        "source": source,
        "pages": pages
    }


# Evaluation dataset
# Reference pages should be checked against the original PDFs.
test_cases = [

    # Academic integrity
    make_case(
        "What are the penalties for plagiarism?",
        "Direct", INTEGRITY, [5, 6]
    ),
    make_case(
        "What happens if I copy someone's assignment?",
        "Paraphrased", INTEGRITY, [5, 6]
    ),
    make_case(
        "plagarism punishmnt?",
        "Misspelled", INTEGRITY, [5, 6]
    ),
    make_case(
        "Can I use generative AI for assessments?",
        "Direct", INTEGRITY, [2, 3]
    ),
    make_case(
        "Can I appeal an academic misconduct decision?",
        "Direct", INTEGRITY, [6]
    ),

    # Attendance
    make_case(
        "What are the attendance requirements for BDS students?",
        "Direct", REGULATIONS, [45, 46]
    ),
    make_case(
        "What happens if I miss five classes in one BDS subject without approval?",
        "Scenario", REGULATIONS, [46]
    ),
    make_case(
        "Can I get an excused absence if I am sick?",
        "Paraphrased", REGULATIONS, [45]
    ),
    make_case(
        "Can BDS students take a retest after an attendance downgrade?",
        "Direct", REGULATIONS, [46]
    ),
    make_case(
        "How long do I have to correct an incorrect BDS attendance record?",
        "Direct", REGULATIONS, [46]
    ),

    # Complaints and appeals
    make_case(
        "How do I submit a formal complaint?",
        "Direct", APPEALS, [3, 5]
    ),
    make_case(
        "What is the deadline for lodging a formal grievance?",
        "Direct", APPEALS, [5]
    ),
    make_case(
        "How long do I have to appeal a Stage One academic grievance decision?",
        "Direct", APPEALS, [5]
    ),
    make_case(
        "Do I have to pay to lodge an internal appeal?",
        "Paraphrased", APPEALS, [4, 5]
    ),
    make_case(
        "Can I submit an anonymous complaint?",
        "Direct", APPEALS, [3]
    ),

    # Student support
    make_case(
        "How can I book an appointment with a student counsellor?",
        "Direct", SUPPORT, [6]
    ),
    make_case(
        "Does the university provide English language support?",
        "Direct", SUPPORT, [4]
    ),
    make_case(
        "Can I meet my professor outside class for additional help?",
        "Paraphrased", SUPPORT, [4]
    ),
    make_case(
        "How does the university identify students at risk?",
        "Direct", SUPPORT, [4]
    ),
    make_case(
        "What online library resources are available?",
        "Direct", SUPPORT, [3]
    ),

    # Student conduct
    make_case(
        "Can I use my phone during lectures?",
        "Direct", CONDUCT, [2]
    ),
    make_case(
        "Am I allowed to drink alcohol on campus?",
        "Direct", CONDUCT, [2]
    ),
    make_case(
        "Do students have to carry their student ID on campus?",
        "Direct", CONDUCT, [2]
    ),
    make_case(
        "What is the university's policy on cyberbullying?",
        "Paraphrased", CONDUCT, [2]
    ),
    make_case(
        "Who should I report a breach of the student code of conduct to?",
        "Direct", CONDUCT, [4]
    )
]


# Validate reference documents and pages
def validate_dataset():

    available = {
        (chunk["source"], chunk["page"])
        for chunk in hybrid.chunks
    }

    errors = []

    for case in test_cases:

        if not any(
            (case["source"], page) in available
            for page in case["pages"]
        ):
            errors.append(case)

    if errors:

        print("\nInvalid reference documents or pages:")

        for case in errors:
            print(
                f"{case['source']} | "
                f"Pages: {case['pages']}"
            )

        print("\nAvailable document filenames:")

        for source in sorted({
            chunk["source"] for chunk in hybrid.chunks
        }):
            print(source)

        raise ValueError(
            "Correct the evaluation references before continuing."
        )

    print("Dataset reference validation passed.")


# Evaluate retrieval performance
def evaluate_retrieval(module, name, top_k):

    records = []

    # Warm up the embedding model
    module.retrieve(
        "University policies",
        top_k=top_k
    )

    for i, case in enumerate(test_cases, start=1):

        start = time.perf_counter()

        results = module.retrieve(
            case["question"],
            top_k=top_k
        )

        elapsed = time.perf_counter() - start

        hit = 0
        reciprocal_rank = 0
        first_relevant_rank = 0

        for rank, result in enumerate(results, start=1):

            if (
                result["source"] == case["source"]
                and result["page"] in case["pages"]
            ):
                hit = 1
                first_relevant_rank = rank
                reciprocal_rank = 1 / rank
                break

        records.append({
            "architecture": name,
            "top_k": top_k,
            "question": case["question"],
            "category": case["category"],
            "expected_source": case["source"],
            "expected_pages": str(case["pages"]),
            "hit": hit,
            "first_relevant_rank": first_relevant_rank,
            "reciprocal_rank": reciprocal_rank,
            "retrieval_time": round(elapsed, 4)
        })

        print(
            f"[{i}/{len(test_cases)}] "
            f"{name} | "
            f"Hit: {hit} | "
            f"Rank: {first_relevant_rank} | "
            f"Time: {elapsed:.3f}s | "
            f"{case['question']}"
        )

    return records


# Calculate evaluation metrics
def summarize(records, name, top_k):

    n = len(records)

    successful = sum(r["hit"] for r in records)

    hit_rate = successful / n

    mrr = sum(
        r["reciprocal_rank"] for r in records
    ) / n

    avg_time = sum(
        r["retrieval_time"] for r in records
    ) / n

    print(f"\n{name} Results (K={top_k})")
    print("-" * 40)

    print(f"Successful retrievals: {successful}/{n}")
    print(f"Hit Rate@{top_k}: {hit_rate:.2%}")
    print(f"MRR@{top_k}: {mrr:.4f}")
    print(f"Average retrieval time: {avg_time:.4f}s")

    return {
        "architecture": name,
        "top_k": top_k,
        "total_questions": n,
        "successful_retrievals": successful,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "average_retrieval_time": round(avg_time, 4)
    }


# Save results to CSV
def save_csv(records, filename):

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved: {filename}")


# Main evaluation
if __name__ == "__main__":

    validate_dataset()

    summary_results = []

    for top_k in [10, 20]:

        print("\n" + "=" * 60)
        print(f"EVALUATING TOP_K = {top_k}")
        print("=" * 60)

        baseline_results = evaluate_retrieval(
            baseline,
            "Baseline",
            top_k
        )

        hybrid_results = evaluate_retrieval(
            hybrid,
            "Hybrid",
            top_k
        )

        summary_results.append(
            summarize(
                baseline_results,
                "Baseline",
                top_k
            )
        )

        summary_results.append(
            summarize(
                hybrid_results,
                "Hybrid",
                top_k
            )
        )

        save_csv(
            baseline_results + hybrid_results,
            f"evaluation_results_k{top_k}.csv"
        )

    save_csv(
        summary_results,
        "evaluation_summary.csv"
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)