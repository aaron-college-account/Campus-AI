# CampusAI

CampusAI is a retrieval-augmented generation (RAG) application that answers SP Jain students' questions about university policies and regulations. It retrieves relevant passages from official documents and uses a language model to generate answers with source references. The system is designed to refuse questions when evidence is insufficient.

## Dataset

The corpus consists of seven university PDFs (161 extracted pages, 526 chunks):

- Academic Integrity Policy
- Academic Regulations
- Assessment Policies
- Complaints and Appeals
- Student Code of Conduct
- Student Handbook
- Student Support Information

PyMuPDF extracts text page by page. Whitespace is normalized and text is divided into 1,000-character chunks with 200-character overlap. Chunks shorter than 100 characters are discarded. Each chunk retains its document filename and PDF page number.

## Architecture

```text
PDF documents -> Text extraction -> Chunking -> MiniLM embeddings -> FAISS
                                     |
                                     +-> BM25 keyword index (at runtime)

User question -> Query rewriting (if conversation history exists)
              -> FAISS semantic search + BM25 keyword search
              -> Reciprocal Rank Fusion -> Top 20 chunks
              -> Cosine similarity filter (>= 0.30)
              -> GPT-OSS-20B via Groq -> Answer with source references
```

### NLP techniques

- **Sentence embeddings:** `all-MiniLM-L6-v2` maps text into 384-dimensional normalized vectors. It provides contextual semantic representations without requiring local model training.
- **Semantic retrieval:** FAISS `IndexFlatIP` performs exact inner-product search. With normalized vectors, inner product equals cosine similarity.
- **Keyword retrieval:** BM25 ranks chunks using term frequency, inverse document frequency and document-length normalization. Text is tokenized using a lowercase word-matching regular expression.
- **Hybrid ranking:** Reciprocal Rank Fusion combines the top 50 candidates from each retrieval method using `RRF(d) = sum(1 / (60 + rank(d)))` across rankings containing the document chunk. Rank fusion avoids directly mixing incompatible score scales.
- **Relevance filtering:** The top 20 fused chunks are filtered using cosine similarity of at least 0.30.
- **Generation:** `openai/gpt-oss-20b` is accessed through Groq at temperature 0.1. The prompt instructs the model to use retrieved evidence, cite sources and refuse unsupported questions.
- **Conversation handling:** Up to six recent messages are used to rewrite follow-up questions as standalone search queries.
- **Interface:** Streamlit provides the interactive chat application.

The baseline uses FAISS semantic retrieval alone. The final hybrid system combines semantic and keyword retrieval. No model is fine-tuned or trained from scratch.

## Installation and usage

Requirements: Python, internet access for the initial embedding model download and Groq API requests, and a valid Groq API key. Run commands from the CampusAI root directory.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

Place the seven PDFs in `documents/`. Generate or rebuild the vector store:

```powershell
python ingest.py
```

If the existing vector store is supplied and matches the PDFs, rebuilding is optional.

Launch the application:

```powershell
streamlit run app.py
```

Alternatively, test the backend directly:

```powershell
python rag.py
```

Example questions:

- What are the penalties for plagiarism?
- What are the attendance requirements for BDS students?
- How do I submit a formal complaint?
- How can I book a student counsellor?

## Evaluation

### Retrieval comparison

A 25-question benchmark compares semantic and hybrid retrieval at K=10 and K=20. Hit rate measures whether at least one expected document-page reference appears among the retrieved chunks. Mean reciprocal rank (MRR) measures how early the first matching reference appears. These are page-level retrieval metrics, not factual answer accuracy.

| Method | K | Hit rate | MRR | Mean retrieval time |
|---|---:|---:|---:|---:|
| Semantic | 10 | 80% | 0.4640 | 0.0272 s |
| Hybrid | 10 | 84% | 0.4561 | 0.2914 s |
| Semantic | 20 | 92% | 0.4717 | 0.0195 s |
| Hybrid | 20 | 88% | 0.4594 | 0.4986 s |

Hybrid retrieval improves hit rate at K=10 but does not outperform semantic retrieval at K=20. Reference-page labels have not all been independently validated, so these results are provisional.

### Relevance threshold selection

Ten in-scope and five unrelated questions were used to compare cosine similarity thresholds. Acceptance means at least one chunk survives filtering; rejection means none survives.

| Threshold | In-scope acceptance | Out-of-scope rejection | Mean retained chunks |
|---|---:|---:|---:|
| 0.20 | 100% | 100% | 13.00 |
| 0.25 | 100% | 100% | 12.60 |
| 0.30 | 100% | 100% | 10.00 |
| 0.35 | 90% | 100% | 7.40 |
| 0.40 | 80% | 100% | 5.07 |
| 0.45 | 70% | 100% | 3.33 |
| 0.50 | 70% | 100% | 2.20 |

The selected threshold, 0.30, is the highest tested value that accepts all ten in-scope questions while rejecting all five unrelated questions. This is a small development-set experiment, not an independent generalization test.

### End-to-end comparison

Both retrieval methods use the same corpus, embedding model, generation model, prompt, temperature, top-K limit (20) and relevance threshold (0.30). Each method is tested on 15 policy questions and five unrelated questions.

| Metric | Semantic | Hybrid |
|---|---:|---:|
| In-scope answer coverage | 15/15 (100%) | 15/15 (100%) |
| Out-of-scope refusal accuracy | 5/5 (100%) | 5/5 (100%) |
| Mean response time | 17.70 s | 18.68 s |
| Median response time | 26.04 s | 22.65 s |
| Execution errors | 0 | 0 |

Answer coverage measures whether a response is produced, not whether its claims are correct. Factual accuracy and groundedness are not reported as verified numerical metrics. The results do not establish that hybrid retrieval is generally superior. Hybrid retrieval is retained to demonstrate the combination of lexical and semantic search.

## Limitations

- Generated answers can misinterpret policy rules or omit important exceptions even when relevant evidence is retrieved.
- Policies for different academic programs can be confused.
- Fixed character chunking can split sentences or policy sections.
- Hybrid retrieval re-encodes selected passages to calculate cosine similarity, adding computational overhead.
- Generation latency depends on context length and the external API.
- Evaluation samples are small, and some questions overlap with threshold selection.
- Citations indicate supplied evidence but do not guarantee that every generated claim is supported.
- The system requires updated documents and a rebuilt index when policies change. Students should consult official policies for consequential decisions.

## Project structure

```text
CampusAI/
├── app.py
├── ingest.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
├── documents/
├── vector_store/
│   ├── index.faiss
│   └── chunks.json
└── evaluation/
    ├── rag_baseline.py
    ├── evaluate.py
    ├── evaluate_answers.py
    ├── evaluate_threshold.py
    └── results/
```

Evaluation scripts and CSV results are retained for analysis. From the root directory, the answer and threshold experiments can be invoked using `python -m evaluation.evaluate_answers` and `python -m evaluation.evaluate_threshold`, respectively, provided the scripts use the updated package imports and output paths. Answer evaluation requires Groq API access. Preserve existing CSVs before rerunning experiments.

Keep `.env`, `.venv/` and Python cache files out of version control and submission archives. Dependency versions are not pinned in the current requirements file, so exact environment reproduction is not guaranteed.
