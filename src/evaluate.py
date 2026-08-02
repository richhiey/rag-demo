from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from common import format_context, load_config, load_source_documents, split_documents
from rag_pipeline import build_llm, generate_answer, retrieve


@dataclass(frozen=True)
class GoldenExample:
    question: str
    expected_source: str
    required_terms: tuple[str, ...]


GOLDEN_SET = [
    GoldenExample(
        "How can I reach the Nationalparkzentrum Ruhestein, and does the official page mention ÖPNV?",
        "schwarzwald-arrival",
        ("Ruhestein", "ÖPNV"),
    ),
    GoldenExample(
        "Is the Spechtpfad an easy accessible trail for a wheelchair or stroller, and is it 1,2 km long?",
        "schwarzwald-easy-tours",
        ("Spechtpfad", "1,2"),
    ),
    GoldenExample(
        "What does the Wegesperrungen page say about checking the interactive map tagesaktuell before a visit?",
        "schwarzwald-closures",
        ("Wegesperrungen", "tagesaktuell"),
    ),
    GoldenExample(
        "What are the Öffnungszeiten for the Nationalparkzentrum: 10 to 17 in winter and 10 to 18 in summer?",
        "schwarzwald-tickets",
        ("Öffnungszeiten", "10"),
    ),
]


def retrieval_hit_rate(k: int) -> tuple[int, int]:
    hits = 0
    for example in GOLDEN_SET:
        docs = retrieve(example.question, k)
        source_ids = {doc.metadata.get("source_id") for doc in docs}
        hit = example.expected_source in source_ids
        hits += int(hit)
        print(f"retrieval | expected={example.expected_source} | hit={hit}")
    return hits, len(GOLDEN_SET)


def citation_check(answer: str, documents) -> bool:
    urls = {doc.metadata.get("source_url") for doc in documents}
    return any(url and url in answer for url in urls)


def answer_relevance_proxy(answer: str, example: GoldenExample) -> bool:
    """Small deterministic proxy for a golden internal-data question.

    This is intentionally not presented as a general semantic evaluator. The
    optional Groq judge provides a broader model-based check when configured.
    """
    answer_lower = answer.lower()
    return bool(answer.strip()) and all(
        term.lower() in answer_lower for term in example.required_terms
    )


def groundedness_proxy(answer: str, documents, example: GoldenExample) -> bool:
    """Check that the golden answer's key phrases appear in context and answer."""
    context = format_context(documents).lower()
    answer_lower = answer.lower()
    return all(
        phrase.lower() in context and phrase.lower() in answer_lower
        for phrase in example.required_terms
    )


def run_basic_checks(offline: bool = False) -> None:
    documents = load_source_documents()
    chunks = split_documents(documents)
    metadata_complete = all(
        bool(doc.metadata.get("source_id")) and bool(doc.metadata.get("source_url"))
        for doc in documents
    )
    print(f"ingestion_documents={len(documents)}")
    print(f"ingestion_metadata_complete={metadata_complete}")
    print(f"ingestion_non_empty_chunks={bool(chunks) and all(chunk.page_content.strip() for chunk in chunks)}")

    k = int(os.getenv("RETRIEVAL_K", "3"))
    hits, total = retrieval_hit_rate(k)
    print(f"retrieval_hit_at_{k}={hits / total:.2f}")

    for index, example in enumerate(GOLDEN_SET, start=1):
        docs = retrieve(example.question, k)
        answer = generate_answer(example.question, docs, offline=offline)
        print(f"answer_{index}_relevance_proxy={answer_relevance_proxy(answer, example)}")
        print(f"answer_{index}_groundedness_proxy={groundedness_proxy(answer, docs, example)}")
        print(f"answer_{index}_has_source_url={citation_check(answer, docs)}")

    unsupported = "How do I tune a bicycle in downtown Baden-Baden?"
    unsupported_docs = retrieve(unsupported, k)
    unsupported_answer = generate_answer(unsupported, unsupported_docs, offline=offline)
    limitation_terms = (
        "insufficient",
        "does not contain enough",
        "not enough",
        "cannot",
        "don't have",
        "do not have",
    )
    refused = any(term in unsupported_answer.lower() for term in limitation_terms)
    print(f"unsupported_question_acknowledges_limit={refused}")


def run_llm_judge() -> None:
    question = GOLDEN_SET[0].question
    documents = retrieve(question, int(os.getenv("RETRIEVAL_K", "3")))
    answer = generate_answer(question, documents)
    judge_prompt = (
        "You are evaluating a RAG answer. Return exactly two lines: "
        "RELEVANCE: pass or fail; GROUNDEDNESS: pass or fail. "
        "Pass relevance if the answer addresses the question. "
        "Pass groundedness only if its claims are supported by the context.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{format_context(documents)}\n\nANSWER:\n{answer}"
    )
    judged = build_llm().invoke(judge_prompt).content
    print("llm_judge:")
    print(judged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer behavior.")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live generation and use the deterministic context-based graph path.",
    )
    args = parser.parse_args()
    load_config()
    run_basic_checks(offline=args.offline)
    if args.llm_judge:
        run_llm_judge()


if __name__ == "__main__":
    main()
