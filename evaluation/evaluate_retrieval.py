import argparse
import json
import tempfile
from pathlib import Path

import config_data as config
from agents.router import DomainRouterAgent
from evaluation.llm_judge import LLMJudge
from services.chat_service import OfficeMateChatService
from services.lexical_index import BM25LexicalIndex
from services.vector_store import OfficeMateVectorStore


def build_lexical_baseline():
    temporary_directory = tempfile.TemporaryDirectory()
    index = BM25LexicalIndex(Path(temporary_directory.name) / "lexical.json")
    for sample in config.SAMPLE_DOCS:
        content = (config.SAMPLE_DOC_DIR / sample["file_name"]).read_text(encoding="utf-8")
        index.add_document(
            sample["file_name"],
            [content],
            [
                {
                    "title": sample["title"],
                    "category": sample["category"],
                    "version": sample["version"],
                    "file_name": sample["file_name"],
                }
            ],
        )
    return index, temporary_directory


def evaluate_retrieval(mode):
    """检索评估：路由准确率、Hit@K、MRR、拒答准确率。"""
    cases_path = Path(__file__).with_name("cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    router = DomainRouterAgent()
    temporary_directory = None
    if mode == "hybrid":
        retriever = OfficeMateVectorStore()
    else:
        retriever, temporary_directory = build_lexical_baseline()

    route_hits = 0
    retrieval_hits = 0
    reciprocal_rank_sum = 0.0
    abstention_hits = 0
    evidence_cases = [case for case in cases if case["expected_title"]]
    abstention_cases = [case for case in cases if not case["expected_title"]]

    try:
        for case in cases:
            route = router.route(case["question"])
            route_hits += route.category == case["expected_category"]
            results = retriever.search(case["question"], category=route.category)
            if mode == "lexical":
                results = [result for result in results if result.score >= config.min_relevance_score]
            titles = [result.document.metadata.get("title", "") for result in results]
            expected_title = case["expected_title"]
            if expected_title and expected_title in titles:
                rank = titles.index(expected_title) + 1
                retrieval_hits += 1
                reciprocal_rank_sum += 1 / rank
            if expected_title is None and not results:
                abstention_hits += 1

            diagnostics = [format_diagnostic(result, mode) for result in results]
            print(f"[{case['question']}] route={route.category} results={diagnostics}")
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    print(f"\n=== {mode.title()} Retrieval Evaluation ===")
    print(f"Cases           : {len(cases)}")
    print(f"Route accuracy  : {route_hits / len(cases):.1%}")
    print(f"Hit@K           : {retrieval_hits / max(len(evidence_cases), 1):.1%}")
    print(f"MRR             : {reciprocal_rank_sum / max(len(evidence_cases), 1):.3f}")
    print(f"Abstention acc. : {abstention_hits / max(len(abstention_cases), 1):.1%}")

    return {
        "route_accuracy": route_hits / len(cases),
        "hit_k": retrieval_hits / max(len(evidence_cases), 1),
        "mrr": reciprocal_rank_sum / max(len(evidence_cases), 1),
        "abstention_accuracy": abstention_hits / max(len(abstention_cases), 1),
    }


def evaluate_judge():
    """LLM-as-Judge 语义评估：对完整问答流程进行多维度评估。"""
    cases_path = Path(__file__).with_name("cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    try:
        from services.model_factory import create_chat_model
        llm = create_chat_model()
    except Exception as exc:
        print(f"无法初始化 LLM: {exc}")
        return None

    judge = LLMJudge(llm)
    service = OfficeMateChatService(chat_model=llm)

    # 评估用例（只评估有期望答案的 case）
    judge_cases = [c for c in cases if c["expected_title"]]

    dimension_scores = {
        "faithfulness": [],
        "relevance": [],
        "format": [],
        "completeness": [],
    }
    overall_scores = []
    all_issues = []

    for case in judge_cases:
        try:
            result = service.answer_question(case["question"], session_id="eval")
            answer = result.get("answer", "")
            context = "\n\n".join(
                f"[{i+1}] {d.get('title', '')}: {d.get('excerpt', '')}"
                for i, d in enumerate(result.get("source_docs", []))
            )

            report = judge.judge(case["question"], answer, context)

            dimension_scores["faithfulness"].append(report.faithfulness.score)
            dimension_scores["relevance"].append(report.relevance.score)
            dimension_scores["format"].append(report.format_compliance.score)
            dimension_scores["completeness"].append(report.completeness.score)
            overall_scores.append(report.overall)
            all_issues.extend(report.issues)

            print(f"[{case['question']}]")
            print(f"  Faithfulness: {report.faithfulness.score}/5 - {report.faithfulness.reason}")
            print(f"  Relevance   : {report.relevance.score}/5 - {report.relevance.reason}")
            print(f"  Format      : {report.format_compliance.score}/5 - {report.format_compliance.reason}")
            print(f"  Completeness: {report.completeness.score}/5 - {report.completeness.reason}")
            print(f"  Overall     : {report.overall:.2f}")
            if report.issues:
                print(f"  Issues      : {report.issues}")
            print()
        except Exception as exc:
            print(f"[{case['question']}] 评估失败: {exc}")

    print("\n=== LLM-as-Judge Evaluation ===")
    print(f"Cases           : {len(judge_cases)}")
    for dim, scores in dimension_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"{dim.capitalize():15s}: {avg:.2f} / 5.00")
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print(f"{'Overall':15s}: {overall_avg:.2f} / 1.00")
    if all_issues:
        print(f"\nCommon Issues:")
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  - {issue}: {count}")

    return {
        "faithfulness": sum(dimension_scores["faithfulness"]) / max(len(dimension_scores["faithfulness"]), 1),
        "relevance": sum(dimension_scores["relevance"]) / max(len(dimension_scores["relevance"]), 1),
        "format": sum(dimension_scores["format"]) / max(len(dimension_scores["format"]), 1),
        "completeness": sum(dimension_scores["completeness"]) / max(len(dimension_scores["completeness"]), 1),
        "overall": overall_avg,
    }


def format_diagnostic(result, mode):
    diagnostic = {
        "title": result.document.metadata.get("title", ""),
        "score": result.score if mode == "lexical" else result.relevance_score,
    }
    if mode == "hybrid":
        diagnostic.update({"rrf": result.rrf_score, "sources": result.retrieval_sources})
    return diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OfficeMate retrieval and answer quality.")
    parser.add_argument(
        "--mode",
        choices=("lexical", "hybrid", "judge", "all"),
        default="lexical",
        help="lexical: BM25 offline | hybrid: vector+BM25+RRF (needs API) | judge: LLM-as-Judge (needs API) | all: full evaluation",
    )
    args = parser.parse_args()

    if args.mode in ("lexical", "hybrid"):
        evaluate_retrieval(args.mode)
    elif args.mode == "judge":
        evaluate_judge()
    elif args.mode == "all":
        print("=" * 60)
        print("Phase 1: Retrieval Evaluation")
        print("=" * 60)
        evaluate_retrieval("hybrid")
        print("\n" + "=" * 60)
        print("Phase 2: LLM-as-Judge Evaluation")
        print("=" * 60)
        evaluate_judge()
