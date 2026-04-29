import json
import subprocess
import sys


def test_rag_answers_with_citations_and_trace():
    from chimera_runtime import HallucinationGuardedRAG, RagDocument

    rag = HallucinationGuardedRAG(min_similarity=0.2, max_risk=0.65)
    rag.index(
        [
            RagDocument(
                id="chimera",
                text="ChimeraLang tracks uncertainty with belief distributions and guard checks.",
                metadata={"source": "readme"},
            ),
            RagDocument(
                id="unrelated",
                text="The weather station records wind speed and rainfall.",
            ),
        ]
    )

    result = rag.answer("How does ChimeraLang track uncertainty?")

    assert result.passed
    assert "belief distributions" in result.answer
    assert result.citations[0].document_id == "chimera"
    assert result.confidence >= 0.35
    assert result.trace[0].startswith("indexed 2 document")


def test_rag_refuses_when_retrieval_is_not_grounded():
    from chimera_runtime import HallucinationGuardedRAG, RagDocument

    rag = HallucinationGuardedRAG(min_similarity=0.45, max_risk=0.4)
    rag.index([RagDocument(id="alpha", text="ChimeraLang supports guarded retrieval.")])

    result = rag.answer("What is the warranty period for a dishwasher?")

    assert not result.passed
    assert "not enough retrieved evidence" in result.answer.lower()
    assert result.citations == []
    assert any("below min_similarity" in item for item in result.violations)


def test_rag_blocks_constitution_violations_from_retrieved_context():
    from chimera_runtime import HallucinationGuardedRAG, RagDocument

    rag = HallucinationGuardedRAG(
        min_similarity=0.1,
        max_risk=0.9,
        principles=["Do not provide harmful instructions."],
    )
    rag.index([RagDocument(id="unsafe", text="This document describes a dangerous hack.")])

    result = rag.answer("What dangerous hack is described?")

    assert not result.passed
    assert any("Forbidden content" in item for item in result.violations)


def test_cli_rag_outputs_json(tmp_path):
    docs = tmp_path / "corpus.json"
    docs.write_text(
        json.dumps(
            [
                {
                    "id": "rag",
                    "text": "RAG grounds answers in retrieved context and returns citations.",
                }
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chimera.cli",
            "rag",
            str(docs),
            "--query=How does RAG ground answers?",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["passed"] is True
    assert payload["citations"][0]["document_id"] == "rag"
