"""ChimeraLang CLI — parse, check, and run .chimera programs.

Usage:
  chimera run    <file>   Execute a .chimera program
  chimera check  <file>   Type-check without executing
  chimera rag    <file>   Run hallucination-guarded RAG over a JSON corpus
  chimera lex    <file>   Dump token stream
  chimera parse  <file>   Dump AST
  chimera prove  <file>   Run + generate integrity report
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Ensure stdout can handle Unicode box-drawing characters on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from chimera.detect import HallucinationDetector
from chimera.integrity import IntegrityEngine
from chimera.lexer import Lexer, LexError
from chimera.parser import ParseError, Parser
from chimera.type_checker import TypeChecker
from chimera.vm import ChimeraVM


def _read_source(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        print(f"chimera: error: file not found: {p}", file=sys.stderr)
        sys.exit(1)
    if p.suffix != ".chimera":
        print(f"chimera: warning: expected .chimera extension, got '{p.suffix}'", file=sys.stderr)
    return p.read_text(encoding="utf-8")


def _lex(source: str, filename: str = "<stdin>"):
    lexer = Lexer(source, filename)
    return lexer.tokenize()


def _parse(source: str, filename: str = "<stdin>"):
    tokens = _lex(source, filename)
    parser = Parser(tokens)
    return parser.parse()


def cmd_lex(path: str) -> None:
    """Dump the token stream."""
    source = _read_source(path)
    try:
        tokens = _lex(source, path)
        for tok in tokens:
            print(f"{tok.span.line:4d}:{tok.span.col:<3d}  {tok.kind.name:<24s} {tok.value!r}")
    except LexError as e:
        print(f"chimera: lex error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_parse(path: str) -> None:
    """Dump the AST as indented repr."""
    source = _read_source(path)
    try:
        program = _parse(source, path)
        for decl in program.declarations:
            _print_node(decl, indent=0)
    except (LexError, ParseError) as e:
        print(f"chimera: parse error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_check(path: str) -> None:
    """Type-check a .chimera file."""
    source = _read_source(path)
    try:
        program = _parse(source, path)
    except (LexError, ParseError) as e:
        print(f"chimera: parse error: {e}", file=sys.stderr)
        sys.exit(1)

    checker = TypeChecker()
    result = checker.check(program)

    for w in result.warnings:
        print(f"  warning: {w}")
    for e in result.errors:
        print(f"  ERROR: {e}")
    if result.ok:
        print(f"chimera: {path} — type check PASSED")
    else:
        print(f"chimera: {path} — type check FAILED ({len(result.errors)} error(s))")
        sys.exit(1)


def cmd_run(path: str, *, show_trace: bool = False, extra_args: list[str] | None = None) -> None:
    """Execute a .chimera program."""
    source = _read_source(path)
    try:
        program = _parse(source, path)
    except (LexError, ParseError) as e:
        print(f"chimera: parse error: {e}", file=sys.stderr)
        sys.exit(1)

    # Route to CIR path if program uses belief constructs
    from chimera.ast_nodes import BeliefDecl
    uses_cir = any(isinstance(d, BeliefDecl) for d in program.declarations)

    if uses_cir:
        from chimera.cir import run_cir
        args = extra_args or []
        save_sym = next(
            (a.split("=", 1)[1] for a in args if a.startswith("--save-symbols=")), None
        )
        load_sym = next(
            (a.split("=", 1)[1] for a in args if a.startswith("--load-symbols=")), None
        )
        cir_result = run_cir(program, save_symbols=save_sym, load_symbols=load_sym)

        if cir_result.emitted:
            for name, dist in cir_result.emitted:
                print(f"  emit: {name}  [mean={dist.mean:.3f} variance={dist.variance:.4f}]")

        if show_trace and cir_result.trace:
            print("\n— CIR Reasoning Trace —")
            for entry in cir_result.trace:
                print(f"  {entry}")

        if cir_result.guard_violations:
            print("\n— Guard Violations —")
            for v in cir_result.guard_violations:
                print(f"  {v}")

        warnings = cir_result.meta.get("lowering_warnings", [])
        if warnings:
            print("\n— Lowering Warnings —")
            for w in warnings:
                print(f"  {w}")

        print(f"\nchimera: {path} — CIR executed in {cir_result.duration_ms:.1f}ms")
        if cir_result.guard_violations:
            sys.exit(1)
        return

    # Existing VM path (fn/gate/goal/reason programs)
    vm = ChimeraVM()
    result = vm.execute(program)

    if result.emitted:
        for val in result.emitted:
            conf_str = f"{val.confidence.value:.2f}"
            print(f"  emit: {val.raw}  [confidence={conf_str}]")

    if show_trace and result.trace:
        print("\n— Reasoning Trace —")
        for entry in result.trace:
            print(f"  {entry}")

    if result.errors:
        print("\n— Errors —")
        for e in result.errors:
            print(f"  {e}")
        sys.exit(1)

    print(
        f"\nchimera: {path} — executed in {result.duration_ms:.1f}ms"
        f" (assertions: {result.assertions_passed} passed, {result.assertions_failed} failed)"
    )


def cmd_compile(path: str, backend: str = "pytorch", out: str | None = None) -> None:
    """Compile a .chimera program to a supported backend."""
    source = _read_source(path)
    try:
        program = _parse(source, path)
    except (LexError, ParseError) as e:
        print(f"chimera: parse error: {e}", file=sys.stderr)
        sys.exit(1)

    if backend == "pytorch":
        from chimera.compiler import compile_to_pytorch
        code = compile_to_pytorch(program)
    elif backend == "llvm":
        from chimera.compiler import compile_to_llvm
        code = compile_to_llvm(program)
    else:
        print(f"chimera: unsupported backend '{backend}'", file=sys.stderr)
        sys.exit(1)

    if out:
        Path(out).write_text(code, encoding="utf-8")
        print(f"chimera: compiled {path} -> {out} ({len(code)} chars)")
    else:
        print(code)


def cmd_rag(
    path: str,
    *,
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.25,
    as_json: bool = False,
) -> None:
    """Run hallucination-guarded RAG over a JSON corpus."""
    from chimera_runtime import HallucinationGuardedRAG

    p = Path(path)
    if not p.exists():
        print(f"chimera: error: corpus not found: {p}", file=sys.stderr)
        sys.exit(1)
    if not query.strip():
        print("chimera: rag requires --query=TEXT", file=sys.stderr)
        sys.exit(1)

    try:
        documents = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"chimera: invalid corpus JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(documents, list):
        print("chimera: corpus must be a JSON array of documents", file=sys.stderr)
        sys.exit(1)

    rag = HallucinationGuardedRAG(top_k=top_k, min_similarity=min_similarity)
    try:
        rag.index(documents)
        result = rag.answer(query, top_k=top_k)
    except ValueError as e:
        print(f"chimera: rag error: {e}", file=sys.stderr)
        sys.exit(1)

    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        if not result.passed:
            sys.exit(1)
        return

    print(result.answer)
    print(f"\nconfidence={result.confidence:.3f} variance={result.variance:.4f} passed={result.passed}")
    if result.citations:
        print("\nCitations:")
        for citation in result.citations:
            print(f"  - {citation.document_id} score={citation.score:.3f}")
    if result.violations:
        print("\nGuard violations:")
        for violation in result.violations:
            print(f"  - {violation}")
    if not result.passed:
        sys.exit(1)


def cmd_prove(path: str) -> None:
    """Execute + full integrity report."""
    source = _read_source(path)
    try:
        program = _parse(source, path)
    except (LexError, ParseError) as e:
        print(f"chimera: parse error: {e}", file=sys.stderr)
        sys.exit(1)

    # Execute
    vm = ChimeraVM()
    exec_result = vm.execute(program)

    # Hallucination scan
    detector = HallucinationDetector()
    detection = detector.full_scan(exec_result.gate_logs, exec_result.emitted)

    # Integrity report
    engine = IntegrityEngine()
    report = engine.certify(exec_result, detection, source)

    # Print report
    print("═══════════════════════════════════════════════")
    print("        CHIMERALANG INTEGRITY REPORT           ")
    print("═══════════════════════════════════════════════")
    print(report.to_json())
    print("═══════════════════════════════════════════════")
    print(f"  Verdict: {report.verdict}")
    print("═══════════════════════════════════════════════")


def cmd_repl() -> None:
    """Interactive Read-Eval-Print Loop for ChimeraLang."""
    import readline  # noqa: F401 — enables history & arrow keys on most platforms

    print("ChimeraLang REPL v0.2.0  (type ':exit' or Ctrl-D to quit, ':help' for help)")
    print()

    vm = ChimeraVM()
    # Persistent environment across REPL lines
    accumulated: list[str] = []
    indent_keywords = {"fn", "gate", "goal", "reason", "if", "for", "match", "detect"}
    dedent_keyword = "end"
    depth = 0

    def _eval_buffer(lines: list[str]) -> None:
        source = "\n".join(lines)
        try:
            tokens = _lex(source)
            program = Parser(tokens).parse()
        except (LexError, ParseError) as e:
            print(f"  parse error: {e}")
            return
        result = vm.execute(program)
        for val in result.emitted:
            conf_str = f"{val.confidence.value:.4f}"
            print(f"  >> {val.raw!r}  [confidence={conf_str}]")
        for err in result.errors:
            print(f"  error: {err}")

    while True:
        prompt = "chimera" + ("..." * depth) + "> "
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = line.strip()

        if stripped == ":exit":
            break
        if stripped == ":help":
            print("  :exit         Quit the REPL")
            print("  :clear        Reset REPL state")
            print("  :trace        Toggle reasoning trace output")
            print("  :help         Show this message")
            print("  Any .chimera code is accepted; multi-line blocks use 'end'")
            continue
        if stripped == ":clear":
            vm = ChimeraVM()
            accumulated = []
            depth = 0
            print("  [REPL state cleared]")
            continue
        if stripped == ":trace":
            print("  [trace display not yet configurable in REPL — use 'chimera run --trace']")
            continue

        accumulated.append(line)

        # Track block depth
        first_word = stripped.split()[0] if stripped.split() else ""
        if first_word in indent_keywords:
            depth += 1
        elif first_word == dedent_keyword:
            depth = max(0, depth - 1)

        # Execute when we're back at the top level
        if depth == 0 and accumulated:
            _eval_buffer(accumulated)
            accumulated = []



    prefix = "  " * indent
    name = type(node).__name__
    print(f"{prefix}{name}", end="")
    if hasattr(node, "name"):
        print(f" ({node.name})", end="")  # type: ignore[attr-defined]
    if hasattr(node, "description"):
        print(f' ("{node.description}")', end="")  # type: ignore[attr-defined]
    print()
    for attr in ("params", "body", "declarations", "then_body", "else_body"):
        children = getattr(node, attr, None)
        if isinstance(children, list):
            for child in children:
                _print_node(child, indent + 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

USAGE = """\
ChimeraLang v0.2.0 — A programming language for AI cognition

Usage:
  chimera run     <file.chimera>                   Execute a program
  chimera check   <file.chimera>                   Type-check only
  chimera lex     <file.chimera>                   Dump token stream
  chimera parse   <file.chimera>                   Dump AST
  chimera prove   <file.chimera>                   Run + integrity report
  chimera compile <file.chimera> [--backend=pytorch] [--out=file.py]
                                                   Compile to PyTorch Python
  chimera rag     <corpus.json> --query=TEXT [--top-k=N] [--min-similarity=N] [--json]
                                                   Run hallucination-guarded RAG
  chimera repl                                     Interactive REPL

Options:
  --trace           Show reasoning trace (with run)
  --backend=NAME    Compilation backend (default: pytorch)
  --out=FILE        Write compiled output to file instead of stdout
  --query=TEXT      RAG query
  --top-k=N         RAG retrieval count (default: 3)
  --min-similarity=N
                   Minimum retrieval similarity (default: 0.25)
  --json            Emit JSON for RAG
  --help            Show this message
"""


def main() -> None:
    args = sys.argv[1:]
    if not args or "--help" in args or "-h" in args:
        print(USAGE)
        sys.exit(0)

    cmd = args[0]

    # REPL doesn't need a file argument
    if cmd == "repl":
        cmd_repl()
        return

    rest = args[1:]
    trace = "--trace" in rest
    positional = [a for a in rest if not a.startswith("--")]

    if not positional:
        print(f"chimera: '{cmd}' requires a file argument", file=sys.stderr)
        sys.exit(1)

    filepath = positional[0]

    flag_args = [a for a in rest if a.startswith("--") and a != "--trace"]
    commands = {
        "run": lambda: cmd_run(filepath, show_trace=trace, extra_args=flag_args),
        "check": lambda: cmd_check(filepath),
        "lex": lambda: cmd_lex(filepath),
        "parse": lambda: cmd_parse(filepath),
        "prove": lambda: cmd_prove(filepath),
        "compile": lambda: cmd_compile(
            filepath,
            backend=next((a.split("=")[1] for a in flag_args if a.startswith("--backend=")), "pytorch"),
            out=next((a.split("=")[1] for a in flag_args if a.startswith("--out=")), None),
        ),
        "rag": lambda: cmd_rag(
            filepath,
            query=next((a.split("=", 1)[1] for a in flag_args if a.startswith("--query=")), ""),
            top_k=int(next((a.split("=", 1)[1] for a in flag_args if a.startswith("--top-k=")), "3")),
            min_similarity=float(next((a.split("=", 1)[1] for a in flag_args if a.startswith("--min-similarity=")), "0.25")),
            as_json="--json" in flag_args,
        ),
    }

    handler = commands.get(cmd)
    if handler is None:
        print(f"chimera: unknown command '{cmd}'", file=sys.stderr)
        print(USAGE)
        sys.exit(1)

    handler()


if __name__ == "__main__":
    main()
