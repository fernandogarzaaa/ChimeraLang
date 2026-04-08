# ChimeraLang

**A programming language designed for AI cognition** — probabilistic types, quantum consensus gates, directed hallucination, and cryptographic integrity proofs.

ChimeraLang treats uncertainty, confidence, and epistemic state as **first-class language primitives** rather than bolted-on libraries. Programs in ChimeraLang describe *how an AI should think*, not just what it should compute.

---

## Key Features

| Feature | Description |
|---|---|
| **Probabilistic Types** | `Confident<T>`, `Explore<T>`, `Converge<T>`, `Provisional<T>` — types that carry confidence scores |
| **Quantum Consensus Gates** | Multiple candidate values vote under Gaussian noise; the result is the *consensus* of an ensemble |
| **Hallucination Detection** | Inline `detect` blocks with `range`, `dictionary`, and `confidence_threshold` strategies |
| **Cryptographic Integrity** | Merkle-chain proofs and gate certificates ensure reasoning traces are tamper-evident |
| **Memory Modifiers** | `Ephemeral`, `Persistent`, `Provisional` — explicit lifecycle for every binding |
| **Intent-First Goals** | `goal` blocks declare desired outcomes; `reasoning` blocks show the derivation |
| **Control Flow** | `for x in list`, `match expr | pattern => body`, `if/else` |
| **Built-in Functions** | `len`, `sum`, `max_val`, `min_val`, `abs_val`, `floor`, `ceil`, `round_val`, `confident`, `explore`, `confidence_of` |
| **Interactive REPL** | `chimera repl` — try the language live in your terminal |

## Quick Start

### Prerequisites

- Python ≥ 3.11

### Run an example

```bash
cd ChimeraLang
python -m chimera.cli run examples/hello_chimera.chimera
```

### Available commands

```
python -m chimera.cli run    <file>   # Execute a .chimera program
python -m chimera.cli check  <file>   # Type-check without running
python -m chimera.cli prove  <file>   # Run and generate integrity proof
python -m chimera.cli ast    <file>   # Print the AST
python -m chimera.cli tokens <file>   # Print the token stream
python -m chimera.cli repl           # Interactive REPL
```

## Examples

Six example programs are included in [`examples/`](examples/):

| File | What it demonstrates |
|---|---|
| `hello_chimera.chimera` | Basic emit, confident values |
| `quantum_reasoning.chimera` | Consensus gates with Gaussian noise, confidence propagation |
| `goal_driven.chimera` | Goals, reasoning blocks, semantic constraints |
| `hallucination_guard.chimera` | All 5 hallucination-detection strategies |
| `for_loop.chimera` | For loops, list builtins, match expressions, two-arg constructors |
| `advanced_reasoning.chimera` | Detect blocks, nested gates + reason, aggregate confidence |

### Sample run

```
$ python -m chimera.cli run examples/for_loop.chimera

  emit: 4.08  [confidence=1.00]
  emit: 5  [confidence=1.00]
  emit: high  [confidence=1.00]
  emit: medium  [confidence=1.00]
  ...
  emit: running  [confidence=1.00]
  emit: 3.14159  [confidence=0.99]
  emit: dark energy is repulsive  [confidence=0.65]

chimera: examples/for_loop.chimera — executed in 0.2ms (assertions: 1 passed, 0 failed)
```

## Project Structure

```
ChimeraLang/
├── chimera/                  # Core language implementation
│   ├── tokens.py             # 75+ token types
│   ├── lexer.py              # Tokenizer
│   ├── ast_nodes.py          # AST node hierarchy
│   ├── parser.py             # Recursive-descent parser
│   ├── types.py              # Runtime type system & confidence propagation
│   ├── type_checker.py       # Static type checker
│   ├── vm.py                 # Quantum Consensus VM
│   ├── detect.py             # Hallucination detector
│   ├── integrity.py          # Merkle chains & gate certificates
│   └── cli.py                # Command-line interface + REPL
├── examples/                 # Example .chimera programs
├── spec/
│   └── SPEC.md               # Formal language specification
├── paper/
│   └── chimeralang.tex       # ArXiv whitepaper (LaTeX)
├── tests/                    # Test suite (60 tests)
└── pyproject.toml
```

## Language Overview

### Probabilistic Types

```chimera
val answer: Confident<Int> = confident(42, 0.95)
val idea:   Explore<Text>  = explore("maybe this?", 0.60)
```

Every value carries a **confidence score** (0.0–1.0). Confidence propagates through operations — if you combine two uncertain values, the result inherits a combined confidence.

### Quantum Consensus Gates

```chimera
gate consensus_answer(question: Text) -> Converge<Text>
  branches: 5
  collapse: weighted_vote
  threshold: 0.80
  val answer: Text = "Reasoned answer to: " + question
  return answer
end
val result = consensus_answer("What causes consciousness?")
```

Candidates are perturbed with Gaussian noise, then collapsed via `majority`, `weighted_vote`, or `highest_confidence`. The gate only passes if collective confidence meets the threshold.

### For Loops

```chimera
val scores: List<Float> = [0.92, 0.76, 0.88, 0.55, 0.97]
for s in scores
  emit s
end
```

Iterate over any `List<T>` or `Text` (character by character).

### Match Expressions

```chimera
val status: Int = 2
match status
  | 1 => emit "initialising"
  | 2 => emit "running"
  | 3 => emit "done"
  | _ => emit "unknown"
end
```

Pattern-match against literals; `_` is the wildcard arm.

### Hallucination Detection Blocks

```chimera
val temperature: Float = 37.2

detect hallucination
  strategy: "range"
  on: temperature
  valid_range: [-50.0, 60.0]
  action: "flag"
end
```

Inline `detect` blocks record probes to the reasoning trace; out-of-range values are automatically flagged.

### Built-in Functions

| Function | Description |
|---|---|
| `confident(val, score)` | Construct a `ConfidentValue` with the given score |
| `explore(val, score)` | Construct an `ExploreValue` with the given score |
| `confidence_of(val)` | Extract the confidence float from any value |
| `consensus(val)` | True if a `ConvergeValue` reached multi-branch consensus |
| `no_hallucination(val)` | True if value has trace and confidence > 0.5 |
| `len(collection)` | Length of a list or string |
| `sum(list)` | Sum of a numeric list |
| `max_val(list)` | Maximum element |
| `min_val(list)` | Minimum element |
| `abs_val(x)` | Absolute value |
| `floor(x)` | Floor of a float |
| `ceil(x)` | Ceiling of a float |
| `round_val(x, n?)` | Round to n decimal places |
| `print(...)` | Log to reasoning trace |

### Integrity Proofs

```bash
python -m chimera.cli prove examples/quantum_reasoning.chimera
```

Generates a Merkle-chain proof with SHA-256 hashes so that every step of the reasoning trace is tamper-evident.

### Interactive REPL

```bash
python -m chimera.cli repl
```

```
ChimeraLang REPL v0.2.0  (type ':exit' or Ctrl-D to quit, ':help' for help)

chimera> val x = confident(42, 0.98)
chimera> emit x
  >> 42  [confidence=0.9800]
chimera> :exit
```

## Academic Paper

A full whitepaper is available in [`paper/chimeralang.tex`](paper/chimeralang.tex). It covers the formal type system, execution model, consensus algorithms, and cryptographic integrity framework with 25 academic references.

To compile:
```bash
pdflatex chimeralang.tex   # or upload to Overleaf
bibtex chimeralang
pdflatex chimeralang.tex
pdflatex chimeralang.tex
```

## How It Differs

| Aspect | Traditional Languages | ChimeraLang |
|---|---|---|
| Values | Deterministic | Carry confidence scores |
| Execution | Single-path | Ensemble consensus |
| Correctness | Tests/assertions | Continuous hallucination detection |
| Auditability | Logs | Cryptographic Merkle proofs |
| Intent | Implicit in code | Explicit `goal` declarations |
| Control flow | Loops, conditionals | `for`, `match`, `if/else` with confidence propagation |

## License

MIT

## Citation

```bibtex
@article{chimeralang2025,
  title   = {ChimeraLang: A Programming Language for AI Cognition},
  year    = {2025},
  note    = {https://github.com/fernandogarzaaa/ChimeraLang}
}
```
