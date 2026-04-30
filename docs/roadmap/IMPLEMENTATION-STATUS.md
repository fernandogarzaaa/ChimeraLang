# ChimeraLang ML Roadmap Implementation Status

This file tracks the implementation state of `CHIMERALANG-ML-SPEC-V2.md` in the current production-ready alpha.

## Implemented

| Roadmap area | Implementation |
|---|---|
| ML type system | Tensor dimensionality, causal/privacy annotations, vector store types, spike trains, multimodal payloads, and memory pointer types are parsed into AST nodes and resolved by the type checker. |
| Model declarations | Braced and `end` model bodies are supported with layer declarations, train blocks, constitution blocks, retrieval blocks, and MoE layers. |
| Retrieval and vector memory | `retrieval` blocks are validated during type checking and generated PyTorch modules initialize runtime `VectorStore` instances. |
| Mixture-of-experts | MoE layers emit executable PyTorch routing code and record load-balancing auxiliary losses. |
| Roadmap declarations | Causal models, federated training, meta-training, self-improvement, swarms, replay buffers, reward systems, and predictive coding declarations are parsed and validated. |
| Runtime systems | `chimera_runtime` exports vector storage, spiking primitives, swarm message/consensus helpers, and roadmap system containers. |
| Compiler backends | `chimera compile` supports `pytorch` and `llvm` backends. The PyTorch backend emits executable model modules; the LLVM backend emits typed IR scaffolds for model declarations. |
| Production validation | Invalid dimensions, duplicate symbols, invalid retrieval settings, malformed constitutions, bad roadmap declarations, and vector-store misuse are covered by tests. |
| CI and packaging | GitHub Actions test Python 3.11-3.13, run CLI compiler smokes, build wheel and sdist artifacts, and upload build outputs. |

## Verification Targets

Run these commands from the repository root before release:

```bash
python -m pytest
python -m chimera.cli check examples/mnist_classifier.chimera
python -m chimera.cli compile examples/mnist_classifier.chimera --backend=pytorch --out=build/mnist_classifier.py
python -m py_compile build/mnist_classifier.py
python -m chimera.cli compile examples/mnist_classifier.chimera --backend=llvm --out=build/mnist_classifier.ll
python -m build
```

## Known Alpha Boundaries

The roadmap is implemented to production-ready alpha depth rather than as a full optimizing ML stack. The LLVM backend currently emits typed IR scaffolding (function signatures, basic-block structure, per-layer commentary); it is **not yet guaranteed to assemble cleanly under `llvm-as`** — several layer bodies reference SSA values whose definitions are pending and will be filled in during Phase 2 completion. Federated training, causal inference, swarm coordination, predictive coding, and self-improvement are represented as validated language/runtime surfaces ready for deeper execution engines.
