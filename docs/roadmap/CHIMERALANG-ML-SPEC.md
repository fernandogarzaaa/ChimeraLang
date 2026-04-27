# ChimeraLang ML — Full Language Spec & Blueprint
**Version:** 0.1-draft  
**Date:** 2026-04-27  
**Status:** Design Phase

---

## Executive Summary

ChimeraLang evolves from a reasoning DSL into a full programming language for building AI models and ML algorithms. Its irreducible differentiator: **uncertainty is a first-class citizen at the language level** — not a library, not a pattern, not an afterthought.

Two implementation paths are specified:

| | Path A | Path B |
|---|---|---|
| **Strategy** | Compile to PyTorch/JAX | Native LLVM compiler |
| **Time to first demo** | 2–3 months | 12–18 months |
| **GPU support** | Day 1 (via backend) | After tensor runtime |
| **Performance ceiling** | Backend-limited | Unlimited |
| **Risk** | Low | High |
| **Recommended for** | Now | After Path A proves traction |

---

## 1. Language Design (Shared — Both Paths)

### 1.1 Philosophy

> Computation = belief transformation under resource constraints.

Three additions to the existing ChimeraLang philosophy:
1. **Shape safety** — tensor shapes are part of the type, checked at compile time
2. **Gradient awareness** — every differentiable value knows it is differentiable
3. **Epistemic ML** — models carry uncertainty estimates as first-class values, not post-hoc calibration

### 1.2 New Primitive Types

```chimera
// Scalars (existing, extended)
Int, Float, Bool, Text

// Tensor type with shape annotation
Tensor<dtype>[dim1, dim2, ...]
  // e.g. Tensor<Float>[batch, 784]
  // e.g. Tensor<Float>[seq_len, hidden]
  // ? means dynamic dimension

// Distribution types (extend existing BetaDist)
Distribution<T>          // abstract
Normal<T>[shape]         // Gaussian
Beta<T>[shape]           // existing belief system
Categorical<T>[classes]  // softmax output
Dirichlet<T>[k]          // concentration over categories

// Gradient-tracked scalar
Grad<T>                  // T that participates in autograd

// Device-aware tensor
Tensor<dtype>[shape] on cpu | gpu | tpu
```

### 1.3 Module System

```chimera
// Defining a module
module chimera.nn

// Importing
import chimera.nn { Dense, ReLU, Conv2d }
import chimera.data { DataLoader, mnist }
import chimera.optim { Adam, SGD }
import chimera.loss { CrossEntropy, MSE }

// Aliased import
import chimera.tensor as T
```

### 1.4 Model Definition Syntax

```chimera
model Classifier {
  // Layer declarations with shape flow
  layer fc1: Dense(784 -> 128)
  layer act1: ReLU
  layer fc2: Dense(128 -> 10)
  layer out:  Softmax

  // Uncertainty mode: point | epistemic | aleatoric | full
  uncertainty: epistemic

  // Forward pass — shape-annotated
  forward(x: Tensor<Float>[?, 784]) -> Distribution<Categorical>[10] {
    val h1 = act1(fc1(x))          // shape: [?, 128]
    val logits = fc2(h1)            // shape: [?, 10]
    return out(logits)
  }
}
```

### 1.5 Training Loop Syntax

```chimera
train Classifier on mnist_train {
  optimizer: Adam { lr: 0.001, weight_decay: 1e-4 }
  loss: CrossEntropy
  epochs: 50
  batch_size: 128

  // CIR belief guards apply during training
  guard predictions against overconfidence { max_risk: 0.05, strategy: variance }

  // Gradient consensus — DS combination across batch
  resolve gradients with consensus { threshold: 0.85 }

  // Early stopping as evolve condition
  evolve weights until converged {
    metric: val_loss
    patience: 5
    max_iter: 50
  }

  on_epoch {
    emit metrics.loss
    emit metrics.accuracy
    emit metrics.epistemic_uncertainty   // unique to ChimeraLang
  }
}
```

### 1.6 Bayesian / Epistemic Constructs

```chimera
// Bayesian layer — weights are distributions, not points
layer fc1: BayesianDense(784 -> 128) {
  prior: Normal { mean: 0.0, std: 1.0 }
  posterior: Normal                     // learned mean + variance
}

// Uncertainty quantification at inference
val prediction = model.forward(x)
val epistemic = prediction.variance     // what the MODEL doesn't know
val aleatoric = prediction.entropy      // irreducible noise in the data

guard prediction against uncertainty {
  max_epistemic: 0.1    // refuse to predict if model is too uncertain
  fallback: abstain
}
```

### 1.7 Standard Library (Proposed)

```
chimera.tensor     — creation, indexing, slicing, math ops
chimera.nn         — Dense, Conv2d, LSTM, Transformer, Attention, Norm layers
chimera.optim      — Adam, SGD, AdamW, LAMB, Lion
chimera.loss       — CrossEntropy, MSE, KLDivergence, ELBOLoss, HuberLoss
chimera.data       — DataLoader, Dataset, transforms, mnist/cifar/imagenet loaders
chimera.metric     — Accuracy, F1, AUC, CalibrationError, ECE
chimera.io         — File read/write, JSON, CSV, Parquet, HDF5
chimera.device     — CPU/GPU/TPU placement, memory management
chimera.autograd   — grad(), jacobian(), hessian(), vmap(), jit()
chimera.dist       — DistributedTrainer, AllReduce, data/model parallelism
chimera.onnx       — Export to ONNX for deployment
```

### 1.8 I/O and General Language Primitives (Phase 1 prerequisite)

```chimera
// File I/O
val data = read_file("data/train.csv")
write_file("output/results.json", results)

// Error handling
val result: Result<Tensor<Float>[100], IOError> = load_tensor("weights.bin")
match result {
  | Ok(weights) => model.load(weights)
  | Err(e)      => emit "Failed: " + e.message
}

// Loops with index
for epoch in range(0, 100) {
  val loss = train_step(model, batch)
  if loss < 0.001 { break }
}

// Functions with generics
fn softmax<T>(x: Tensor<T>[?, n]) -> Tensor<T>[?, n] {
  val e = exp(x - max(x))
  return e / sum(e)
}
```

---

## 2. Path A — PyTorch/JAX Compiler

### 2.1 Architecture

```
ChimeraLang Source (.chimera)
        │
   Lexer / Parser           (extend existing — add new tokens/AST nodes)
        │
   Type Checker             (shape inference, gradient tracking, device placement)
        │
   IR Lowering              (CIR extended with TensorNode, GradNode, TrainNode)
        │
   Backend Selector         ──→ PyTorch Codegen
        │                   ──→ JAX Codegen  
        │                   ──→ Python fallback (existing VM)
        │
   Generated Python         (valid PyTorch/JAX code)
        │
   Python Runtime           (existing Python + torch/jax installed)
        │
   Output                   (trained model, ONNX export, reasoning trace)
```

### 2.2 Code Generation Example

**ChimeraLang input:**

```chimera
model Classifier {
  layer fc1: Dense(784 -> 128)
  layer act1: ReLU
  layer fc2: Dense(128 -> 10)
  uncertainty: epistemic

  forward(x: Tensor<Float>[?, 784]) -> Tensor<Float>[?, 10] {
    return fc2(act1(fc1(x)))
  }
}

train Classifier on mnist_train {
  optimizer: Adam { lr: 0.001 }
  loss: CrossEntropy
  epochs: 10
}
```

**Generated PyTorch output:**

```python
# AUTO-GENERATED BY CHIMERALANG COMPILER — DO NOT EDIT
import torch
import torch.nn as nn
from chimera_runtime import BeliefTracker, GuardLayer

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self._belief_tracker = BeliefTracker(mode="epistemic")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape: [?, 784] -> [?, 128] -> [?, 10]
        h = self.act1(self.fc1(x))
        out = self.fc2(h)
        self._belief_tracker.record(out)
        return out

model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in mnist_train:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
```

### 2.3 New AST Nodes Required

```
TensorType(dtype, shape)
ModelDecl(name, layers, uncertainty_mode, forward_fn)
LayerDecl(name, kind, in_dim, out_dim, config)
TrainStmt(model, dataset, optimizer, loss, epochs, guards)
ForwardFn(params, return_type, body)
BayesianLayerDecl(name, kind, prior, posterior)
ImportStmt(module, symbols)
FnDecl extended with generics
ResultType(ok_type, err_type)
BreakStmt
RangeExpr(start, end, step)
```

### 2.4 Type Checker Extensions

The existing `type_checker.py` needs:

1. **Shape inference engine** — propagate tensor shapes forward through layers, error on mismatches at compile time
2. **Gradient tracking** — mark which values are in the autograd graph
3. **Device placement** — warn on CPU↔GPU transfers in hot paths
4. **Distribution type checking** — `Normal` output cannot be assigned to `Categorical` without explicit conversion
5. **Epistemic mode checking** — `uncertainty: epistemic` requires Bayesian layers

### 2.5 `chimera_runtime` Python Package

A thin Python package that Path A generated code depends on:

```
chimera_runtime/
├── belief_tracker.py    — wraps BeliefState/BetaDist for PyTorch integration
├── guard_layer.py       — nn.Module that applies CIR guard checks
├── ds_loss.py           — DS combination applied to gradient tensors
├── temporal_decay.py    — TTL-based weight regularization
├── symbol_store.py      — persist learned architecture patterns
└── uncertainty.py       — epistemic/aleatoric decomposition utilities
```

### 2.6 File Structure (Path A additions)

```
chimera/
├── cir/                         (existing)
├── compiler/                    (NEW)
│   ├── __init__.py
│   ├── shape_checker.py         — shape inference + static checking
│   ├── gradient_tracker.py      — autograd graph analysis
│   ├── codegen_pytorch.py       — PyTorch code generator
│   ├── codegen_jax.py           — JAX code generator
│   └── ir_nodes.py              — TensorNode, GradNode, TrainNode, ModelNode
├── stdlib/                      (NEW)
│   ├── nn.chimera               — standard layers
│   ├── optim.chimera            — optimizers
│   ├── loss.chimera             — loss functions
│   └── data.chimera             — data utilities
└── runtime/                     (NEW — pip-installable)
    ├── belief_tracker.py
    ├── guard_layer.py
    └── uncertainty.py
```

### 2.7 CLI Extensions

```bash
# Compile to PyTorch Python
chimera compile model.chimera --backend=pytorch --out=model.py

# Compile and run directly
chimera run model.chimera --backend=pytorch

# Compile to JAX
chimera compile model.chimera --backend=jax --out=model.py

# Export trained model to ONNX
chimera export model.chimera --format=onnx --out=model.onnx

# Type-check only (shape inference)
chimera check model.chimera --shapes
```

### 2.8 Path A Milestones

| Milestone | Deliverable | Est. Time |
|---|---|---|
| **M1** | I/O, error handling, modules, for/range in existing VM | 3 weeks |
| **M2** | `TensorType` in lexer/parser/AST/type checker, shape inference | 4 weeks |
| **M3** | `ModelDecl`, `LayerDecl`, `ForwardFn` — PyTorch codegen for simple MLP | 4 weeks |
| **M4** | `TrainStmt` codegen — training loop with optimizer + loss | 3 weeks |
| **M5** | `chimera_runtime` package — BeliefTracker, GuardLayer wired in | 2 weeks |
| **M6** | BayesianDense layer — epistemic uncertainty output | 3 weeks |
| **M7** | JAX codegen backend | 3 weeks |
| **M8** | chimera.stdlib — nn, optim, loss, data modules | 4 weeks |
| **M9** | ONNX export | 2 weeks |
| **M10** | End-to-end demo: train MNIST, export, deploy | 2 weeks |
| **Total** | **~30 weeks / ~7 months** | |

---

## 3. Path B — Native LLVM Compiler

### 3.1 Architecture

```
ChimeraLang Source (.chimera)
        │
   Lexer / Parser           (same frontend as Path A)
        │
   HIR — High-level IR      (typed, shape-annotated, device-aware)
        │
   MIR — Mid-level IR       (SSA form, explicit memory, autograd graph built)
        │
   Autograd Pass            (differentiate MIR — forward + reverse mode)
        │
   Device Lowering          (CPU kernel selection / GPU kernel launch)
        │
   LLVM IR                  (standard LLVM, target-independent)
        │
   ┌───────────┬────────────┬──────────┐
   │           │            │          │
  x86-64    ARM/Apple M   NVPTX      Metal
  (CPU)      (CPU)         (CUDA)    (Apple GPU)
```

### 3.2 Three-Level IR

#### HIR (High-Level IR) — shape-aware, close to ChimeraLang

```
ModelNode {
  name: "Classifier"
  layers: [DenseLayer(784, 128), ReLULayer, DenseLayer(128, 10)]
  forward: FnHIR {
    params: [(x, Tensor<f32>[?, 784])]
    returns: Tensor<f32>[?, 10]
    body: [MatMul, Add, ReLU, MatMul, Add]
  }
}

TrainNode {
  model: ModelNode
  optimizer: AdamNode { lr: 0.001 }
  loss: CrossEntropyNode
  epochs: 10
  guards: [VarianceGuardNode { max_risk: 0.05 }]
}
```

#### MIR (Mid-level IR) — explicit memory, SSA, autograd graph

```
// Every tensor has an allocation site and lifetime
%w1 = alloc Tensor<f32>[784, 128] on heap
%b1 = alloc Tensor<f32>[128] on heap
%h1 = matmul(%x, %w1)           // forward op
%h1b = add(%h1, %b1)
%h1r = relu(%h1b)

// Autograd nodes alongside forward nodes
%grad_w1 = backward_matmul(%grad_h1, %x)   // generated by autograd pass
%grad_x  = backward_matmul(%grad_h1, %w1)
```

#### LLVM IR — standard

```llvm
; ChimeraLang tensor matmul lowered to LLVM
define void @chimera_matmul_f32(
  float* %A, float* %B, float* %C,
  i64 %M, i64 %N, i64 %K
) {
  ; loop nest with LLVM vectorization hints
  ; CUDA path: replaced by NVPTX call
}
```

### 3.3 Autograd Engine Design

The core of Path B. Two modes:

**Reverse-mode (backprop)** — for training neural networks
- Build computation graph during forward pass
- Walk backward, accumulate gradients
- Implemented as a MIR pass: `AutogradPass.differentiate(fn: FnMIR) -> FnMIR`

**Forward-mode (jvp)** — for gradient checking, scientific computing
- Dual numbers: each value carries `(primal, tangent)`
- Implemented as a HIR → HIR transformation

**Operator registry:**

```
chimera/autograd/ops/
├── matmul.rs      — ∂(AB)/∂A = grad·Bᵀ, ∂(AB)/∂B = Aᵀ·grad
├── relu.rs        — ∂ReLU(x)/∂x = 1 if x>0 else 0
├── softmax.rs     — Jacobian of softmax
├── conv2d.rs      — cross-correlation backward
├── layernorm.rs
└── attention.rs   — scaled dot-product attention backward
```

### 3.4 Tensor Runtime

A custom tensor library — the deepest engineering component of Path B:

```
chimera_tensor/
├── allocator/
│   ├── cpu_allocator.rs      — aligned malloc, memory pool
│   └── cuda_allocator.rs     — cudaMalloc, pinned memory
├── kernels/
│   ├── cpu/
│   │   ├── matmul_avx512.rs  — hand-tuned SIMD matmul
│   │   └── elementwise.rs    — fused add/relu/mul
│   └── cuda/
│       ├── matmul.cu         — cuBLAS wrapper + custom tiling
│       ├── elementwise.cu    — fused CUDA kernels
│       └── reduction.cu      — sum/mean/max
├── dispatch/
│   └── dispatcher.rs         — route op to right kernel based on dtype + device
└── shape/
    └── strided.rs             — strided tensor view system (like numpy)
```

**Compiler language for Path B: Rust** (memory safety without GC, excellent LLVM integration via `inkwell`).

### 3.5 Belief System Integration at Native Level

The key innovation in Path B: beliefs are **not a Python wrapper** but a native type:

```
// In MIR, every tensor can carry a BetaDist alongside its values
BelievedTensor {
  data: Tensor<f32>[shape]
  belief: BetaDist            // alpha, beta per-element or scalar
  grad: Option<Tensor<f32>>   // gradient (if in autograd graph)
  device: Device
}
```

This means:
- DS combination runs at native speed alongside tensor ops
- Guard checks are LLVM-inlined (zero Python overhead)
- Symbol emergence operates on the MIR graph, not the Python object graph

### 3.6 Compilation Pipeline Implementation

| Component | Language | Dependency |
|---|---|---|
| Frontend (lexer/parser) | Python (port to Rust in Phase 2) | existing |
| HIR/MIR | Rust | `serde` for serialization |
| LLVM IR gen | Rust | `inkwell` (LLVM 17 bindings) |
| CPU kernels | Rust + C (SIMD intrinsics) | `packed_simd` |
| CUDA kernels | CUDA C | CUDA Toolkit 12+ |
| Metal kernels | Metal Shading Language | Apple Metal SDK |
| Python bindings | PyO3 | expose as `import chimera_native` |
| Build system | Cargo + CMake | standard |

### 3.7 Path B Milestones

| Milestone | Deliverable | Est. Time |
|---|---|---|
| **M1** | Core tensor allocator + strided view + CPU matmul kernel | 6 weeks |
| **M2** | LLVM IR codegen for scalar + tensor ops | 6 weeks |
| **M3** | Reverse-mode autograd pass on MIR | 8 weeks |
| **M4** | Forward pass for MLP — CPU only | 4 weeks |
| **M5** | Training loop — SGD optimizer, cross-entropy loss | 4 weeks |
| **M6** | CUDA backend — matmul + elementwise on GPU | 8 weeks |
| **M7** | BelievedTensor native type — DS combination at MIR level | 6 weeks |
| **M8** | Native guard checks inlined into LLVM IR | 4 weeks |
| **M9** | Python bindings via PyO3 | 4 weeks |
| **M10** | Port frontend (lexer/parser) to Rust | 8 weeks |
| **M11** | Conv2d, LSTM, Attention ops + backward | 10 weeks |
| **M12** | Distributed training (NCCL integration) | 8 weeks |
| **M13** | End-to-end: train ResNet-50 on ImageNet | 4 weeks |
| **Total** | **~80 weeks / ~18 months** | |

---

## 4. Shared Innovations (Both Paths)

These features are unique to ChimeraLang regardless of implementation path:

### 4.1 Epistemic Loss Function

A new loss function that penalizes overconfident wrong predictions:

```
EpistemicLoss = CrossEntropy + λ × KL(posterior || prior)
```

Where the KL term regularizes the model's uncertainty estimates — forcing the model to maintain calibrated confidence. Variational inference made first-class.

### 4.2 Belief-Propagating Backprop

Standard backprop computes `∂Loss/∂weight`. ChimeraLang's extended backprop computes:

```
∂Loss/∂weight          — standard gradient
∂Uncertainty/∂weight   — how much does changing this weight affect model uncertainty?
```

The second term enables training toward lower epistemic uncertainty, not just lower loss.

### 4.3 Symbol-Emerged Architecture Search

The existing symbol emergence system extended to ML:

1. After each training run, extract reusable layer subgraphs (e.g., "Dense → BatchNorm → ReLU → Dropout" is a common pattern)
2. Assign fitness: `compression × reuse_frequency × validation_improvement`
3. High-fitness symbols become first-class named architectures in the symbol store
4. Future models can import them: `layer block1: symbol("ResBlock")`

This is **automatic neural architecture search** through the same mechanism that discovers reusable CIR belief patterns.

### 4.4 Temporal Model Decay

Models trained on time-sensitive data degrade gracefully:

```chimera
model NewsClassifier {
  layer fc1: Dense(768 -> 256)
  layer fc2: Dense(256 -> 10)
  ttl: 604800   // model beliefs decay after 7 days
  
  guard predictions against staleness {
    max_age: 604800
    fallback: retrain
  }
}
```

When a model's TTL expires, its output distributions decay toward uniform — the model becomes uncertain rather than confidently wrong.

### 4.5 Guard-Protected Training

Training loops that stop themselves when they detect pathological behavior:

```chimera
train Classifier on dataset {
  // Stop training if gradients are exploding
  guard gradients against explosion { max_norm: 10.0, action: clip }
  
  // Stop training if model becomes overconfident on training set
  guard train_predictions against overconfidence {
    max_confidence_gap: 0.3   // train_conf - val_conf < 0.3
    action: stop
  }
  
  // Detect catastrophic forgetting
  guard val_metrics against degradation {
    max_drop: 0.05
    action: restore_checkpoint
  }
}
```

---

## 5. Comparison: Before vs. After

| Capability | ChimeraLang Today | ChimeraLang ML (Path A) | ChimeraLang ML (Path B) |
|---|---|---|---|
| Belief system | ✅ Beta distributions | ✅ + tensor beliefs | ✅ native BelievedTensor |
| Symbol emergence | ✅ CIR subgraphs | ✅ + architecture patterns | ✅ + MIR-level patterns |
| General I/O | ❌ | ✅ | ✅ |
| Tensor ops | ❌ | ✅ via PyTorch/JAX | ✅ native |
| Neural network layers | ❌ | ✅ compiled to torch.nn | ✅ native kernels |
| GPU support | ❌ | ✅ via CUDA backend | ✅ CUDA + Metal |
| Autograd | ❌ | ✅ via torch.autograd | ✅ native MIR pass |
| Bayesian layers | ❌ | ✅ compiled to BNN libs | ✅ native BelievedTensor |
| Epistemic uncertainty | ✅ (belief system) | ✅ + model uncertainty | ✅ + at LLVM level |
| ONNX export | ❌ | ✅ | ✅ |
| Distributed training | ❌ | ✅ via torch.distributed | ✅ native NCCL |
| Performance | Python speed | PyTorch speed | Near-metal |

---

## 6. Recommended Execution Order

```
NOW (0-1 month)
└── Phase 0: Language foundations
    ├── Add I/O, error handling, modules, generics to existing VM
    └── Define final syntax for Tensor types, model/train blocks

NEXT (1-7 months)
└── Path A: PyTorch Compiler
    ├── Shape inference type checker
    ├── TensorType + ModelDecl + TrainStmt AST nodes
    ├── PyTorch codegen
    ├── chimera_runtime Python package
    ├── Bayesian layers + epistemic uncertainty output
    └── Demo: train MNIST with belief-guarded training loop

AFTER TRACTION (7-18 months)
└── Path B: Native compiler (in parallel with Path A maintenance)
    ├── Rust tensor runtime
    ├── LLVM IR codegen
    ├── Autograd MIR pass
    ├── CUDA kernels
    ├── BelievedTensor native type
    └── Python bindings via PyO3
```

---

## 7. Success Criteria

### Path A Success
- [ ] Train a 3-layer MLP on MNIST in ChimeraLang, get >98% accuracy
- [ ] Output includes epistemic uncertainty per prediction
- [ ] Guard violations during training are surfaced in reasoning trace
- [ ] Generated PyTorch code is readable and idiomatic
- [ ] Symbol store learns the MLP pattern after 3 training runs
- [ ] Export to ONNX, run in any ONNX runtime

### Path B Success
- [ ] Train ResNet-50 on ImageNet to >75% top-1 accuracy in ChimeraLang
- [ ] Training speed within 2× of native PyTorch on same hardware
- [ ] BelievedTensor uncertainty overhead < 5% vs standard tensor
- [ ] Zero Python in the hot path (pure native execution)
- [ ] Symbol emergence discovers ResBlock pattern automatically

---

*This spec should be reviewed before any implementation begins.*  
*Path A is approved for immediate planning. Path B begins after Path A ships.*
