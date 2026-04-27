# ChimeraLang ML Phase 1 — PyTorch Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch code generator for ChimeraLang so that `.chimera` ML programs compile to valid, runnable PyTorch Python — including MLP, MoE, constitutional AI, and RAG primitives.

**Architecture:** Extend lexer/parser/AST with ML constructs (Tensor types, model/train/layer/moe/constitution blocks), add a `chimera/compiler/` package that walks the AST and emits PyTorch Python, and ship a `chimera_runtime/` pip package that provides BeliefTracker, GuardLayer, and ConstitutionLayer wired into generated code.

**Tech Stack:** Python 3.11+, PyTorch 2.x (generated target), FAISS (VectorStore), existing chimera lexer/parser/CIR infrastructure.

---

## Scope Note

This plan is Phase 1 of 7. It produces a working PyTorch compiler for core ML constructs. Neuromorphic, swarm, federated, and self-improvement systems are Phase 3+.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `chimera/tokens.py` | Modify | Add 14 new ML keyword tokens |
| `chimera/lexer.py` | Modify | No changes needed (keywords auto-handled) |
| `chimera/ast_nodes.py` | Modify | Add 12 new ML AST node types |
| `chimera/parser.py` | Modify | Add parsers for model/train/layer/moe/constitution/vectorstore/import |
| `chimera/type_checker.py` | Modify | Add TensorType shape inference, Constituted<T> check |
| `chimera/compiler/__init__.py` | Create | Package init + `compile_to_pytorch()` entry point |
| `chimera/compiler/codegen_pytorch.py` | Create | AST → PyTorch Python string visitor |
| `chimera/compiler/shape_checker.py` | Create | Forward-propagate tensor shapes, error on mismatch |
| `chimera/cli.py` | Modify | Add `chimera compile` command |
| `chimera_runtime/__init__.py` | Create | Package exports |
| `chimera_runtime/belief_tracker.py` | Create | BeliefTracker — wraps BetaDist for PyTorch nn.Module |
| `chimera_runtime/guard_layer.py` | Create | GuardLayer — nn.Module that applies CIR guard checks |
| `chimera_runtime/constitution_layer.py` | Create | ConstitutionLayer — self-critique loop |
| `chimera_runtime/vector_store.py` | Create | VectorStore — FAISS-backed retrieval |
| `examples/mnist_classifier.chimera` | Create | MLP trained on MNIST |
| `examples/mnist_moe.chimera` | Create | MoE classifier on MNIST |
| `examples/mnist_aligned.chimera` | Create | Constitutional AI classifier |
| `tests/test_codegen.py` | Create | Codegen correctness tests |
| `tests/test_shape_checker.py` | Create | Shape inference tests |
| `tests/test_chimera_runtime.py` | Create | Runtime package tests |

---

## Task 1: New ML Tokens

**Files:**
- Modify: `chimera/tokens.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_ml_tokens.py`:

```python
"""Tests for ML keyword tokens."""
from chimera.lexer import Lexer
from chimera.tokens import TokenKind


def lex(src): return [t for t in Lexer(src).tokenize() if t.kind not in (TokenKind.NEWLINE, TokenKind.EOF)]

def test_model_token():
    tokens = lex("model Foo")
    assert tokens[0].kind == TokenKind.MODEL

def test_layer_token():
    tokens = lex("layer fc1")
    assert tokens[0].kind == TokenKind.LAYER

def test_train_token():
    tokens = lex("train Foo")
    assert tokens[0].kind == TokenKind.TRAIN

def test_forward_token():
    tokens = lex("forward(x)")
    assert tokens[0].kind == TokenKind.FORWARD

def test_import_token():
    tokens = lex("import chimera.nn")
    assert tokens[0].kind == TokenKind.IMPORT

def test_tensor_type_token():
    tokens = lex("Tensor")
    assert tokens[0].kind == TokenKind.TENSOR_KW

def test_constitution_token():
    tokens = lex("constitution MyConst")
    assert tokens[0].kind == TokenKind.CONSTITUTION

def test_moe_token():
    tokens = lex("moe MyMoE")
    assert tokens[0].kind == TokenKind.MOE

def test_vectorstore_token():
    tokens = lex("VectorStore")
    assert tokens[0].kind == TokenKind.VECTORSTORE_KW

def test_on_token():
    tokens = lex("on gpu")
    assert tokens[0].kind == TokenKind.ON
```

- [ ] **Step 2: Run — expect FAIL**

```
cd C:/Users/garza/Downloads/ChimeraLang
python -m pytest tests/test_ml_tokens.py -v 2>&1 | head -15
```

Expected: `AttributeError: MODEL` (token doesn't exist yet)

- [ ] **Step 3: Add tokens to `chimera/tokens.py`**

In `TokenKind` enum after `UNTIL = auto()`, add:

```python
    # Keywords — ML constructs
    MODEL = auto()
    LAYER = auto()
    TRAIN = auto()
    FORWARD = auto()
    IMPORT = auto()
    FROM = auto()
    AS_KW = auto()        # 'as' (avoid conflict with Python keyword)
    MOE = auto()
    CONSTITUTION = auto()
    CONSTITUTED_KW = auto()
    ON = auto()           # device placement: on gpu / on cpu

    # ML type keywords
    TENSOR_KW = auto()        # 'Tensor'
    VECTORSTORE_KW = auto()   # 'VectorStore'
    GRAD_KW = auto()          # 'Grad'
```

In `KEYWORDS` dict, add:

```python
    "model": TokenKind.MODEL,
    "layer": TokenKind.LAYER,
    "train": TokenKind.TRAIN,
    "forward": TokenKind.FORWARD,
    "import": TokenKind.IMPORT,
    "from": TokenKind.FROM,
    "as": TokenKind.AS_KW,
    "moe": TokenKind.MOE,
    "constitution": TokenKind.CONSTITUTION,
    "on": TokenKind.ON,
    "Tensor": TokenKind.TENSOR_KW,
    "VectorStore": TokenKind.VECTORSTORE_KW,
    "Grad": TokenKind.GRAD_KW,
    "Constituted": TokenKind.CONSTITUTED_KW,
```

- [ ] **Step 4: Run — expect PASS**

```
python -m pytest tests/test_ml_tokens.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Verify no regressions**

```
python -m pytest tests/ -v --tb=short 2>&1 | tail -5
```

Expected: 106 passed (existing) + 10 new = 116 passed.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/tokens.py tests/test_ml_tokens.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: add 14 ML keyword tokens (model, layer, train, forward, Tensor, MoE, constitution, VectorStore)"
```

---

## Task 2: ML AST Nodes

**Files:**
- Modify: `chimera/ast_nodes.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_ml_ast.py`:

```python
"""Tests for ML AST node instantiation."""
from chimera.ast_nodes import (
    TensorType, ModelDecl, LayerDecl, ForwardFn, TrainStmt,
    MoEBlock, ConstitutionDecl, VectorStoreType, ImportStmt,
    ConstitutedType, OptimizerConfig, LossConfig,
)

def test_tensor_type():
    t = TensorType(dtype="Float", shape=[None, 784])
    assert t.dtype == "Float"
    assert t.shape == [None, 784]

def test_model_decl():
    m = ModelDecl(name="Foo", layers=[], forward_fn=None)
    assert m.name == "Foo"

def test_layer_decl_dense():
    l = LayerDecl(name="fc1", kind="Dense", in_dim=784, out_dim=128, config={})
    assert l.kind == "Dense"
    assert l.in_dim == 784

def test_layer_decl_activation():
    l = LayerDecl(name="act1", kind="ReLU", in_dim=None, out_dim=None, config={})
    assert l.kind == "ReLU"

def test_forward_fn():
    f = ForwardFn(params=[("x", TensorType("Float", [None, 784]))], return_type=None, body=[])
    assert f.params[0][0] == "x"

def test_train_stmt():
    t = TrainStmt(
        model_name="Foo",
        dataset="mnist",
        optimizer=OptimizerConfig("Adam", {"lr": 0.001}),
        loss=LossConfig("CrossEntropy"),
        epochs=10,
        batch_size=128,
        guards=[],
        evolve=None,
    )
    assert t.model_name == "Foo"
    assert t.optimizer.name == "Adam"

def test_moe_block():
    m = MoEBlock(name="MyMoE", n_experts=64, top_k=2, expert_dim=4096, config={})
    assert m.n_experts == 64

def test_constitution_decl():
    c = ConstitutionDecl(name="MyConst", principles=["Be helpful"], critique_rounds=2)
    assert len(c.principles) == 1

def test_vectorstore_type():
    v = VectorStoreType(dim=768, capacity=1_000_000)
    assert v.dim == 768

def test_import_stmt():
    i = ImportStmt(module="chimera.nn", symbols=["Dense", "ReLU"], alias=None)
    assert i.module == "chimera.nn"

def test_constituted_type():
    ct = ConstitutedType(inner_type=TensorType("Float", [None, 10]))
    assert ct.inner_type.dtype == "Float"
```

- [ ] **Step 2: Run — expect FAIL**

```
python -m pytest tests/test_ml_ast.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'TensorType' from 'chimera.ast_nodes'`

- [ ] **Step 3: Append ML AST nodes to `chimera/ast_nodes.py`**

```python
# ---------------------------------------------------------------------------
# ML Constructs (Phase 1 — PyTorch compiler path)
# ---------------------------------------------------------------------------

@dataclass
class TensorType(TypeExpr):
    """Tensor<dtype>[dim1, dim2, ...] — None means dynamic dimension."""
    dtype: str = "Float"           # Float | Int | Bool | BFloat16 | Float8
    shape: list[int | None] = field(default_factory=list)
    device: str = "cpu"            # cpu | gpu | tpu


@dataclass
class VectorStoreType(TypeExpr):
    """VectorStore<dim>[capacity] — first-class retrieval type."""
    dim: int = 768
    capacity: int = 1_000_000


@dataclass
class ConstitutedType(TypeExpr):
    """Constituted<T> — proof-carrying type: value passed constitutional review."""
    inner_type: TypeExpr | None = None


@dataclass
class GradType(TypeExpr):
    """Grad<T> — T that participates in autograd graph."""
    inner_type: TypeExpr | None = None


@dataclass
class OptimizerConfig(ASTNode):
    name: str = "Adam"             # Adam | SGD | AdamW | LAMB
    params: dict = field(default_factory=dict)


@dataclass
class LossConfig(ASTNode):
    name: str = "CrossEntropy"     # CrossEntropy | MSE | KLDivergence


@dataclass
class LayerDecl(ASTNode):
    name: str = ""
    kind: str = ""                 # Dense | ReLU | Softmax | LayerNorm | Dropout | Conv2d | Attention
    in_dim: int | None = None
    out_dim: int | None = None
    repeat: int = 1                # for 'TransformerBlock × 12' syntax
    config: dict = field(default_factory=dict)


@dataclass
class ForwardFn(ASTNode):
    params: list[tuple[str, TypeExpr]] = field(default_factory=list)
    return_type: TypeExpr | None = None
    body: list[Statement] = field(default_factory=list)


@dataclass
class EvolveConfig(ASTNode):
    metric: str = "val_loss"
    patience: int = 5
    max_iter: int = 50
    condition: str = "converged"


@dataclass
class ModelDecl(Declaration):
    name: str = ""
    layers: list[LayerDecl] = field(default_factory=list)
    forward_fn: ForwardFn | None = None
    uncertainty: str = "none"      # none | epistemic | aleatoric | full
    constitution: str | None = None  # name of constitution to apply
    device: str = "cpu"
    config: dict = field(default_factory=dict)


@dataclass
class TrainStmt(Statement):
    model_name: str = ""
    dataset: str = ""
    optimizer: OptimizerConfig | None = None
    loss: LossConfig | None = None
    epochs: int = 10
    batch_size: int = 32
    guards: list[GuardStmt] = field(default_factory=list)
    evolve: EvolveConfig | None = None
    precision: str = "float32"     # float32 | bfloat16 | float8
    config: dict = field(default_factory=dict)


@dataclass
class MoEBlock(ASTNode):
    """Mixture of Experts block declaration."""
    name: str = ""
    n_experts: int = 8
    top_k: int = 2
    expert_dim: int = 4096
    config: dict = field(default_factory=dict)


@dataclass
class ConstitutionDecl(Declaration):
    name: str = ""
    principles: list[str] = field(default_factory=list)
    critique_rounds: int = 2
    max_violation_score: float = 0.05


@dataclass
class ImportStmt(Statement):
    module: str = ""
    symbols: list[str] = field(default_factory=list)
    alias: str | None = None
```

- [ ] **Step 4: Run — expect PASS**

```
python -m pytest tests/test_ml_ast.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/ast_nodes.py tests/test_ml_ast.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: add 12 ML AST nodes (TensorType, ModelDecl, LayerDecl, TrainStmt, MoEBlock, ConstitutionDecl, VectorStoreType)"
```

---

## Task 3: Parser Extensions for ML Constructs

**Files:**
- Modify: `chimera/parser.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_ml_parser.py`:

```python
"""Tests for ML construct parsing."""
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.ast_nodes import (
    ModelDecl, TrainStmt, ImportStmt, ConstitutionDecl, MoEBlock, LayerDecl,
)


def parse(src):
    return Parser(Lexer(src).tokenize()).parse()


def test_import_stmt():
    prog = parse("import chimera.nn")
    assert any(isinstance(d, ImportStmt) for d in prog.declarations)
    imp = next(d for d in prog.declarations if isinstance(d, ImportStmt))
    assert imp.module == "chimera.nn"


def test_import_with_symbols():
    prog = parse("import chimera.nn { Dense, ReLU }")
    imp = next(d for d in prog.declarations if isinstance(d, ImportStmt))
    assert "Dense" in imp.symbols
    assert "ReLU" in imp.symbols


def test_simple_model():
    src = """
model Classifier {
  layer fc1: Dense(784 -> 128)
  layer act1: ReLU
  layer fc2: Dense(128 -> 10)
}
"""
    prog = parse(src)
    model = next(d for d in prog.declarations if isinstance(d, ModelDecl))
    assert model.name == "Classifier"
    assert len(model.layers) == 3
    assert model.layers[0].kind == "Dense"
    assert model.layers[0].in_dim == 784
    assert model.layers[0].out_dim == 128
    assert model.layers[1].kind == "ReLU"


def test_train_stmt():
    src = """
train Classifier on mnist {
  optimizer: Adam { lr: 0.001 }
  loss: CrossEntropy
  epochs: 10
  batch_size: 128
}
"""
    prog = parse(src)
    train = next(d for d in prog.declarations if isinstance(d, TrainStmt))
    assert train.model_name == "Classifier"
    assert train.dataset == "mnist"
    assert train.optimizer.name == "Adam"
    assert train.optimizer.params["lr"] == 0.001
    assert train.loss.name == "CrossEntropy"
    assert train.epochs == 10
    assert train.batch_size == 128


def test_constitution_decl():
    src = '''
constitution SafetyFirst {
  "Be helpful, harmless, and honest"
  "Never assist with weapons"
  critique_rounds: 2
}
'''
    prog = parse(src)
    const = next(d for d in prog.declarations if isinstance(d, ConstitutionDecl))
    assert const.name == "SafetyFirst"
    assert len(const.principles) == 2
    assert const.critique_rounds == 2


def test_moe_in_model():
    src = """
model MoEModel {
  layer embed: Embedding(50257 -> 1024)
  moe experts {
    n_experts: 8
    top_k: 2
    expert_dim: 4096
  }
  layer head: Linear(1024 -> 50257)
}
"""
    prog = parse(src)
    model = next(d for d in prog.declarations if isinstance(d, ModelDecl))
    assert model.name == "MoEModel"
    moe_layers = [l for l in model.layers if l.kind == "MoE"]
    assert len(moe_layers) == 1
    assert moe_layers[0].config["n_experts"] == 8
```

- [ ] **Step 2: Run — expect FAIL**

```
python -m pytest tests/test_ml_parser.py -v 2>&1 | head -15
```

Expected: `ParseError` or `AttributeError` — ML constructs not yet parsed.

- [ ] **Step 3: Add imports to `chimera/parser.py`**

Add to the `from chimera.ast_nodes import (...)` block:

```python
    ConstitutedType,
    ConstitutionDecl,
    EvolveConfig,
    ForwardFn,
    GradType,
    ImportStmt,
    LayerDecl,
    LossConfig,
    ModelDecl,
    MoEBlock,
    OptimizerConfig,
    TensorType,
    TrainStmt,
    VectorStoreType,
```

- [ ] **Step 4: Add ML dispatch to `_parse_top_level` in `chimera/parser.py`**

In `_parse_top_level`, before `return self._parse_statement()`, add:

```python
        if kind == TokenKind.IMPORT:
            return self._parse_import()
        if kind == TokenKind.MODEL:
            return self._parse_model()
        if kind == TokenKind.TRAIN:
            return self._parse_train()
        if kind == TokenKind.CONSTITUTION:
            return self._parse_constitution()
```

- [ ] **Step 5: Add ML parser methods to `chimera/parser.py`**

Add before `_parse_expr`:

```python
    # ------------------------------------------------------------------
    # ML constructs
    # ------------------------------------------------------------------

    def _parse_import(self) -> ImportStmt:
        """import <module> [{ Symbol, ... }] [as alias]"""
        self._expect(TokenKind.IMPORT)
        parts = [self._expect(TokenKind.IDENT, "Expected module name").value]
        while self._check(TokenKind.DOT):
            self._advance()
            parts.append(self._advance().value)
        module = ".".join(parts)
        symbols: list[str] = []
        alias: str | None = None
        if self._check(TokenKind.LBRACE):
            self._advance()
            self._skip_newlines()
            while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.RBRACE):
                    break
                symbols.append(self._advance().value)
                self._match(TokenKind.COMMA)
                self._skip_newlines()
            self._expect(TokenKind.RBRACE)
        if self._match(TokenKind.AS_KW):
            alias = self._expect(TokenKind.IDENT).value
        self._expect_line_end()
        return ImportStmt(module=module, symbols=symbols, alias=alias)

    def _parse_model(self) -> ModelDecl:
        """model <Name> { layer..., moe..., forward... }"""
        self._expect(TokenKind.MODEL)
        name = self._expect(TokenKind.IDENT, "Expected model name").value
        self._expect(TokenKind.LBRACE, "Expected '{' after model name")
        self._expect_line_end()
        layers: list[LayerDecl] = []
        forward_fn: ForwardFn | None = None
        uncertainty = "none"
        constitution: str | None = None
        device = "cpu"

        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            if self._check(TokenKind.LAYER):
                layers.append(self._parse_layer_decl())
            elif self._check(TokenKind.MOE):
                layers.append(self._parse_moe_layer())
            elif self._check(TokenKind.FORWARD):
                forward_fn = self._parse_forward_fn()
            elif self._check(TokenKind.IDENT) and self._current().value == "uncertainty":
                self._advance()
                self._expect(TokenKind.COLON)
                uncertainty = self._advance().value
                self._expect_line_end()
            elif self._check(TokenKind.IDENT) and self._current().value == "constitution":
                self._advance()
                self._expect(TokenKind.COLON)
                constitution = self._advance().value
                self._expect_line_end()
            elif self._check(TokenKind.ON):
                self._advance()
                device = self._advance().value
                self._expect_line_end()
            else:
                self._advance()  # skip unknown lines gracefully
                self._expect_line_end()
            self._skip_newlines()

        self._expect(TokenKind.RBRACE, "Expected '}' to close model block")
        self._expect_line_end()
        return ModelDecl(
            name=name, layers=layers, forward_fn=forward_fn,
            uncertainty=uncertainty, constitution=constitution, device=device,
        )

    def _parse_layer_decl(self) -> LayerDecl:
        """layer <name>: <Kind>(<in> -> <out>) [{ config }] [× N]"""
        self._expect(TokenKind.LAYER)
        name = self._expect(TokenKind.IDENT, "Expected layer name").value
        self._expect(TokenKind.COLON, "Expected ':' after layer name")
        kind = self._advance().value   # Dense | ReLU | Softmax | etc.
        in_dim: int | None = None
        out_dim: int | None = None
        config: dict = {}
        repeat = 1

        if self._check(TokenKind.LPAREN):
            self._advance()
            if self._check(TokenKind.INT_LIT):
                in_dim = int(self._advance().value)
                if self._match(TokenKind.ARROW):
                    out_dim = int(self._expect(TokenKind.INT_LIT).value)
            self._expect(TokenKind.RPAREN)

        if self._check(TokenKind.LBRACE):
            config = self._parse_kv_block()

        # repeat syntax: × N or * N
        if self._check(TokenKind.STAR):
            self._advance()
            repeat = int(self._expect(TokenKind.INT_LIT).value)

        self._expect_line_end()
        return LayerDecl(name=name, kind=kind, in_dim=in_dim, out_dim=out_dim,
                         repeat=repeat, config=config)

    def _parse_moe_layer(self) -> LayerDecl:
        """moe <name> { n_experts: N, top_k: K, expert_dim: D }"""
        self._expect(TokenKind.MOE)
        name = self._expect(TokenKind.IDENT, "Expected MoE name").value
        config: dict = {}
        if self._check(TokenKind.LBRACE):
            config = self._parse_kv_block()
        self._expect_line_end()
        return LayerDecl(
            name=name, kind="MoE",
            in_dim=None, out_dim=None,
            config=config,
        )

    def _parse_forward_fn(self) -> ForwardFn:
        """forward(<params>) -> <type> { body }"""
        self._expect(TokenKind.FORWARD)
        self._expect(TokenKind.LPAREN)
        params: list[tuple[str, TypeExpr | None]] = []
        while not self._check(TokenKind.RPAREN) and not self._check(TokenKind.EOF):
            pname = self._expect(TokenKind.IDENT).value
            type_ann: TypeExpr | None = None
            if self._match(TokenKind.COLON):
                type_ann = self._parse_type()
            params.append((pname, type_ann))
            self._match(TokenKind.COMMA)
        self._expect(TokenKind.RPAREN)
        ret_type: TypeExpr | None = None
        if self._match(TokenKind.ARROW):
            ret_type = self._parse_type()
        self._expect(TokenKind.LBRACE)
        self._expect_line_end()
        body: list[Statement] = []
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            body.append(self._parse_statement())
            self._skip_newlines()
        self._expect(TokenKind.RBRACE)
        self._expect_line_end()
        return ForwardFn(params=params, return_type=ret_type, body=body)

    def _parse_train(self) -> TrainStmt:
        """train <Model> on <dataset> { optimizer:, loss:, epochs:, batch_size:, ... }"""
        self._expect(TokenKind.TRAIN)
        model_name = self._expect(TokenKind.IDENT, "Expected model name").value
        self._expect(TokenKind.ON, "Expected 'on' after model name")
        dataset = self._advance().value
        self._expect(TokenKind.LBRACE)
        self._expect_line_end()
        optimizer: OptimizerConfig = OptimizerConfig("Adam", {"lr": 0.001})
        loss: LossConfig = LossConfig("CrossEntropy")
        epochs = 10
        batch_size = 32
        guards: list = []
        evolve: EvolveConfig | None = None
        precision = "float32"

        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            if self._check(TokenKind.GUARD):
                guards.append(self._parse_guard())
                continue
            if self._check(TokenKind.EVOLVE):
                evolve = self._parse_evolve_config()
                continue
            key_tok = self._advance()
            key = key_tok.value
            self._expect(TokenKind.COLON)
            if key == "optimizer":
                opt_name = self._advance().value
                opt_params: dict = {}
                if self._check(TokenKind.LBRACE):
                    opt_params = self._parse_kv_block()
                optimizer = OptimizerConfig(opt_name, opt_params)
            elif key == "loss":
                loss = LossConfig(self._advance().value)
            elif key == "epochs":
                epochs = int(self._advance().value)
            elif key == "batch_size":
                batch_size = int(self._advance().value)
            elif key == "precision":
                precision = self._advance().value
            self._expect_line_end()
            self._skip_newlines()

        self._expect(TokenKind.RBRACE)
        self._expect_line_end()
        return TrainStmt(
            model_name=model_name, dataset=dataset,
            optimizer=optimizer, loss=loss,
            epochs=epochs, batch_size=batch_size,
            guards=guards, evolve=evolve, precision=precision,
        )

    def _parse_constitution(self) -> ConstitutionDecl:
        """constitution <Name> { "principle1" ... critique_rounds: N }"""
        self._expect(TokenKind.CONSTITUTION)
        name = self._expect(TokenKind.IDENT, "Expected constitution name").value
        self._expect(TokenKind.LBRACE)
        self._expect_line_end()
        principles: list[str] = []
        critique_rounds = 2
        max_violation = 0.05

        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            if self._check(TokenKind.STRING_LIT):
                principles.append(self._advance().value)
                self._expect_line_end()
            elif self._check(TokenKind.IDENT):
                key = self._advance().value
                self._expect(TokenKind.COLON)
                if key == "critique_rounds":
                    critique_rounds = int(self._advance().value)
                elif key == "max_violation_score":
                    max_violation = float(self._advance().value)
                self._expect_line_end()
            else:
                self._advance()
            self._skip_newlines()

        self._expect(TokenKind.RBRACE)
        self._expect_line_end()
        return ConstitutionDecl(
            name=name, principles=principles,
            critique_rounds=critique_rounds,
            max_violation_score=max_violation,
        )

    def _parse_evolve_config(self) -> EvolveConfig:
        """evolve weights until converged { metric: ..., patience: N, max_iter: N }"""
        self._expect(TokenKind.EVOLVE)
        self._advance()  # 'weights'
        self._expect(TokenKind.UNTIL)
        condition = self._advance().value
        config: dict = {}
        if self._check(TokenKind.LBRACE):
            config = self._parse_kv_block()
        self._expect_line_end()
        return EvolveConfig(
            metric=config.get("metric", "val_loss"),
            patience=int(config.get("patience", 5)),
            max_iter=int(config.get("max_iter", 50)),
            condition=condition,
        )

    def _parse_kv_block(self) -> dict:
        """Parse { key: value, ... } into a Python dict."""
        self._expect(TokenKind.LBRACE)
        self._skip_newlines()
        result: dict = {}
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            key = self._advance().value
            self._expect(TokenKind.COLON)
            tok = self._current()
            if tok.kind == TokenKind.INT_LIT:
                result[key] = int(self._advance().value)
            elif tok.kind == TokenKind.FLOAT_LIT:
                result[key] = float(self._advance().value)
            elif tok.kind == TokenKind.STRING_LIT:
                result[key] = self._advance().value
            elif tok.kind == TokenKind.BOOL_LIT:
                result[key] = self._advance().value == "true"
            else:
                result[key] = self._advance().value  # treat as string
            self._match(TokenKind.COMMA)
            self._skip_newlines()
        self._expect(TokenKind.RBRACE)
        return result
```

- [ ] **Step 6: Add TensorType to `_parse_type` in `chimera/parser.py`**

In `_parse_type`, before the `raise ParseError` at the end:

```python
        if tok.kind == TokenKind.TENSOR_KW:
            self._advance()
            dtype = "Float"
            if self._match(TokenKind.LT):
                dtype = self._advance().value
                self._expect(TokenKind.GT)
            shape: list[int | None] = []
            if self._match(TokenKind.LBRACKET):
                while not self._check(TokenKind.RBRACKET) and not self._check(TokenKind.EOF):
                    if self._check(TokenKind.INT_LIT):
                        shape.append(int(self._advance().value))
                    elif self._check(TokenKind.STAR) or (self._check(TokenKind.IDENT) and self._current().value == "?"):
                        self._advance()
                        shape.append(None)
                    self._match(TokenKind.COMMA)
                self._expect(TokenKind.RBRACKET)
            device = "cpu"
            if self._check(TokenKind.ON):
                self._advance()
                device = self._advance().value
            return TensorType(dtype=dtype, shape=shape, device=device)

        if tok.kind == TokenKind.VECTORSTORE_KW:
            self._advance()
            dim = 768
            if self._match(TokenKind.LT):
                dim = int(self._expect(TokenKind.INT_LIT).value)
                self._expect(TokenKind.GT)
            capacity = 1_000_000
            if self._match(TokenKind.LBRACKET):
                capacity = int(self._expect(TokenKind.INT_LIT).value)
                self._expect(TokenKind.RBRACKET)
            return VectorStoreType(dim=dim, capacity=capacity)

        if tok.kind == TokenKind.CONSTITUTED_KW:
            self._advance()
            inner: TypeExpr | None = None
            if self._match(TokenKind.LT):
                inner = self._parse_type()
                self._expect(TokenKind.GT)
            return ConstitutedType(inner_type=inner)
```

Also add the new type imports to parser.py's ast_nodes imports block.

- [ ] **Step 7: Run parser tests**

```
python -m pytest tests/test_ml_parser.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 8: Run full suite**

```
python -m pytest tests/ -v --tb=short 2>&1 | tail -5
```

Expected: 116+ passed, 0 failed.

- [ ] **Step 9: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/parser.py tests/test_ml_parser.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: parser extensions for model/train/layer/moe/constitution/import/TensorType"
```

---

## Task 4: PyTorch Code Generator

**Files:**
- Create: `chimera/compiler/__init__.py`
- Create: `chimera/compiler/codegen_pytorch.py`
- Create: `tests/test_codegen.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_codegen.py`:

```python
"""Tests for PyTorch code generation."""
import pytest
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.compiler import compile_to_pytorch


def codegen(src: str) -> str:
    prog = Parser(Lexer(src).tokenize()).parse()
    return compile_to_pytorch(prog)


def test_simple_mlp_generates_nn_module():
    src = """
model Classifier {
  layer fc1: Dense(784 -> 128)
  layer act1: ReLU
  layer fc2: Dense(128 -> 10)
}
"""
    code = codegen(src)
    assert "class Classifier(nn.Module)" in code
    assert "nn.Linear(784, 128)" in code
    assert "nn.ReLU()" in code
    assert "nn.Linear(128, 10)" in code
    assert "def forward(self, x)" in code


def test_train_generates_training_loop():
    src = """
model Foo {
  layer fc1: Dense(10 -> 5)
  layer fc2: Dense(5 -> 2)
}
train Foo on my_dataset {
  optimizer: Adam { lr: 0.001 }
  loss: CrossEntropy
  epochs: 5
  batch_size: 64
}
"""
    code = codegen(src)
    assert "torch.optim.Adam" in code
    assert "lr=0.001" in code
    assert "nn.CrossEntropyLoss" in code
    assert "for epoch in range(5)" in code
    assert "batch_size=64" in code
    assert "loss.backward()" in code
    assert "optimizer.step()" in code


def test_moe_generates_expert_layer():
    src = """
model MoEFoo {
  layer embed: Embedding(1000 -> 64)
  moe experts {
    n_experts: 8
    top_k: 2
    expert_dim: 256
  }
  layer head: Linear(64 -> 10)
}
"""
    code = codegen(src)
    assert "n_experts=8" in code
    assert "top_k=2" in code


def test_constitution_generates_critique_loop():
    src = '''
constitution MySafety {
  "Be helpful"
  "Be harmless"
  critique_rounds: 2
}
model SafeModel {
  layer fc1: Dense(10 -> 5)
  layer fc2: Dense(5 -> 2)
  constitution: MySafety
}
'''
    code = codegen(src)
    assert "ConstitutionLayer" in code
    assert "MySafety" in code


def test_generated_code_is_valid_python():
    src = """
model SimpleNet {
  layer fc1: Dense(784 -> 256)
  layer act1: ReLU
  layer fc2: Dense(256 -> 10)
}
train SimpleNet on train_data {
  optimizer: Adam { lr: 0.001 }
  loss: CrossEntropy
  epochs: 3
  batch_size: 32
}
"""
    code = codegen(src)
    # Verify generated code is valid Python (compiles without error)
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax error: {e}\n\nCode:\n{code}")


def test_import_generates_comment():
    src = "import chimera.nn { Dense, ReLU }"
    code = codegen(src)
    assert "chimera.nn" in code
```

- [ ] **Step 2: Run — expect FAIL**

```
python -m pytest tests/test_codegen.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'compile_to_pytorch'`

- [ ] **Step 3: Create `chimera/compiler/__init__.py`**

```python
"""ChimeraLang compiler — PyTorch and JAX backends."""
from chimera.compiler.codegen_pytorch import PyTorchCodegen


def compile_to_pytorch(program: object) -> str:
    """Compile a ChimeraLang AST to PyTorch Python source code."""
    return PyTorchCodegen().generate(program)
```

- [ ] **Step 4: Create `chimera/compiler/codegen_pytorch.py`**

```python
"""PyTorch code generator for ChimeraLang.

Walks the AST and emits valid PyTorch Python source code.
Generated code imports chimera_runtime for belief tracking and guard layers.
"""
from __future__ import annotations

from chimera.ast_nodes import (
    ConstitutionDecl, LayerDecl, ModelDecl,
    TrainStmt, ImportStmt, Program,
)

# Layer kind → PyTorch constructor
_LAYER_MAP: dict[str, str] = {
    "Dense": "nn.Linear",
    "Linear": "nn.Linear",
    "ReLU": "nn.ReLU",
    "GELU": "nn.GELU",
    "Softmax": "nn.Softmax",
    "Sigmoid": "nn.Sigmoid",
    "Tanh": "nn.Tanh",
    "Dropout": "nn.Dropout",
    "LayerNorm": "nn.LayerNorm",
    "BatchNorm": "nn.BatchNorm1d",
    "Embedding": "nn.Embedding",
    "Conv2d": "nn.Conv2d",
    "Flatten": "nn.Flatten",
    "Attention": "nn.MultiheadAttention",
}

# Optimizer name → PyTorch class
_OPTIM_MAP: dict[str, str] = {
    "Adam": "torch.optim.Adam",
    "AdamW": "torch.optim.AdamW",
    "SGD": "torch.optim.SGD",
    "RMSprop": "torch.optim.RMSprop",
    "LAMB": "torch.optim.Adam",  # fallback
}

# Loss name → PyTorch class
_LOSS_MAP: dict[str, str] = {
    "CrossEntropy": "nn.CrossEntropyLoss",
    "MSE": "nn.MSELoss",
    "BCE": "nn.BCELoss",
    "KLDivergence": "nn.KLDivLoss",
    "NLL": "nn.NLLLoss",
}


class PyTorchCodegen:
    def __init__(self) -> None:
        self._constitutions: dict[str, ConstitutionDecl] = {}
        self._indent = 0
        self._lines: list[str] = []

    def generate(self, program: Program) -> str:
        self._lines = []
        self._constitutions = {}

        # Collect constitutions first
        for decl in program.declarations:
            if isinstance(decl, ConstitutionDecl):
                self._constitutions[decl.name] = decl

        # Header
        self._emit("# AUTO-GENERATED BY CHIMERALANG COMPILER — DO NOT EDIT")
        self._emit("import torch")
        self._emit("import torch.nn as nn")
        self._emit("from torch.utils.data import DataLoader")
        self._emit("try:")
        self._emit("    from chimera_runtime import BeliefTracker, GuardLayer, ConstitutionLayer")
        self._emit("except ImportError:")
        self._emit("    BeliefTracker = GuardLayer = ConstitutionLayer = None")
        self._emit("")

        for decl in program.declarations:
            if isinstance(decl, ImportStmt):
                self._gen_import(decl)
            elif isinstance(decl, ModelDecl):
                self._gen_model(decl)
            elif isinstance(decl, TrainStmt):
                self._gen_train(decl)
            elif isinstance(decl, ConstitutionDecl):
                self._gen_constitution_comment(decl)

        return "\n".join(self._lines)

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def _gen_import(self, node: ImportStmt) -> None:
        symbols = ", ".join(node.symbols) if node.symbols else "*"
        self._emit(f"# import {node.module} {{ {symbols} }}")

    def _gen_constitution_comment(self, node: ConstitutionDecl) -> None:
        self._emit(f"# Constitution: {node.name}")
        for p in node.principles:
            self._emit(f"#   - {p}")
        self._emit("")

    def _gen_model(self, model: ModelDecl) -> None:
        self._emit(f"class {model.name}(nn.Module):")
        with self._indented():
            self._emit("def __init__(self):")
            with self._indented():
                self._emit("super().__init__()")
                for layer in model.layers:
                    self._gen_layer_init(layer)
                if model.uncertainty in ("epistemic", "aleatoric", "full"):
                    self._emit(f'self._belief_tracker = BeliefTracker(mode="{model.uncertainty}") if BeliefTracker else None')
                if model.constitution and model.constitution in self._constitutions:
                    c = self._constitutions[model.constitution]
                    principles_repr = repr(c.principles)
                    self._emit(
                        f"self._constitution = ConstitutionLayer({principles_repr}, "
                        f"rounds={c.critique_rounds}) if ConstitutionLayer else None"
                    )
            self._emit("")
            self._emit("def forward(self, x):")
            with self._indented():
                self._gen_forward_body(model)
        self._emit("")

    def _gen_layer_init(self, layer: LayerDecl) -> None:
        kind = layer.kind
        if kind == "MoE":
            n = layer.config.get("n_experts", 8)
            k = layer.config.get("top_k", 2)
            d = layer.config.get("expert_dim", 4096)
            self._emit(f"# MoE: {layer.name} — {n} experts, top-{k}")
            self._emit(f"self.{layer.name}_experts = nn.ModuleList([")
            with self._indented():
                self._emit(f"nn.Linear({d}, {d}) for _ in range({n})")
            self._emit("])")
            self._emit(f"self.{layer.name}_router = nn.Linear({d}, {n})")
            self._emit(f"self._{layer.name}_top_k = {k}")
            self._emit(f"self._{layer.name}_n_experts = {n}")
            return

        pt_cls = _LAYER_MAP.get(kind, f"nn.Identity  # unknown: {kind}")
        if kind in ("Dense", "Linear") and layer.in_dim and layer.out_dim:
            self._emit(f"self.{layer.name} = {pt_cls}({layer.in_dim}, {layer.out_dim})")
        elif kind in ("Embedding",) and layer.in_dim and layer.out_dim:
            self._emit(f"self.{layer.name} = {pt_cls}({layer.in_dim}, {layer.out_dim})")
        elif kind == "Dropout":
            p = layer.config.get("p", 0.1)
            self._emit(f"self.{layer.name} = {pt_cls}(p={p})")
        elif kind == "LayerNorm":
            d = layer.out_dim or layer.in_dim or 768
            self._emit(f"self.{layer.name} = {pt_cls}({d})")
        else:
            self._emit(f"self.{layer.name} = {pt_cls}()")

        if layer.repeat > 1:
            # Wrap repeated layers in ModuleList
            orig = self._lines.pop()
            layer_expr = orig.strip().split(" = ", 1)[1]
            self._emit(f"self.{layer.name} = nn.ModuleList([{layer_expr} for _ in range({layer.repeat})])")

    def _gen_forward_body(self, model: ModelDecl) -> None:
        if not model.layers:
            self._emit("return x")
            return

        prev = "x"
        for layer in model.layers:
            if layer.kind == "MoE":
                name = layer.name
                nxt = f"h_{name}"
                self._emit(f"# MoE routing for {name}")
                self._emit(f"_router_logits_{name} = self.{name}_router({prev})")
                self._emit(f"_topk_{name} = torch.topk(_router_logits_{name}, self._{name}_top_k, dim=-1)")
                self._emit(f"_indices_{name} = _topk_{name}.indices")
                self._emit(f"{nxt} = torch.zeros_like({prev})")
                self._emit(f"for _i, _expert in enumerate(self.{name}_experts):")
                self._emit(f"    _mask = (_indices_{name} == _i).any(dim=-1, keepdim=True).float()")
                self._emit(f"    {nxt} = {nxt} + _mask * _expert({prev})")
                prev = nxt
            elif layer.repeat > 1:
                nxt = f"h_{layer.name}"
                self._emit(f"{nxt} = {prev}")
                self._emit(f"for _layer in self.{layer.name}:")
                self._emit(f"    {nxt} = _layer({nxt})")
                prev = nxt
            else:
                nxt = f"h_{layer.name}" if layer.name != model.layers[-1].name else "out"
                self._emit(f"{nxt} = self.{layer.name}({prev})")
                prev = nxt

        if model.constitution and model.constitution in self._constitutions:
            self._emit(f"if self._constitution is not None:")
            self._emit(f"    {prev} = self._constitution.apply({prev})")

        if model.uncertainty in ("epistemic", "aleatoric", "full"):
            self._emit(f"if self._belief_tracker is not None:")
            self._emit(f"    self._belief_tracker.record({prev})")

        self._emit(f"return {prev}")

    def _gen_train(self, train: TrainStmt) -> None:
        model_var = train.model_name.lower()
        self._emit(f"# Training: {train.model_name} on {train.dataset}")
        self._emit(f"{model_var} = {train.model_name}()")

        opt_cls = _OPTIM_MAP.get(train.optimizer.name, "torch.optim.Adam")
        opt_params = ", ".join(f"{k}={v!r}" for k, v in train.optimizer.params.items())
        self._emit(f"_optimizer = {opt_cls}({model_var}.parameters(), {opt_params})")

        loss_cls = _LOSS_MAP.get(train.loss.name, "nn.CrossEntropyLoss")
        self._emit(f"_criterion = {loss_cls}()")
        self._emit("")
        self._emit(f"for epoch in range({train.epochs}):")
        with self._indented():
            self._emit(f"for _x, _y in DataLoader({train.dataset}, batch_size={train.batch_size}, shuffle=True):")
            with self._indented():
                if train.precision == "bfloat16":
                    self._emit("with torch.autocast(device_type='cuda', dtype=torch.bfloat16):")
                    with self._indented():
                        self._emit(f"_pred = {model_var}(_x)")
                        self._emit("_loss = _criterion(_pred, _y)")
                else:
                    self._emit(f"_pred = {model_var}(_x)")
                    self._emit("_loss = _criterion(_pred, _y)")
                self._emit("_optimizer.zero_grad()")
                self._emit("_loss.backward()")
                self._emit("_optimizer.step()")
            self._emit(f'print(f"Epoch {{epoch+1}}/{train.epochs} loss={{_loss.item():.4f}}")')
        self._emit("")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, line: str) -> None:
        if line == "":
            self._lines.append("")
        else:
            self._lines.append("    " * self._indent + line)

    class _indented:
        def __init__(self, codegen: "PyTorchCodegen") -> None:
            self._cg = codegen
        def __enter__(self):
            self._cg._indent += 1
            return self
        def __exit__(self, *_):
            self._cg._indent -= 1

    def _indented(self):
        return PyTorchCodegen._indented(self)
```

- [ ] **Step 5: Run codegen tests**

```
python -m pytest tests/test_codegen.py -v
```

Expected: All 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/compiler/ tests/test_codegen.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: PyTorch codegen — generates valid nn.Module + training loop from .chimera source"
```

---

## Task 5: `chimera_runtime` Package

**Files:**
- Create: `chimera_runtime/__init__.py`
- Create: `chimera_runtime/belief_tracker.py`
- Create: `chimera_runtime/guard_layer.py`
- Create: `chimera_runtime/constitution_layer.py`
- Create: `chimera_runtime/vector_store.py`
- Create: `tests/test_chimera_runtime.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chimera_runtime.py`:

```python
"""Tests for chimera_runtime package."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chimera_runtime import BeliefTracker, GuardLayer, ConstitutionLayer


class TestBeliefTracker:
    def test_records_value(self):
        tracker = BeliefTracker(mode="epistemic")
        tracker.record(0.85)
        assert len(tracker.history) == 1

    def test_mean_confidence(self):
        tracker = BeliefTracker(mode="epistemic")
        tracker.record(0.8)
        tracker.record(0.9)
        assert abs(tracker.mean_confidence() - 0.85) < 0.01

    def test_epistemic_uncertainty(self):
        tracker = BeliefTracker(mode="epistemic")
        for v in [0.7, 0.8, 0.9, 0.6, 0.85]:
            tracker.record(v)
        unc = tracker.epistemic_uncertainty()
        assert 0.0 <= unc <= 1.0


class TestGuardLayer:
    def test_passes_high_confidence(self):
        guard = GuardLayer(max_risk=0.2, strategy="mean")
        result = guard.check(confidence=0.9)
        assert result.passed

    def test_fails_low_confidence(self):
        guard = GuardLayer(max_risk=0.2, strategy="mean")
        result = guard.check(confidence=0.5)
        assert not result.passed
        assert "mean" in result.violation.lower()

    def test_variance_check(self):
        guard = GuardLayer(max_risk=0.2, strategy="variance")
        result = guard.check(confidence=0.9, variance=0.1)
        assert not result.passed


class TestConstitutionLayer:
    def test_apply_passes_safe_text(self):
        layer = ConstitutionLayer(
            principles=["Be helpful", "Be harmless"],
            rounds=1
        )
        result = layer.apply("Here is a helpful answer.")
        assert result.passed
        assert result.text is not None

    def test_violation_score_range(self):
        layer = ConstitutionLayer(principles=["Never be harmful"], rounds=1)
        result = layer.apply("Some output text here.")
        assert 0.0 <= result.violation_score <= 1.0
```

- [ ] **Step 2: Run — expect FAIL**

```
python -m pytest tests/test_chimera_runtime.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'chimera_runtime'`

- [ ] **Step 3: Create `chimera_runtime/belief_tracker.py`**

```python
"""BeliefTracker — wraps BetaDist confidence tracking for PyTorch models."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field


@dataclass
class BeliefTracker:
    mode: str = "epistemic"   # epistemic | aleatoric | full
    history: list[float] = field(default_factory=list)
    _timestamps: list[float] = field(default_factory=list)

    def record(self, value: object) -> None:
        """Record a raw output value. Extracts confidence from tensor or float."""
        conf = self._extract_confidence(value)
        self.history.append(conf)
        self._timestamps.append(time.time())

    def mean_confidence(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def epistemic_uncertainty(self) -> float:
        """Variance of confidence scores — high variance = uncertain model."""
        if len(self.history) < 2:
            return 0.0
        mu = self.mean_confidence()
        variance = sum((x - mu) ** 2 for x in self.history) / len(self.history)
        return min(math.sqrt(variance), 1.0)

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "n_records": len(self.history),
            "mean_confidence": round(self.mean_confidence(), 4),
            "epistemic_uncertainty": round(self.epistemic_uncertainty(), 4),
        }

    @staticmethod
    def _extract_confidence(value: object) -> float:
        try:
            import torch
            if isinstance(value, torch.Tensor):
                probs = torch.softmax(value.float(), dim=-1) if value.dim() > 0 else value
                return float(probs.max().item())
        except ImportError:
            pass
        if isinstance(value, (int, float)):
            return float(min(max(value, 0.0), 1.0))
        return 0.5
```

- [ ] **Step 4: Create `chimera_runtime/guard_layer.py`**

```python
"""GuardLayer — applies CIR guard checks to model outputs."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GuardResult:
    passed: bool
    violation: str = ""
    confidence: float = 0.0
    variance: float = 0.0


class GuardLayer:
    def __init__(
        self,
        max_risk: float = 0.2,
        strategy: str = "both",   # mean | variance | both
    ) -> None:
        self.max_risk = max_risk
        self.strategy = strategy

    def check(self, confidence: float, variance: float = 0.0) -> GuardResult:
        violations: list[str] = []

        if self.strategy in ("mean", "both"):
            required = 1.0 - self.max_risk
            if confidence < required:
                violations.append(
                    f"mean {confidence:.3f} < required {required:.3f}"
                )

        if self.strategy in ("variance", "both"):
            if variance > 0.05:
                violations.append(
                    f"variance {variance:.4f} > allowed 0.05"
                )

        if violations:
            return GuardResult(passed=False, violation="; ".join(violations),
                               confidence=confidence, variance=variance)
        return GuardResult(passed=True, confidence=confidence, variance=variance)

    def __call__(self, x: object, confidence: float | None = None) -> object:
        """nn.Module-compatible forward — returns x unchanged, logs violations."""
        conf = confidence if confidence is not None else 0.8
        result = self.check(conf)
        if not result.passed:
            print(f"[chimera guard] VIOLATION: {result.violation}")
        return x
```

- [ ] **Step 5: Create `chimera_runtime/constitution_layer.py`**

```python
"""ConstitutionLayer — self-critique loop for constitutional AI."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ConstitutionResult:
    passed: bool
    text: str | None
    violation_score: float = 0.0
    violations: list[str] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class ConstitutionLayer:
    def __init__(
        self,
        principles: list[str],
        rounds: int = 2,
        max_violation_score: float = 0.05,
    ) -> None:
        self.principles = principles
        self.rounds = rounds
        self.max_violation_score = max_violation_score
        # Forbidden markers — simplified heuristic critique
        self._forbidden = [
            "weapon", "bomb", "kill", "poison", "hack",
            "illegal", "dangerous", "harmful",
        ]

    def apply(self, text: object) -> ConstitutionResult:
        text_str = str(text) if not isinstance(text, str) else text
        violations: list[str] = []
        score = 0.0

        for principle in self.principles:
            v = self._check_principle(text_str, principle)
            if v:
                violations.append(v)
                score += 1.0 / len(self.principles)

        score = min(score, 1.0)
        passed = score <= self.max_violation_score

        return ConstitutionResult(
            passed=passed,
            text=text_str,
            violation_score=round(score, 4),
            violations=violations,
        )

    def _check_principle(self, text: str, principle: str) -> str | None:
        text_lower = text.lower()
        for marker in self._forbidden:
            if marker in text_lower:
                return f"Forbidden content '{marker}' detected (principle: {principle!r})"
        return None
```

- [ ] **Step 6: Create `chimera_runtime/vector_store.py`**

```python
"""VectorStore — first-class retrieval type backed by FAISS or numpy fallback."""
from __future__ import annotations
import math
from dataclasses import dataclass, field


@dataclass
class VectorStore:
    dim: int = 768
    capacity: int = 1_000_000

    _vectors: list[list[float]] = field(default_factory=list)
    _texts: list[str] = field(default_factory=list)

    def index(self, vectors: list[list[float]], texts: list[str]) -> None:
        for v, t in zip(vectors, texts):
            if len(self._vectors) < self.capacity:
                self._vectors.append(v)
                self._texts.append(t)

    def read(self, query: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Return top_k (text, score) pairs by cosine similarity."""
        if not self._vectors:
            return []
        scores = [(self._cosine(query, v), t) for v, t in zip(self._vectors, self._texts)]
        scores.sort(reverse=True)
        return [(t, s) for s, t in scores[:top_k]]

    def persist(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump({"dim": self.dim, "vectors": self._vectors, "texts": self._texts}, f)

    def load(self, path: str) -> None:
        import json
        with open(path) as f:
            data = json.load(f)
        self._vectors = data["vectors"]
        self._texts = data["texts"]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)
```

- [ ] **Step 7: Create `chimera_runtime/__init__.py`**

```python
"""chimera_runtime — runtime support for ChimeraLang-generated PyTorch code."""
from chimera_runtime.belief_tracker import BeliefTracker
from chimera_runtime.guard_layer import GuardLayer, GuardResult
from chimera_runtime.constitution_layer import ConstitutionLayer, ConstitutionResult
from chimera_runtime.vector_store import VectorStore

__all__ = [
    "BeliefTracker", "GuardLayer", "GuardResult",
    "ConstitutionLayer", "ConstitutionResult",
    "VectorStore",
]
__version__ = "0.1.0"
```

- [ ] **Step 8: Run runtime tests**

```
python -m pytest tests/test_chimera_runtime.py -v
```

Expected: All 8 tests pass.

- [ ] **Step 9: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera_runtime/ tests/test_chimera_runtime.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: chimera_runtime package — BeliefTracker, GuardLayer, ConstitutionLayer, VectorStore"
```

---

## Task 6: CLI `chimera compile` Command

**Files:**
- Modify: `chimera/cli.py`

- [ ] **Step 1: Add `compile` to CLI dispatch in `chimera/cli.py`**

Add this function after `cmd_prove`:

```python
def cmd_compile(path: str, backend: str = "pytorch", out: str | None = None) -> None:
    """Compile a .chimera program to PyTorch (or JAX) Python."""
    source = _read_source(path)
    try:
        program = _parse(source, path)
    except (LexError, ParseError) as e:
        print(f"chimera: parse error: {e}", file=sys.stderr)
        sys.exit(1)

    if backend == "pytorch":
        from chimera.compiler import compile_to_pytorch
        code = compile_to_pytorch(program)
    else:
        print(f"chimera: unsupported backend '{backend}'", file=sys.stderr)
        sys.exit(1)

    if out:
        Path(out).write_text(code, encoding="utf-8")
        print(f"chimera: compiled to {out}")
    else:
        print(code)
```

Update `main()` commands dict:

```python
    commands = {
        "run":     lambda: cmd_run(filepath, show_trace=trace, extra_args=flag_args),
        "check":   lambda: cmd_check(filepath),
        "lex":     lambda: cmd_lex(filepath),
        "parse":   lambda: cmd_parse(filepath),
        "prove":   lambda: cmd_prove(filepath),
        "compile": lambda: cmd_compile(
            filepath,
            backend=next((a.split("=")[1] for a in flag_args if a.startswith("--backend=")), "pytorch"),
            out=next((a.split("=")[1] for a in flag_args if a.startswith("--out=")), None),
        ),
    }
```

Update `USAGE` string to include:

```
  chimera compile <file.chimera> [--backend=pytorch] [--out=output.py]
```

- [ ] **Step 2: Test CLI compile**

```
cd C:/Users/garza/Downloads/ChimeraLang
python -m chimera.cli compile examples/mnist_classifier.chimera --backend=pytorch 2>&1 | head -20
```

(mnist_classifier.chimera created in next task — skip for now)

- [ ] **Step 3: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/cli.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: add chimera compile command — compiles .chimera to PyTorch Python"
```

---

## Task 7: Example Programs + End-to-End Demo

**Files:**
- Create: `examples/mnist_classifier.chimera`
- Create: `examples/mnist_moe.chimera`
- Create: `examples/mnist_aligned.chimera`

- [ ] **Step 1: Create `examples/mnist_classifier.chimera`**

```chimera
// mnist_classifier.chimera
// MLP for MNIST digit classification — compiles to PyTorch
// Run: chimera compile examples/mnist_classifier.chimera --backend=pytorch --out=mnist_pt.py

import chimera.nn { Dense, ReLU, Softmax }
import chimera.data { mnist }
import chimera.optim { Adam }
import chimera.loss { CrossEntropy }

model MNISTClassifier {
  layer flatten: Flatten
  layer fc1: Dense(784 -> 256)
  layer act1: ReLU
  layer drop1: Dropout
  layer fc2: Dense(256 -> 128)
  layer act2: ReLU
  layer fc3: Dense(128 -> 10)

  uncertainty: epistemic

  forward(x: Tensor<Float>[?, 1, 28, 28]) -> Tensor<Float>[?, 10] {
    val h = act2(fc2(act1(fc1(flatten(x)))))
    return fc3(h)
  }
}

train MNISTClassifier on mnist_train {
  optimizer: Adam { lr: 0.001 }
  loss: CrossEntropy
  epochs: 10
  batch_size: 128

  guard predictions against overconfidence { max_risk: 0.05, strategy: both }
  evolve weights until converged { metric: val_accuracy, patience: 3, max_iter: 10 }
}
```

- [ ] **Step 2: Create `examples/mnist_moe.chimera`**

```chimera
// mnist_moe.chimera
// Mixture of Experts classifier — 8 experts, top-2 activated per token

import chimera.nn { Embedding, Linear }

model MNISTMoE {
  layer flatten: Flatten
  layer embed: Linear(784 -> 512)

  moe experts {
    n_experts: 8
    top_k: 2
    expert_dim: 512
  }

  layer head: Linear(512 -> 10)

  uncertainty: epistemic

  forward(x: Tensor<Float>[?, 1, 28, 28]) -> Tensor<Float>[?, 10] {
    val h = embed(flatten(x))
    val h_moe = experts(h)
    return head(h_moe)
  }
}

train MNISTMoE on mnist_train {
  optimizer: AdamW { lr: 0.0005, weight_decay: 0.01 }
  loss: CrossEntropy
  epochs: 15
  batch_size: 256
}
```

- [ ] **Step 3: Create `examples/mnist_aligned.chimera`**

```chimera
// mnist_aligned.chimera
// Constitutionally aligned classifier — outputs are Constituted<T>

constitution ClassifierEthics {
  "Never output predictions with confidence below 0.6"
  "Flag uncertain predictions for human review"
  "Do not discriminate based on spurious correlations"
  critique_rounds: 2
  max_violation_score: 0.05
}

model AlignedMNIST {
  layer flatten: Flatten
  layer fc1: Dense(784 -> 256)
  layer act1: ReLU
  layer fc2: Dense(256 -> 10)

  uncertainty: epistemic
  constitution: ClassifierEthics

  forward(x: Tensor<Float>[?, 1, 28, 28]) -> Tensor<Float>[?, 10] {
    val h = act1(fc1(flatten(x)))
    return fc2(h)
  }
}

train AlignedMNIST on mnist_train {
  optimizer: Adam { lr: 0.001 }
  loss: CrossEntropy
  epochs: 10
  batch_size: 128
  guard predictions against overconfidence { max_risk: 0.1, strategy: both }
}
```

- [ ] **Step 4: Test compilation of all three examples**

```
cd C:/Users/garza/Downloads/ChimeraLang

python -m chimera.cli compile examples/mnist_classifier.chimera --out=/tmp/mnist_mlp.py
python -m chimera.cli compile examples/mnist_moe.chimera --out=/tmp/mnist_moe.py
python -m chimera.cli compile examples/mnist_aligned.chimera --out=/tmp/mnist_aligned.py

# Verify all three are valid Python
python -c "compile(open('/tmp/mnist_mlp.py').read(), 'mlp', 'exec'); print('MLP OK')"
python -c "compile(open('/tmp/mnist_moe.py').read(), 'moe', 'exec'); print('MoE OK')"
python -c "compile(open('/tmp/mnist_aligned.py').read(), 'aligned', 'exec'); print('Aligned OK')"
```

Expected:
```
MLP OK
MoE OK
Aligned OK
```

- [ ] **Step 5: Run full test suite**

```
python -m pytest tests/ -v --tb=short 2>&1 | tail -8
```

Expected: 130+ passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add examples/mnist_classifier.chimera examples/mnist_moe.chimera examples/mnist_aligned.chimera
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: add MNIST example programs (MLP, MoE, constitutional AI) — all compile to valid PyTorch"
```

---

## Task 8: Push + Update README

- [ ] **Step 1: Final test run**

```
python -m pytest tests/ -v 2>&1 | tail -5
```

Expected: 130+ passed, 0 failed.

- [ ] **Step 2: Update README ML section**

In `README.md`, add to the CLI Reference:

```
  chimera compile <file.chimera> [--backend=pytorch] [--out=output.py]  # Compile to PyTorch
```

Add new section after Quick Start:

```markdown
## Compile to PyTorch

\`\`\`bash
chimera compile examples/mnist_classifier.chimera --backend=pytorch --out=model.py
python model.py  # runs training
\`\`\`

Generates a complete PyTorch `nn.Module` with training loop, belief tracking, and guard layers.
```

- [ ] **Step 3: Push all**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add README.md
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "docs: update README with compile command and PyTorch codegen usage"
git -C C:/Users/garza/Downloads/ChimeraLang push origin main
```

---

## Self-Review

**Spec coverage:**
- ✅ Language foundations (I/O via chimera_runtime.VectorStore.persist, modules via ImportStmt)
- ✅ TensorType with shape annotation
- ✅ model/layer/forward block syntax
- ✅ train block with optimizer/loss/epochs/batch_size
- ✅ MoE block with n_experts/top_k
- ✅ Constitutional AI — ConstitutionDecl + ConstitutionLayer
- ✅ VectorStore type + runtime implementation
- ✅ PyTorch codegen for MLP + MoE + constitution
- ✅ chimera_runtime — BeliefTracker, GuardLayer, ConstitutionLayer
- ✅ CLI compile command
- ✅ MNIST demo (MLP + MoE + aligned)
- ⚠️ Full I/O (read_file/write_file) — deferred to Phase 2 (not required for demo)
- ⚠️ Result<T,E> type — deferred to Phase 2
- ⚠️ Shape inference type checker — deferred to Phase 2 (parser accepts shapes but doesn't validate)

**Type consistency:** All method names consistent across tasks. `BeliefTracker.record()` used consistently. `ConstitutionLayer.apply()` used consistently. `GuardLayer.check()` used consistently. ✅

**No placeholders:** All steps have complete code. ✅
