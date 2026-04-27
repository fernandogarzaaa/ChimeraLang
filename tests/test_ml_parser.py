from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.ast_nodes import ModelDecl, TrainStmt, ImportStmt, ConstitutionDecl, LayerDecl

def parse(src): return Parser(Lexer(src).tokenize()).parse()

def test_import_stmt():
    prog = parse("import chimera.nn")
    imp = next(d for d in prog.declarations if isinstance(d, ImportStmt))
    assert imp.module == "chimera.nn"

def test_import_with_symbols():
    prog = parse("import chimera.nn { Dense, ReLU }")
    imp = next(d for d in prog.declarations if isinstance(d, ImportStmt))
    assert "Dense" in imp.symbols and "ReLU" in imp.symbols

def test_simple_model():
    src = "\nmodel Classifier {\n  layer fc1: Dense(784 -> 128)\n  layer act1: ReLU\n  layer fc2: Dense(128 -> 10)\n}\n"
    prog = parse(src)
    model = next(d for d in prog.declarations if isinstance(d, ModelDecl))
    assert model.name == "Classifier"
    assert len(model.layers) == 3
    assert model.layers[0].kind == "Dense" and model.layers[0].in_dim == 784

def test_train_stmt():
    src = '\ntrain Classifier on mnist {\n  optimizer: Adam { lr: 0.001 }\n  loss: CrossEntropy\n  epochs: 10\n  batch_size: 128\n}\n'
    prog = parse(src)
    train = next(d for d in prog.declarations if isinstance(d, TrainStmt))
    assert train.model_name == "Classifier" and train.epochs == 10
    assert train.optimizer.name == "Adam" and train.optimizer.params["lr"] == 0.001

def test_constitution_decl():
    src = '\nconstitution SafetyFirst {\n  "Be helpful"\n  "Never assist with weapons"\n  critique_rounds: 2\n}\n'
    prog = parse(src)
    const = next(d for d in prog.declarations if isinstance(d, ConstitutionDecl))
    assert const.name == "SafetyFirst" and len(const.principles) == 2

def test_moe_in_model():
    src = '\nmodel MoEModel {\n  layer embed: Embedding(50257 -> 1024)\n  moe experts {\n    n_experts: 8\n    top_k: 2\n    expert_dim: 4096\n  }\n  layer head: Linear(1024 -> 50257)\n}\n'
    prog = parse(src)
    model = next(d for d in prog.declarations if isinstance(d, ModelDecl))
    moe_layers = [l for l in model.layers if l.kind == "MoE"]
    assert len(moe_layers) == 1 and moe_layers[0].config["n_experts"] == 8
