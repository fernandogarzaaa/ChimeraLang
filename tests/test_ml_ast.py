from chimera.ast_nodes import (
    TensorType, ModelDecl, LayerDecl, ForwardFn, TrainStmt,
    MoEBlock, ConstitutionDecl, VectorStoreType, ImportStmt,
    ConstitutedType, OptimizerConfig, LossConfig,
)
def test_tensor_type():
    t = TensorType(dtype="Float", shape=[None, 784])
    assert t.dtype == "Float" and t.shape == [None, 784]
def test_model_decl():
    assert ModelDecl(name="Foo", layers=[], forward_fn=None).name == "Foo"
def test_layer_decl_dense():
    l = LayerDecl(name="fc1", kind="Dense", in_dim=784, out_dim=128, config={})
    assert l.kind == "Dense" and l.in_dim == 784
def test_layer_decl_activation():
    assert LayerDecl(name="act1", kind="ReLU", in_dim=None, out_dim=None, config={}).kind == "ReLU"
def test_forward_fn():
    f = ForwardFn(params=[("x", TensorType("Float", [None, 784]))], return_type=None, body=[])
    assert f.params[0][0] == "x"
def test_train_stmt():
    t = TrainStmt(model_name="Foo", dataset="mnist", optimizer=OptimizerConfig("Adam", {"lr": 0.001}), loss=LossConfig("CrossEntropy"), epochs=10, batch_size=128, guards=[], evolve=None)
    assert t.model_name == "Foo" and t.optimizer.name == "Adam"
def test_moe_block():
    assert MoEBlock(name="MyMoE", n_experts=64, top_k=2, expert_dim=4096, config={}).n_experts == 64
def test_constitution_decl():
    assert len(ConstitutionDecl(name="MyConst", principles=["Be helpful"], critique_rounds=2).principles) == 1
def test_vectorstore_type():
    assert VectorStoreType(dim=768, capacity=1_000_000).dim == 768
def test_import_stmt():
    assert ImportStmt(module="chimera.nn", symbols=["Dense", "ReLU"], alias=None).module == "chimera.nn"
def test_constituted_type():
    assert ConstitutedType(inner_type=TensorType("Float", [None, 10])).inner_type.dtype == "Float"
