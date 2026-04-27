from chimera.lexer import Lexer
from chimera.tokens import TokenKind

def lex(src): return [t for t in Lexer(src).tokenize() if t.kind not in (TokenKind.NEWLINE, TokenKind.EOF)]
def test_model_token():
    assert lex("model Foo")[0].kind == TokenKind.MODEL
def test_layer_token():
    assert lex("layer fc1")[0].kind == TokenKind.LAYER
def test_train_token():
    assert lex("train Foo")[0].kind == TokenKind.TRAIN
def test_forward_token():
    assert lex("forward(x)")[0].kind == TokenKind.FORWARD
def test_import_token():
    assert lex("import chimera.nn")[0].kind == TokenKind.IMPORT
def test_tensor_type_token():
    assert lex("Tensor")[0].kind == TokenKind.TENSOR_KW
def test_constitution_token():
    assert lex("constitution MyConst")[0].kind == TokenKind.CONSTITUTION
def test_moe_token():
    assert lex("moe MyMoE")[0].kind == TokenKind.MOE
def test_vectorstore_token():
    assert lex("VectorStore")[0].kind == TokenKind.VECTORSTORE_KW
def test_on_token():
    assert lex("on gpu")[0].kind == TokenKind.ON
