from chimera.lexer import Lexer
from chimera.tokens import TokenKind


def test_lexer_ignores_leading_utf8_bom():
    tokens = Lexer("\ufeffval answer: Int = 1").tokenize()

    assert tokens[0].kind == TokenKind.VAL
