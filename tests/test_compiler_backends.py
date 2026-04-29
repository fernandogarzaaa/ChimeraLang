from chimera.lexer import Lexer
from chimera.parser import Parser


def parse(src: str):
    return Parser(Lexer(src).tokenize()).parse()


def test_compiler_exports_llvm_backend():
    from chimera.compiler import compile_to_llvm

    program = parse(
        """
model NativeMLP {
  layer fc1: Dense(4 -> 2)
  layer act1: ReLU
}
"""
    )

    llvm_ir = compile_to_llvm(program)

    assert 'source_filename = "chimeralang.module"' in llvm_ir
    assert "@NativeMLP_forward" in llvm_ir


def test_cli_compile_llvm_backend_outputs_ir(tmp_path):
    from chimera.cli import cmd_compile

    src = tmp_path / "native.chimera"
    out = tmp_path / "native.ll"
    src.write_text(
        """
model NativeMLP {
  layer fc1: Dense(4 -> 2)
}
""",
        encoding="utf-8",
    )

    cmd_compile(str(src), backend="llvm", out=str(out))

    assert out.read_text(encoding="utf-8").startswith('; Module ID "chimeralang.module"')
