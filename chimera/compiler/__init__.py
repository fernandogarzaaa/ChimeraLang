"""ChimeraLang compiler — PyTorch backend."""


def compile_to_pytorch(program: object) -> str:
    """Compile a ChimeraLang AST Program to PyTorch Python source code."""
    from chimera.compiler.codegen_pytorch import PyTorchCodegen
    return PyTorchCodegen().generate(program)
