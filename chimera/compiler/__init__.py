"""ChimeraLang compiler backends."""


def compile_to_pytorch(program: object) -> str:
    """Compile a ChimeraLang AST Program to PyTorch Python source code."""
    from chimera.compiler.codegen_pytorch import PyTorchCodegen
    return PyTorchCodegen().generate(program)


def compile_to_llvm(program: object) -> str:
    """Compile a ChimeraLang AST Program to LLVM IR text."""
    from chimera.compiler.llvm_backend import LLVMEmitter
    return LLVMEmitter().emit(program)
