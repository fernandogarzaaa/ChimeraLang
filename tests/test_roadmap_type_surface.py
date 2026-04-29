from chimera.ast_nodes import MemPtrType, MultimodalType, SpikeTrainType, TensorType
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.type_checker import TypeChecker
from chimera.types import MemPtrTypeDesc, MultimodalTypeDesc, PrivacyTypeDesc, SpikeTrainTypeDesc, TensorTypeDesc


def parse(src: str):
    return Parser(Lexer(src).tokenize()).parse()


def test_parser_accepts_causal_and_private_tensor_annotations():
    program = parse(
        """
val observed: Tensor<Float>[n] @observational = x
val secret: Tensor<Float>[10] @private(epsilon=1.0, delta=0.00001) = y
"""
    )

    observed = program.declarations[0].type_ann
    secret = program.declarations[1].type_ann

    assert isinstance(observed, TensorType)
    assert observed.causality == "observational"
    assert isinstance(secret, TensorType)
    assert secret.privacy == {"epsilon": 1.0, "delta": 0.00001}


def test_parser_accepts_spiketrain_multimodal_and_memptr_types():
    program = parse(
        """
val spikes: SpikeTrain<Float>[128, 100] = s
val prompt: Multimodal<Text> = m
val memory: MemPtr<Text>[42] = ptr
"""
    )

    assert isinstance(program.declarations[0].type_ann, SpikeTrainType)
    assert isinstance(program.declarations[1].type_ann, MultimodalType)
    assert isinstance(program.declarations[2].type_ann, MemPtrType)


def test_type_checker_resolves_roadmap_types():
    checker = TypeChecker()
    result = checker.check(
        parse(
            """
val observed: Tensor<Float>[n] @interventional = x
val secret: Tensor<Float>[10] @private(epsilon=1.0, delta=0.00001) = y
val spikes: SpikeTrain<Float>[128, 100] = s
val prompt: Multimodal<Text> = m
val memory: MemPtr<Text>[42] = ptr
"""
        )
    )

    assert result.ok
    assert isinstance(checker._env.lookup("observed"), TensorTypeDesc)
    assert isinstance(checker._env.lookup("secret"), PrivacyTypeDesc)
    assert isinstance(checker._env.lookup("spikes"), SpikeTrainTypeDesc)
    assert isinstance(checker._env.lookup("prompt"), MultimodalTypeDesc)
    assert isinstance(checker._env.lookup("memory"), MemPtrTypeDesc)
