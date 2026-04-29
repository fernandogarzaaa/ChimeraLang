"""Tests for MoE (Mixture of Experts) — Phase 3."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chimera.ast_nodes import (
    ModelDecl, LayerDecl, ForwardFn, MoEBlock, ExpertRoute
)
from chimera.parser import Parser
from chimera.lexer import Lexer
from chimera.compiler.codegen_pytorch import PyTorchCodegen


class TestMoEBlock:
    def test_expert_route_defaults(self):
        r = ExpertRoute()
        assert r.n_experts == 8
        assert r.top_k == 2
        assert r.uncertainty == 0.0

    def test_expert_route_custom(self):
        r = ExpertRoute(n_experts=64, top_k=3, uncertainty=0.12)
        assert r.n_experts == 64
        assert r.top_k == 3
        assert abs(r.uncertainty - 0.12) < 1e-6

    def test_moe_block_defaults(self):
        m = MoEBlock()
        assert m.n_experts == 8
        assert m.top_k == 2
        assert m.expert_dim == 4096

    def test_moe_block_custom(self):
        m = MoEBlock(name="experts", n_experts=16, top_k=4, expert_dim=2048)
        assert m.name == "experts"
        assert m.n_experts == 16
        assert m.top_k == 4
        assert m.expert_dim == 2048


class TestMoEParsing:
    def _parse(self, src: str):
        lexer = Lexer(src)
        tokens = lexer.tokens
        return Parser(tokens).parse()

    def test_parse_moe_layer_basic(self):
        src = "model Foo\nlayer experts: MoE {n_experts: 8, top_k: 2, expert_dim: 4096}\nforward(x: Tensor<Float>[batch, seq]) -> Tensor<Float>[batch, seq] { return x }\nend"
        prog = self._parse(src)
        model = prog.declarations[0]
        assert model.name == "Foo"
        assert len(model.layers) == 1
        assert model.layers[0].kind == "MoE"
        assert model.layers[0].config["n_experts"] == 8
        assert model.layers[0].config["top_k"] == 2

    def test_parse_moe_layer_large(self):
        src = "model BigMoE\nlayer experts: MoE {n_experts: 64, top_k: 2, expert_dim: 4096, aux_loss_weight: 0.01}\nforward(x) -> x\nend"
        prog = self._parse(src)
        layer = prog.declarations[0].layers[0]
        assert layer.config["n_experts"] == 64
        assert layer.config["aux_loss_weight"] == 0.01

    def test_parse_model_with_multiple_moe_layers(self):
        src = "model DualMoE\nlayer embed: Embedding(50257 -> 1024)\nlayer experts1: MoE {n_experts: 16, top_k: 2, expert_dim: 1024}\nlayer experts2: MoE {n_experts: 16, top_k: 2, expert_dim: 1024}\nlayer head: Linear(1024 -> 50257)\nforward(x) -> x\nend"
        prog = self._parse(src)
        model = prog.declarations[0]
        moe_layers = [l for l in model.layers if l.kind == "MoE"]
        assert len(moe_layers) == 2


class TestMoECodegen:
    def _codegen(self, src: str) -> str:
        lexer = Lexer(src)
        tokens = lexer.tokens
        prog = Parser(tokens).parse()
        return PyTorchCodegen().generate(prog)

    def test_moe_codegen_expert_list(self):
        src = "model MoE\nlayer experts: MoE {n_experts: 8, top_k: 2, expert_dim: 256}\nforward(x) -> x\nend"
        code = self._codegen(src)
        assert "nn.ModuleList([nn.Linear(256, 256)" in code
        assert "self.experts_router = nn.Linear" in code

    def test_moe_codegen_router_has_no_bias(self):
        src = "model MoE\nlayer experts: MoE {n_experts: 4, top_k: 2, expert_dim: 128}\nforward(x) -> x\nend"
        code = self._codegen(src)
        assert "self.experts_router = nn.Linear(128, 4, bias=False)" in code

    def test_moe_codegen_aux_loss_weight(self):
        src = "model MoE\nlayer experts: MoE {n_experts: 8, top_k: 2, expert_dim: 256, aux_loss_weight: 0.02}\nforward(x) -> x\nend"
        code = self._codegen(src)
        assert "self._experts_aux_weight = 0.02" in code

    def test_moe_codegen_router_noise(self):
        src = "model MoE\nlayer experts: MoE {n_experts: 8, top_k: 2, expert_dim: 256}\nforward(x) -> x\nend"
        code = self._codegen(src)
        assert "self._experts_router_noise = nn.Parameter(torch.zeros(1, 8))" in code

    def test_moe_codegen_forward_topk(self):
        src = "model MoE\nlayer experts: MoE {n_experts: 8, top_k: 2, expert_dim: 256}\nforward(x) -> x\nend"
        code = self._codegen(src)
        assert "torch.topk(_router_logits, self._experts_top_k" in code

    def test_moe_codegen_betadist_variance(self):
        src = "model MoE\nlayer experts: MoE {n_experts: 8, top_k: 2, expert_dim: 256}\nforward(x) -> x\nend"
        code = self._codegen(src)
        assert "torch.softmax" in code
        assert "variance" in code or "uncertainty" in code

    def test_moe_codegen_load_balancing_aux_loss(self):
        src = "model MoE\nlayer experts: MoE {n_experts: 8, top_k: 2, expert_dim: 256}\nforward(x) -> x\nend"
        code = self._codegen(src)
        assert "_load_loss" in code or "aux_weight" in code

    def test_moe_codegen_64_experts(self):
        src = "model BigMoE\nlayer experts: MoE {n_experts: 64, top_k: 2, expert_dim: 4096}\nforward(x) -> x\nend"
        code = self._codegen(src)
        assert "nn.ModuleList([nn.Linear(4096, 4096)" in code
        assert "self.experts_router = nn.Linear(4096, 64" in code


class TestMoEIntegration:
    """End-to-end MoE parsing + codegen."""
    def test_moe_roundtrip(self):
        src = "model ChimeraMoE\nlayer embed: Embedding(50257 -> 1024)\nlayer experts: MoE {n_experts: 64, top_k: 2, expert_dim: 4096, aux_loss_weight: 0.01}\nlayer head: Linear(1024 -> 50257)\nuncertainty: epistemic\nforward(tokens: Tensor<Float>[batch, seq]) -> Tensor<Float>[batch, seq] { return tokens }\nend"
        lexer = Lexer(src)
        tokens = lexer.tokens
        prog = Parser(tokens).parse()
        code = PyTorchCodegen().generate(prog)
        assert "nn.ModuleList([nn.Linear(4096, 4096)" in code
        assert "self.experts_router = nn.Linear(4096, 64, bias=False)" in code
        assert "self._experts_aux_weight = 0.01" in code
        assert "self._belief_tracker = BeliefTracker" in code
        assert "torch.softmax" in code