"""Parser for ChimeraLang.

Consumes a token stream and produces an AST.
"""

from __future__ import annotations

from chimera.ast_nodes import (
    AllowConstraint,
    AssertStmt,
    BeliefDecl,
    BinaryOp,
    BoolLiteral,
    CallExpr,
    CompareChain,
    CausalModelDecl,
    ConstitutedType,
    ConstitutionDecl,
    Constraint,
    Declaration,
    EmitStmt,
    EvolveConfig,
    EvolveStmt,
    Expr,
    ExprStmt,
    FederatedTrainStmt,
    FloatLiteral,
    FnDecl,
    ForbiddenConstraint,
    ForStmt,
    ForwardFn,
    GateDecl,
    GoalDecl,
    GradType,
    GuardStmt,
    Identifier,
    IfExpr,
    ImportStmt,
    InquireExpr,
    IntLiteral,
    LayerDecl,
    ListLiteral,
    LossConfig,
    MatchArm,
    MatchExpr,
    MemPtrType,
    MemberExpr,
    MemoryType,
    MetaTrainStmt,
    ModelDecl,
    MoEBlock,
    MustConstraint,
    NamedType,
    OptimizerConfig,
    Param,
    PredictiveCodingDecl,
    PrimitiveType,
    ProbabilisticType,
    Program,
    ReasonDecl,
    ResolveStmt,
    RetrievalDecl,
    ReturnStmt,
    ReplayBufferDecl,
    RewardSystemDecl,
    SelfImproveDecl,
    Statement,
    StringLiteral,
    SwarmDecl,
    SymbolDecl,
    SpikeTrainType,
    TensorType,
    TrainStmt,
    TypeExpr,
    UnaryOp,
    ValDecl,
    VectorStoreType,
    GenericType,
    MultimodalType,
)
from chimera.tokens import Token, TokenKind


class ParseError(Exception):
    def __init__(self, message: str, token: Token) -> None:
        self.token = token
        loc = f"L{token.span.line}:{token.span.col}"
        super().__init__(f"ParseError at {loc}: {message} (got {token.kind.name} {token.value!r})")


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self) -> Program:
        decls: list[Declaration | Statement] = []
        self._skip_newlines()
        while not self._check(TokenKind.EOF):
            decls.append(self._parse_top_level())
            self._skip_newlines()
        return Program(declarations=decls, span=None)

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _peek_kind(self, offset: int = 0) -> TokenKind:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx].kind
        return TokenKind.EOF

    def _check(self, kind: TokenKind) -> bool:
        return self._current().kind == kind

    def _match(self, *kinds: TokenKind) -> Token | None:
        if self._current().kind in kinds:
            return self._advance()
        return None

    def _expect(self, kind: TokenKind, msg: str = "") -> Token:
        if self._current().kind == kind:
            return self._advance()
        raise ParseError(msg or f"Expected {kind.name}", self._current())

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _skip_newlines(self) -> None:
        while self._check(TokenKind.NEWLINE):
            self._advance()

    def _expect_line_end(self) -> None:
        if not self._check(TokenKind.NEWLINE) and not self._check(TokenKind.EOF):
            pass  # lenient — don't error on missing newline
        self._skip_newlines()

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def _parse_top_level(self) -> Declaration | Statement:
        kind = self._current().kind
        if kind == TokenKind.FN:
            return self._parse_fn()
        if kind == TokenKind.GATE:
            return self._parse_gate()
        if kind == TokenKind.GOAL:
            return self._parse_goal()
        if kind == TokenKind.REASON:
            return self._parse_reason()
        if kind == TokenKind.VAL:
            return self._parse_val()
        if kind == TokenKind.BELIEF:
            return self._parse_belief()
        if kind == TokenKind.RESOLVE:
            return self._parse_resolve()
        if kind == TokenKind.GUARD:
            return self._parse_guard()
        if kind == TokenKind.EVOLVE:
            return self._parse_evolve()
        if kind == TokenKind.SYMBOL:
            return self._parse_symbol_decl()
        if kind == TokenKind.IMPORT:
            return self._parse_import()
        if kind == TokenKind.MODEL:
            return self._parse_model()
        if kind == TokenKind.TRAIN:
            return self._parse_train()
        if kind == TokenKind.CONSTITUTION:
            return self._parse_constitution()
        if kind == TokenKind.IDENT:
            value = self._current().value
            if value == "causal_model":
                return self._parse_named_config_block(CausalModelDecl)
            if value == "federated_train":
                return self._parse_named_config_block(FederatedTrainStmt)
            if value == "meta_train":
                return self._parse_named_config_block(MetaTrainStmt)
            if value == "self_improve":
                return self._parse_named_config_block(SelfImproveDecl)
            if value == "swarm":
                return self._parse_named_config_block(SwarmDecl)
            if value == "replay_buffer":
                return self._parse_named_config_block(ReplayBufferDecl)
            if value == "reward_system":
                return self._parse_named_config_block(RewardSystemDecl)
            if value == "predictive_coding":
                return self._parse_named_config_block(PredictiveCodingDecl)
        return self._parse_statement()

    # ------------------------------------------------------------------
    # Function declaration
    # ------------------------------------------------------------------

    def _parse_fn(self) -> FnDecl:
        span = self._expect(TokenKind.FN).span
        name = self._expect(TokenKind.IDENT, "Expected function name").value
        self._expect(TokenKind.LPAREN, "Expected '(' after function name")
        params = self._parse_param_list()
        self._expect(TokenKind.RPAREN, "Expected ')' after parameters")

        ret_type: TypeExpr | None = None
        if self._match(TokenKind.ARROW):
            ret_type = self._parse_type()

        self._expect_line_end()

        constraints: list[Constraint] = []
        body: list[Statement] = []
        while not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.END):
                break
            c = self._try_parse_constraint()
            if c is not None:
                constraints.append(c)
            else:
                body.append(self._parse_statement())
            self._skip_newlines()

        self._expect(TokenKind.END, "Expected 'end' to close function")
        self._expect_line_end()
        return FnDecl(name=name, params=params, return_type=ret_type,
                       constraints=constraints, body=body, span=span)

    # ------------------------------------------------------------------
    # Gate declaration
    # ------------------------------------------------------------------

    def _parse_gate(self) -> GateDecl:
        span = self._expect(TokenKind.GATE).span
        name = self._expect(TokenKind.IDENT, "Expected gate name").value
        self._expect(TokenKind.LPAREN, "Expected '(' after gate name")
        params = self._parse_param_list()
        self._expect(TokenKind.RPAREN, "Expected ')' after parameters")

        ret_type: TypeExpr | None = None
        if self._match(TokenKind.ARROW):
            ret_type = self._parse_type()

        self._expect_line_end()

        branches = 3
        collapse = "majority"
        threshold = 0.85
        fallback = "escalate"
        body: list[Statement] = []

        while not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.END):
                break
            if self._check(TokenKind.BRANCHES):
                self._advance()
                self._expect(TokenKind.COLON, "Expected ':' after 'branches'")
                branches = int(self._expect(TokenKind.INT_LIT, "Expected integer").value)
                self._expect_line_end()
            elif self._check(TokenKind.COLLAPSE):
                self._advance()
                self._expect(TokenKind.COLON, "Expected ':' after 'collapse'")
                collapse = self._advance().value
                self._expect_line_end()
            elif self._check(TokenKind.THRESHOLD):
                self._advance()
                self._expect(TokenKind.COLON, "Expected ':' after 'threshold'")
                tok = self._advance()
                threshold = float(tok.value)
                self._expect_line_end()
            elif self._check(TokenKind.FALLBACK):
                self._advance()
                self._expect(TokenKind.COLON, "Expected ':' after 'fallback'")
                fallback = self._advance().value
                self._expect_line_end()
            else:
                body.append(self._parse_statement())
                self._skip_newlines()

        self._expect(TokenKind.END, "Expected 'end' to close gate")
        self._expect_line_end()
        return GateDecl(name=name, params=params, return_type=ret_type,
                         branches=branches, collapse=collapse,
                         threshold=threshold, fallback=fallback,
                         body=body, span=span)

    # ------------------------------------------------------------------
    # Goal declaration
    # ------------------------------------------------------------------

    def _parse_goal(self) -> GoalDecl:
        span = self._expect(TokenKind.GOAL).span
        desc = self._expect(TokenKind.STRING_LIT, "Expected goal description string").value
        self._expect_line_end()

        constraint_list: list[str] = []
        quality_axes: list[str] = []
        explore_budget = 1.0
        body: list[Statement] = []

        while not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.END):
                break
            if self._check(TokenKind.CONSTRAINTS):
                self._advance()
                self._expect(TokenKind.COLON)
                constraint_list = self._parse_string_list_block()
            elif self._check(TokenKind.QUALITY):
                self._advance()
                self._expect(TokenKind.COLON)
                quality_axes = self._parse_string_list_block()
            elif self._check(TokenKind.EXPLORE_BUDGET):
                self._advance()
                self._expect(TokenKind.COLON)
                tok = self._advance()
                explore_budget = float(tok.value)
                self._expect_line_end()
            else:
                body.append(self._parse_statement())
                self._skip_newlines()

        self._expect(TokenKind.END, "Expected 'end' to close goal")
        self._expect_line_end()
        return GoalDecl(description=desc, constraints_list=constraint_list,
                         quality_axes=quality_axes, explore_budget=explore_budget,
                         body=body, span=span)

    # ------------------------------------------------------------------
    # Reason declaration
    # ------------------------------------------------------------------

    def _parse_reason(self) -> ReasonDecl:
        span = self._expect(TokenKind.REASON).span
        self._expect(TokenKind.ABOUT, "Expected 'about' after 'reason'")
        self._expect(TokenKind.LPAREN)
        params = self._parse_param_list()
        self._expect(TokenKind.RPAREN)

        ret_type: TypeExpr | None = None
        if self._match(TokenKind.ARROW):
            ret_type = self._parse_type()

        self._expect_line_end()

        given: list[str] = []
        commit_strategy = "highest_consensus"
        body: list[Statement] = []

        while not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.END):
                break
            if self._check(TokenKind.GIVEN):
                self._advance()
                self._expect(TokenKind.COLON)
                given = self._parse_string_list_block()
            elif self._check(TokenKind.COMMIT):
                self._advance()
                self._expect(TokenKind.COLON)
                commit_strategy = self._advance().value
                self._expect_line_end()
            else:
                body.append(self._parse_statement())
                self._skip_newlines()

        self._expect(TokenKind.END, "Expected 'end' to close reason block")
        self._expect_line_end()
        return ReasonDecl(name="about", params=params, return_type=ret_type,
                           given=given, commit_strategy=commit_strategy,
                           body=body, span=span)

    # ------------------------------------------------------------------
    # Val declaration
    # ------------------------------------------------------------------

    def _parse_val(self) -> ValDecl:
        self._expect(TokenKind.VAL)
        name = self._expect(TokenKind.IDENT, "Expected variable name").value
        type_ann: TypeExpr | None = None
        if self._match(TokenKind.COLON):
            type_ann = self._parse_type()
        value: Expr | None = None
        if self._match(TokenKind.ASSIGN):
            value = self._parse_expr()
        self._expect_line_end()
        return ValDecl(name=name, type_ann=type_ann, value=value)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _parse_statement(self) -> Statement:
        if self._check(TokenKind.VAL):
            return self._parse_val()
        if self._check(TokenKind.RETURN):
            return self._parse_return()
        if self._check(TokenKind.ASSERT):
            return self._parse_assert()
        if self._check(TokenKind.EMIT):
            return self._parse_emit()
        if self._check(TokenKind.IF):
            return ExprStmt(expr=self._parse_if_expr())
        if self._check(TokenKind.FOR):
            return self._parse_for()
        if self._check(TokenKind.MATCH):
            return ExprStmt(expr=self._parse_match_expr())
        if self._check(TokenKind.DETECT):
            return self._parse_detect()
        if self._check(TokenKind.RESOLVE):
            return self._parse_resolve()
        if self._check(TokenKind.GUARD):
            return self._parse_guard()
        if self._check(TokenKind.EVOLVE):
            return self._parse_evolve()
        expr = self._parse_expr()
        self._expect_line_end()
        return ExprStmt(expr=expr)

    def _parse_return(self) -> ReturnStmt:
        self._expect(TokenKind.RETURN)
        value: Expr | None = None
        if not self._check(TokenKind.NEWLINE) and not self._check(TokenKind.EOF) and not self._check(TokenKind.END):
            value = self._parse_expr()
        self._expect_line_end()
        return ReturnStmt(value=value)

    def _parse_assert(self) -> AssertStmt:
        self._expect(TokenKind.ASSERT)
        cond = self._parse_expr()
        self._expect_line_end()
        return AssertStmt(condition=cond)

    def _parse_emit(self) -> EmitStmt:
        self._expect(TokenKind.EMIT)
        value = self._parse_expr()
        self._expect_line_end()
        return EmitStmt(value=value)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _try_parse_constraint(self) -> Constraint | None:
        if self._check(TokenKind.MUST):
            self._advance()
            self._expect(TokenKind.COLON)
            self._skip_newlines()
            expr = self._parse_expr()
            self._expect_line_end()
            return MustConstraint(expr=expr)
        if self._check(TokenKind.ALLOW):
            self._advance()
            self._expect(TokenKind.COLON)
            caps = self._parse_string_list_block()
            return AllowConstraint(capabilities=caps)
        if self._check(TokenKind.FORBIDDEN):
            self._advance()
            self._expect(TokenKind.COLON)
            caps = self._parse_string_list_block()
            return ForbiddenConstraint(capabilities=caps)
        return None

    def _parse_string_list_block(self) -> list[str]:
        """Parse one or more string literals, each on its own line."""
        self._skip_newlines()
        items: list[str] = []
        while self._check(TokenKind.STRING_LIT):
            items.append(self._advance().value)
            self._skip_newlines()
        return items

    # ------------------------------------------------------------------
    # For loop
    # ------------------------------------------------------------------

    def _parse_for(self) -> ForStmt:
        """Parse: for <ident> in <expr> NEWLINE body end"""
        self._expect(TokenKind.FOR)
        target = self._expect(TokenKind.IDENT, "Expected loop variable name").value
        self._expect(TokenKind.IN, "Expected 'in' after loop variable")
        iterable = self._parse_expr()
        self._expect_line_end()
        body: list[Statement] = []
        while not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.END):
                break
            body.append(self._parse_statement())
            self._skip_newlines()
        self._expect(TokenKind.END, "Expected 'end' to close for loop")
        self._expect_line_end()
        return ForStmt(target=target, iterable=iterable, body=body)

    # ------------------------------------------------------------------
    # Match expression
    # ------------------------------------------------------------------

    def _parse_match_expr(self) -> MatchExpr:
        """Parse: match <expr> NEWLINE (| <pattern> => body)* end"""
        self._expect(TokenKind.MATCH)
        subject = self._parse_expr()
        self._expect_line_end()
        arms: list[MatchArm] = []
        while not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.END):
                break
            self._expect(TokenKind.PIPE, "Expected '|' to start a match arm")
            # Wildcard arm: _
            if self._check(TokenKind.UNDERSCORE):
                self._advance()
                pattern: Expr | None = None
            else:
                pattern = self._parse_expr()
            self._expect(TokenKind.FAT_ARROW, "Expected '=>' after match pattern")
            self._skip_newlines()
            arm_body: list[Statement] = []
            # Arm body: statements until the next '|', 'end', or EOF
            while (
                not self._check(TokenKind.PIPE)
                and not self._check(TokenKind.END)
                and not self._check(TokenKind.EOF)
            ):
                self._skip_newlines()
                if self._check(TokenKind.PIPE) or self._check(TokenKind.END):
                    break
                arm_body.append(self._parse_statement())
                self._skip_newlines()
            arms.append(MatchArm(pattern=pattern, body=arm_body))
        self._expect(TokenKind.END, "Expected 'end' to close match")
        self._expect_line_end()
        return MatchExpr(subject=subject, arms=arms)

    # ------------------------------------------------------------------
    # Detect statement
    # ------------------------------------------------------------------

    def _parse_detect(self) -> ExprStmt:
        """Parse: detect <ident> NEWLINE key: value ... end
        Translates to a call: __detect__(<name>, key=value, ...)"""
        self._expect(TokenKind.DETECT)
        # The thing being detected — typically an identifier like 'hallucination'
        name_tok = self._advance()
        detect_name = name_tok.value
        self._expect_line_end()
        args: list[Expr] = [StringLiteral(value=detect_name)]
        while not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.END):
                break
            # key: value pairs
            key_tok = self._advance()
            self._expect(TokenKind.COLON, "Expected ':' in detect block")
            val_expr = self._parse_expr()
            self._expect_line_end()
            args.append(StringLiteral(value=key_tok.value))
            args.append(val_expr)
        self._expect(TokenKind.END, "Expected 'end' to close detect block")
        self._expect_line_end()
        callee = Identifier(name="__detect__")
        return ExprStmt(expr=CallExpr(callee=callee, args=args))

    # ------------------------------------------------------------------
    # CIR / Belief constructs
    # ------------------------------------------------------------------

    def _parse_belief(self) -> BeliefDecl:
        """belief <name> := inquire { prompt: "...", agents: [...], ttl: N }"""
        span = self._expect(TokenKind.BELIEF).span
        name = self._expect(TokenKind.IDENT, "Expected belief name").value
        self._expect(TokenKind.WALRUS, "Expected ':=' after belief name")
        inquire_expr = self._parse_inquire_expr()
        self._expect_line_end()
        return BeliefDecl(name=name, inquire_expr=inquire_expr, span=span)

    def _parse_inquire_expr(self) -> InquireExpr:
        """inquire { prompt: "...", agents: [...], ttl: N }"""
        self._expect(TokenKind.INQUIRE, "Expected 'inquire'")
        self._expect(TokenKind.LBRACE, "Expected '{' after 'inquire'")
        self._skip_newlines()
        prompt = ""
        agents: list[str] = []
        ttl: float | None = None
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            key_tok = self._advance()
            key = key_tok.value
            self._expect(TokenKind.COLON, f"Expected ':' after '{key}'")
            if key == "prompt":
                prompt = self._expect(TokenKind.STRING_LIT, "Expected prompt string").value
            elif key == "agents":
                self._expect(TokenKind.LBRACKET, "Expected '[' for agents list")
                while not self._check(TokenKind.RBRACKET) and not self._check(TokenKind.EOF):
                    agents.append(self._advance().value)
                    self._match(TokenKind.COMMA)
                self._expect(TokenKind.RBRACKET, "Expected ']' after agents")
            elif key == "ttl":
                tok = self._advance()
                ttl = float(tok.value)
            self._match(TokenKind.COMMA)
            self._skip_newlines()
        self._expect(TokenKind.RBRACE, "Expected '}' to close inquire block")
        return InquireExpr(prompt=prompt, agents=agents, ttl=ttl)

    def _parse_resolve(self) -> ResolveStmt:
        """resolve <name> with consensus { threshold: 0.8, strategy: dempster_shafer }"""
        self._expect(TokenKind.RESOLVE)
        target = self._expect(TokenKind.IDENT, "Expected belief name").value
        self._expect(TokenKind.WITH, "Expected 'with' after belief name")
        self._advance()  # consume 'consensus' identifier
        threshold = 0.8
        strategy = "dempster_shafer"
        if self._check(TokenKind.LBRACE):
            self._advance()
            self._skip_newlines()
            while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.RBRACE):
                    break
                key = self._advance().value
                self._expect(TokenKind.COLON)
                if key == "threshold":
                    threshold = float(self._advance().value)
                elif key == "strategy":
                    strategy = self._advance().value
                self._match(TokenKind.COMMA)
                self._skip_newlines()
            self._expect(TokenKind.RBRACE, "Expected '}' to close resolve block")
        self._expect_line_end()
        return ResolveStmt(target=target, threshold=threshold, strategy=strategy)

    def _parse_guard(self) -> GuardStmt:
        """guard <name> against hallucination { max_risk: 0.2, strategy: both }"""
        self._expect(TokenKind.GUARD)
        target = self._expect(TokenKind.IDENT, "Expected belief name").value
        self._expect(TokenKind.AGAINST, "Expected 'against' after belief name")
        self._advance()  # consume 'hallucination' identifier
        max_risk = 0.2
        strategy = "both"
        if self._check(TokenKind.LBRACE):
            self._advance()
            self._skip_newlines()
            while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.RBRACE):
                    break
                key = self._advance().value
                self._expect(TokenKind.COLON)
                if key == "max_risk":
                    max_risk = float(self._advance().value)
                elif key == "strategy":
                    strategy = self._advance().value
                self._match(TokenKind.COMMA)
                self._skip_newlines()
            self._expect(TokenKind.RBRACE, "Expected '}' to close guard block")
        self._expect_line_end()
        return GuardStmt(target=target, max_risk=max_risk, strategy=strategy)

    def _parse_evolve(self) -> EvolveStmt:
        """evolve <name> until stable { max_iter: 3 }"""
        self._expect(TokenKind.EVOLVE)
        target = self._expect(TokenKind.IDENT, "Expected belief name").value
        self._expect(TokenKind.UNTIL, "Expected 'until' after belief name")
        condition = self._advance().value  # e.g. 'stable'
        max_iter = 3
        if self._check(TokenKind.LBRACE):
            self._advance()
            self._skip_newlines()
            while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.RBRACE):
                    break
                key = self._advance().value
                self._expect(TokenKind.COLON)
                if key == "max_iter":
                    max_iter = int(self._advance().value)
                self._match(TokenKind.COMMA)
                self._skip_newlines()
            self._expect(TokenKind.RBRACE, "Expected '}' to close evolve block")
        self._expect_line_end()
        return EvolveStmt(target=target, condition=condition, max_iter=max_iter)

    def _parse_symbol_decl(self) -> SymbolDecl:
        """symbol <name> { ... }"""
        self._expect(TokenKind.SYMBOL)
        name = self._expect(TokenKind.IDENT, "Expected symbol name").value
        self._expect(TokenKind.LBRACE, "Expected '{' after symbol name")
        self._expect_line_end()
        body: list[Statement] = []
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            body.append(self._parse_statement())
            self._skip_newlines()
        self._expect(TokenKind.RBRACE, "Expected '}' to close symbol block")
        self._expect_line_end()
        return SymbolDecl(name=name, body=body)

    # ------------------------------------------------------------------
    # ML constructs
    # ------------------------------------------------------------------

    def _parse_import(self) -> ImportStmt:
        self._expect(TokenKind.IMPORT)
        parts = [self._expect(TokenKind.IDENT, "Expected module name").value]
        while self._check(TokenKind.DOT):
            self._advance(); parts.append(self._advance().value)
        module = ".".join(parts)
        symbols = []
        alias = None
        if self._check(TokenKind.LBRACE):
            self._advance(); self._skip_newlines()
            while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.RBRACE): break
                symbols.append(self._advance().value)
                self._match(TokenKind.COMMA); self._skip_newlines()
            self._expect(TokenKind.RBRACE)
        if self._match(TokenKind.AS_KW):
            alias = self._expect(TokenKind.IDENT).value
        self._expect_line_end()
        return ImportStmt(module=module, symbols=symbols, alias=alias)

    def _parse_model(self) -> ModelDecl:
        self._expect(TokenKind.MODEL)
        name = self._expect(TokenKind.IDENT, "Expected model name").value
        braced = self._match(TokenKind.LBRACE)
        self._expect_line_end()
        terminator = TokenKind.RBRACE if braced else TokenKind.END
        layers = []; forward_fn = None; retrieval = None; uncertainty = "none"; constitution = None; device = "cpu"
        while not self._check(terminator) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(terminator): break
            if self._check(TokenKind.LAYER):
                layers.append(self._parse_layer_decl())
            elif self._check(TokenKind.MOE):
                layers.append(self._parse_moe_layer())
            elif self._check(TokenKind.FORWARD):
                forward_fn = self._parse_forward_fn()
            elif self._check(TokenKind.IDENT) and self._current().value == "retrieval":
                retrieval = self._parse_retrieval_decl()
            elif self._check(TokenKind.IDENT) and self._current().value == "uncertainty":
                self._advance(); self._expect(TokenKind.COLON); uncertainty = self._advance().value; self._expect_line_end()
            elif self._check(TokenKind.IDENT) and self._current().value == "constitution":
                self._advance(); self._expect(TokenKind.COLON); constitution = self._advance().value; self._expect_line_end()
            elif self._check(TokenKind.ON):
                self._advance(); device = self._advance().value; self._expect_line_end()
            else:
                self._advance(); self._expect_line_end()
            self._skip_newlines()
        self._expect(terminator, "Expected '}'" if braced else "Expected 'end'")
        self._expect_line_end()
        return ModelDecl(name=name, layers=layers, forward_fn=forward_fn, retrieval=retrieval, uncertainty=uncertainty, constitution=constitution, device=device)

    def _parse_layer_decl(self) -> LayerDecl:
        self._expect(TokenKind.LAYER)
        name = self._expect(TokenKind.IDENT, "Expected layer name").value
        self._expect(TokenKind.COLON)
        kind = self._advance().value
        in_dim = out_dim = None; config = {}; repeat = 1
        if self._check(TokenKind.LPAREN):
            self._advance()
            if self._check(TokenKind.INT_LIT):
                in_dim = int(self._advance().value)
                if self._match(TokenKind.ARROW):
                    out_dim = int(self._expect(TokenKind.INT_LIT).value)
            self._expect(TokenKind.RPAREN)
        if self._check(TokenKind.LBRACE):
            config = self._parse_kv_block()
        if self._check(TokenKind.STAR):
            self._advance(); repeat = int(self._expect(TokenKind.INT_LIT).value)
        self._expect_line_end()
        return LayerDecl(name=name, kind=kind, in_dim=in_dim, out_dim=out_dim, repeat=repeat, config=config)

    def _parse_moe_layer(self) -> LayerDecl:
        self._expect(TokenKind.MOE)
        name = self._expect(TokenKind.IDENT, "Expected MoE name").value
        config = {}
        if self._check(TokenKind.LBRACE):
            config = self._parse_kv_block()
        self._expect_line_end()
        return LayerDecl(name=name, kind="MoE", in_dim=None, out_dim=None, repeat=1, config=config)

    def _parse_forward_fn(self) -> ForwardFn:
        self._expect(TokenKind.FORWARD)
        self._expect(TokenKind.LPAREN)
        params = []
        while not self._check(TokenKind.RPAREN) and not self._check(TokenKind.EOF):
            pname = self._expect(TokenKind.IDENT).value
            type_ann = None
            if self._match(TokenKind.COLON): type_ann = self._parse_type()
            params.append((pname, type_ann))
            self._match(TokenKind.COMMA)
        self._expect(TokenKind.RPAREN)
        ret_type = None
        if self._match(TokenKind.ARROW): ret_type = self._parse_type()
        if not self._check(TokenKind.LBRACE):
            self._expect_line_end()
            return ForwardFn(params=params, return_type=ret_type, body=[])
        self._expect(TokenKind.LBRACE); self._expect_line_end()
        body = []
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE): break
            body.append(self._parse_statement()); self._skip_newlines()
        self._expect(TokenKind.RBRACE); self._expect_line_end()
        return ForwardFn(params=params, return_type=ret_type, body=body)

    def _parse_train(self) -> TrainStmt:
        self._expect(TokenKind.TRAIN)
        model_name = self._expect(TokenKind.IDENT, "Expected model name").value
        self._expect(TokenKind.ON, "Expected 'on'")
        dataset = self._advance().value
        self._expect(TokenKind.LBRACE); self._expect_line_end()
        optimizer = OptimizerConfig("Adam", {"lr": 0.001})
        loss = LossConfig("CrossEntropy")
        epochs = 10; batch_size = 32; guards = []; evolve = None; precision = "float32"
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE): break
            if self._check(TokenKind.GUARD): guards.append(self._parse_guard()); continue
            if self._check(TokenKind.EVOLVE): evolve = self._parse_evolve_config(); continue
            key = self._advance().value; self._expect(TokenKind.COLON)
            if key == "optimizer":
                opt_name = self._advance().value; opt_params = {}
                if self._check(TokenKind.LBRACE): opt_params = self._parse_kv_block()
                optimizer = OptimizerConfig(opt_name, opt_params)
            elif key == "loss": loss = LossConfig(self._advance().value)
            elif key == "epochs": epochs = int(self._advance().value)
            elif key == "batch_size": batch_size = int(self._advance().value)
            elif key == "precision": precision = self._advance().value
            self._expect_line_end(); self._skip_newlines()
        self._expect(TokenKind.RBRACE); self._expect_line_end()
        return TrainStmt(model_name=model_name, dataset=dataset, optimizer=optimizer, loss=loss, epochs=epochs, batch_size=batch_size, guards=guards, evolve=evolve, precision=precision)

    def _parse_constitution(self) -> ConstitutionDecl:
        self._expect(TokenKind.CONSTITUTION)
        name = self._expect(TokenKind.IDENT, "Expected constitution name").value
        self._expect(TokenKind.LBRACE); self._expect_line_end()
        principles = []; critique_rounds = 2; max_violation = 0.05
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE): break
            if self._check(TokenKind.STRING_LIT):
                principles.append(self._advance().value); self._expect_line_end()
            elif self._check(TokenKind.IDENT):
                key = self._advance().value; self._expect(TokenKind.COLON)
                if key == "critique_rounds": critique_rounds = int(self._advance().value)
                elif key == "max_violation_score": max_violation = float(self._advance().value)
                self._expect_line_end()
            else: self._advance()
            self._skip_newlines()
        self._expect(TokenKind.RBRACE); self._expect_line_end()
        return ConstitutionDecl(name=name, principles=principles, critique_rounds=critique_rounds, max_violation_score=max_violation)

    def _parse_retrieval_decl(self) -> RetrievalDecl:
        self._expect(TokenKind.IDENT, "Expected retrieval")
        config = self._parse_kv_block()
        self._expect_line_end()
        return RetrievalDecl(config=config)

    def _parse_named_config_block(self, node_cls):
        self._advance()
        name = self._expect(TokenKind.IDENT, "Expected declaration name").value
        config = self._parse_kv_block()
        self._expect_line_end()
        return node_cls(name=name, config=config)

    def _parse_evolve_config(self) -> EvolveConfig:
        self._expect(TokenKind.EVOLVE)
        self._advance()  # 'weights'
        self._expect(TokenKind.UNTIL)
        condition = self._advance().value
        config = {}
        if self._check(TokenKind.LBRACE): config = self._parse_kv_block()
        self._expect_line_end()
        return EvolveConfig(metric=config.get("metric", "val_loss"), patience=int(config.get("patience", 5)), max_iter=int(config.get("max_iter", 50)), condition=condition)

    def _parse_kv_block(self) -> dict:
        self._expect(TokenKind.LBRACE); self._skip_newlines()
        result = {}
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE): break
            key = self._advance().value; self._expect(TokenKind.COLON)
            tok = self._current()
            if tok.kind == TokenKind.INT_LIT: result[key] = int(self._advance().value)
            elif tok.kind == TokenKind.FLOAT_LIT: result[key] = float(self._advance().value)
            elif tok.kind == TokenKind.STRING_LIT: result[key] = self._advance().value
            elif tok.kind == TokenKind.BOOL_LIT: result[key] = self._advance().value == "true"
            else: result[key] = self._advance().value
            self._match(TokenKind.COMMA); self._skip_newlines()
        self._expect(TokenKind.RBRACE)
        return result

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._match(TokenKind.OR):
            right = self._parse_and()
            left = BinaryOp(op="or", left=left, right=right)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_equality()
        while self._match(TokenKind.AND):
            right = self._parse_equality()
            left = BinaryOp(op="and", left=left, right=right)
        return left

    def _parse_equality(self) -> Expr:
        left = self._parse_comparison()
        while True:
            if self._match(TokenKind.EQ):
                left = BinaryOp(op="==", left=left, right=self._parse_comparison())
            elif self._match(TokenKind.NEQ):
                left = BinaryOp(op="!=", left=left, right=self._parse_comparison())
            else:
                break
        return left

    def _parse_comparison(self) -> Expr:
        left = self._parse_addition()
        while True:
            if self._match(TokenKind.LT):
                left = BinaryOp(op="<", left=left, right=self._parse_addition())
            elif self._match(TokenKind.GT):
                left = BinaryOp(op=">", left=left, right=self._parse_addition())
            elif self._match(TokenKind.LTE):
                left = BinaryOp(op="<=", left=left, right=self._parse_addition())
            elif self._match(TokenKind.GTE):
                left = BinaryOp(op=">=", left=left, right=self._parse_addition())
            else:
                break
        return left

    def _parse_addition(self) -> Expr:
        left = self._parse_multiplication()
        while True:
            if self._match(TokenKind.PLUS):
                left = BinaryOp(op="+", left=left, right=self._parse_multiplication())
            elif self._match(TokenKind.MINUS):
                left = BinaryOp(op="-", left=left, right=self._parse_multiplication())
            else:
                break
        return left

    def _parse_multiplication(self) -> Expr:
        left = self._parse_unary()
        while True:
            if self._match(TokenKind.STAR):
                left = BinaryOp(op="*", left=left, right=self._parse_unary())
            elif self._match(TokenKind.SLASH):
                left = BinaryOp(op="/", left=left, right=self._parse_unary())
            elif self._match(TokenKind.PERCENT):
                left = BinaryOp(op="%", left=left, right=self._parse_unary())
            else:
                break
        return left

    def _parse_unary(self) -> Expr:
        if self._match(TokenKind.MINUS):
            return UnaryOp(op="-", operand=self._parse_unary())
        if self._match(TokenKind.NOT):
            return UnaryOp(op="not", operand=self._parse_unary())
        return self._parse_call()

    def _parse_call(self) -> Expr:
        expr = self._parse_primary()
        while True:
            if self._match(TokenKind.LPAREN):
                args: list[Expr] = []
                if not self._check(TokenKind.RPAREN):
                    args.append(self._parse_expr())
                    while self._match(TokenKind.COMMA):
                        args.append(self._parse_expr())
                self._expect(TokenKind.RPAREN, "Expected ')' after arguments")
                expr = CallExpr(callee=expr, args=args)
            elif self._match(TokenKind.DOT):
                member = self._expect(TokenKind.IDENT, "Expected member name").value
                expr = MemberExpr(obj=expr, member=member)
            else:
                break
        return expr

    def _parse_primary(self) -> Expr:
        tok = self._current()

        if tok.kind == TokenKind.INT_LIT:
            self._advance()
            return IntLiteral(value=int(tok.value))

        if tok.kind == TokenKind.FLOAT_LIT:
            self._advance()
            return FloatLiteral(value=float(tok.value))

        if tok.kind == TokenKind.STRING_LIT:
            self._advance()
            return StringLiteral(value=tok.value)

        if tok.kind == TokenKind.BOOL_LIT:
            self._advance()
            return BoolLiteral(value=tok.value == "true")

        if tok.kind == TokenKind.IDENT:
            self._advance()
            return Identifier(name=tok.value)

        if tok.kind == TokenKind.LBRACKET:
            return self._parse_list_literal()

        if tok.kind == TokenKind.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokenKind.RPAREN, "Expected ')'")
            return expr

        if tok.kind == TokenKind.IF:
            return self._parse_if_expr()

        if tok.kind == TokenKind.MATCH:
            return self._parse_match_expr()

        # Allow type-constructor-like calls: Confident(...), Explore(...)
        if tok.kind in (TokenKind.CONFIDENT, TokenKind.EXPLORE_TYPE, TokenKind.CONVERGE,
                        TokenKind.PROVISIONAL, TokenKind.EPHEMERAL, TokenKind.PERSISTENT,
                        TokenKind.ABOUT, TokenKind.EXPLORE,
                        TokenKind.TENSOR_KW, TokenKind.VECTORSTORE_KW, TokenKind.CONSTITUTED_KW):
            self._advance()
            return Identifier(name=tok.value)

        raise ParseError(f"Unexpected token in expression", tok)

    def _parse_if_expr(self) -> IfExpr:
        self._expect(TokenKind.IF)
        cond = self._parse_expr()
        self._expect_line_end()
        then_body: list[Statement] = []
        while not self._check(TokenKind.ELSE) and not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.ELSE) or self._check(TokenKind.END):
                break
            then_body.append(self._parse_statement())
        else_body: list[Statement] | None = None
        if self._match(TokenKind.ELSE):
            self._expect_line_end()
            else_body = []
            while not self._check(TokenKind.END) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.END):
                    break
                else_body.append(self._parse_statement())
        self._expect(TokenKind.END, "Expected 'end' to close if")
        self._expect_line_end()
        return IfExpr(condition=cond, then_body=then_body, else_body=else_body)

    def _parse_list_literal(self) -> ListLiteral:
        self._expect(TokenKind.LBRACKET)
        elements: list[Expr] = []
        if not self._check(TokenKind.RBRACKET):
            elements.append(self._parse_expr())
            while self._match(TokenKind.COMMA):
                elements.append(self._parse_expr())
        self._expect(TokenKind.RBRACKET, "Expected ']'")
        return ListLiteral(elements=elements)

    # ------------------------------------------------------------------
    # Types
    # ------------------------------------------------------------------

    _PROB_TYPES = {TokenKind.CONFIDENT, TokenKind.EXPLORE_TYPE, TokenKind.CONVERGE, TokenKind.PROVISIONAL}
    _MEM_TYPES = {TokenKind.EPHEMERAL, TokenKind.PERSISTENT}
    _PRIM_TYPES = {TokenKind.INT_TYPE, TokenKind.FLOAT_TYPE, TokenKind.BOOL_TYPE,
                   TokenKind.TEXT_TYPE, TokenKind.VOID_TYPE}
    _GENERIC_TYPES = {TokenKind.LIST_TYPE, TokenKind.MAP_TYPE, TokenKind.OPTION_TYPE, TokenKind.RESULT_TYPE}

    def _parse_type(self) -> TypeExpr:
        tok = self._current()

        if tok.kind in self._PROB_TYPES:
            wrapper = self._advance().value
            self._expect(TokenKind.LT, f"Expected '<' after {wrapper}")
            inner = self._parse_type()
            self._expect(TokenKind.GT, f"Expected '>' to close {wrapper}<...>")
            return ProbabilisticType(wrapper=wrapper, inner=inner)

        if tok.kind in self._MEM_TYPES:
            scope = self._advance().value
            self._expect(TokenKind.LT, f"Expected '<' after {scope}")
            inner = self._parse_type()
            self._expect(TokenKind.GT, f"Expected '>' to close {scope}<...>")
            return MemoryType(scope=scope, inner=inner)

        if tok.kind in self._PRIM_TYPES:
            self._advance()
            return PrimitiveType(name=tok.value)

        if tok.kind in self._GENERIC_TYPES:
            name = self._advance().value
            self._expect(TokenKind.LT, f"Expected '<' after {name}")
            params: list[TypeExpr] = [self._parse_type()]
            while self._match(TokenKind.COMMA):
                params.append(self._parse_type())
            self._expect(TokenKind.GT, f"Expected '>' to close {name}<...>")
            return GenericType(name=name, params=params)

        if tok.kind == TokenKind.IDENT:
            self._advance()
            return NamedType(name=tok.value)

        if tok.kind == TokenKind.TENSOR_KW:
            self._advance()
            dtype = "Float"
            if self._match(TokenKind.LT):
                dtype = self._advance().value
                self._expect(TokenKind.GT)
            shape = []
            if self._match(TokenKind.LBRACKET):
                while not self._check(TokenKind.RBRACKET) and not self._check(TokenKind.EOF):
                    if self._check(TokenKind.INT_LIT):
                        shape.append(int(self._advance().value))
                    elif self._check(TokenKind.IDENT):
                        shape.append(self._advance().value)
                    else:
                        self._advance(); shape.append(None)
                    self._match(TokenKind.COMMA)
                self._expect(TokenKind.RBRACKET)
            device = "cpu"
            if self._check(TokenKind.ON):
                self._advance(); device = self._advance().value
            causality = None
            privacy = None
            while self._match(TokenKind.AT):
                annotation = self._advance().value
                if annotation == "private":
                    privacy = self._parse_annotation_args()
                else:
                    causality = annotation
            return TensorType(dtype=dtype, shape=shape, device=device, causality=causality, privacy=privacy)
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
        if tok.kind == TokenKind.SPIKETRAIN_KW:
            self._advance()
            dtype = "Float"
            if self._match(TokenKind.LT):
                dtype = self._advance().value
                self._expect(TokenKind.GT)
            neurons = timesteps = None
            if self._match(TokenKind.LBRACKET):
                neurons = self._parse_dim_value()
                if self._match(TokenKind.COMMA):
                    timesteps = self._parse_dim_value()
                self._expect(TokenKind.RBRACKET)
            return SpikeTrainType(dtype=dtype, neurons=neurons, timesteps=timesteps)
        if tok.kind == TokenKind.MULTIMODAL_KW:
            self._advance()
            inner = None
            if self._match(TokenKind.LT):
                inner = self._parse_type()
                self._expect(TokenKind.GT)
            return MultimodalType(inner_type=inner)
        if tok.kind == TokenKind.MEMPTR_KW:
            self._advance()
            inner = None
            if self._match(TokenKind.LT):
                inner = self._parse_type()
                self._expect(TokenKind.GT)
            address = None
            if self._match(TokenKind.LBRACKET):
                address = self._parse_dim_value()
                self._expect(TokenKind.RBRACKET)
            return MemPtrType(inner_type=inner, address=address)
        if tok.kind == TokenKind.CONSTITUTED_KW:
            self._advance()
            inner = None
            if self._match(TokenKind.LT):
                inner = self._parse_type()
                self._expect(TokenKind.GT)
            return ConstitutedType(inner_type=inner)

        raise ParseError("Expected type expression", tok)

    def _parse_dim_value(self) -> int | str | None:
        if self._check(TokenKind.INT_LIT):
            return int(self._advance().value)
        if self._check(TokenKind.IDENT):
            return self._advance().value
        self._advance()
        return None

    def _parse_annotation_args(self) -> dict:
        result = {}
        self._expect(TokenKind.LPAREN)
        while not self._check(TokenKind.RPAREN) and not self._check(TokenKind.EOF):
            key = self._advance().value
            if self._match(TokenKind.ASSIGN) or self._match(TokenKind.COLON):
                tok = self._current()
                if tok.kind == TokenKind.INT_LIT:
                    result[key] = int(self._advance().value)
                elif tok.kind == TokenKind.FLOAT_LIT:
                    result[key] = float(self._advance().value)
                else:
                    result[key] = self._advance().value
            self._match(TokenKind.COMMA)
        self._expect(TokenKind.RPAREN)
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _parse_param_list(self) -> list[Param]:
        params: list[Param] = []
        if self._check(TokenKind.RPAREN):
            return params
        params.append(self._parse_param())
        while self._match(TokenKind.COMMA):
            params.append(self._parse_param())
        return params

    def _parse_param(self) -> Param:
        name = self._expect(TokenKind.IDENT, "Expected parameter name").value
        self._expect(TokenKind.COLON, "Expected ':' after parameter name")
        type_ann = self._parse_type()
        return Param(name=name, type_ann=type_ann)

    def _parse_ident_list_bracket(self) -> list[str]:
        self._expect(TokenKind.LBRACKET, "Expected '['")
        items: list[str] = []
        if not self._check(TokenKind.RBRACKET):
            items.append(self._advance().value)
            while self._match(TokenKind.COMMA):
                items.append(self._advance().value)
        self._expect(TokenKind.RBRACKET, "Expected ']'")
        return items

    def _parse_ident_list_comma(self) -> list[str]:
        items: list[str] = [self._advance().value]
        while self._match(TokenKind.COMMA):
            items.append(self._advance().value)
        return items

    def _parse_quality_chain(self) -> list[str]:
        items: list[str] = [self._advance().value]
        while self._match(TokenKind.GT):
            items.append(self._advance().value)
        return items
