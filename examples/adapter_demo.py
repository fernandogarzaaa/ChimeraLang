# Demo: ClaudeConstraintMiddleware adapter for ChimeraLang
# This file demonstrates the use of ClaudeConstraintMiddleware with all detection strategies and consensus patterns.

from chimera.claude_adapter import ClaudeConstraintMiddleware, ToolCallSpec

def dummy_tool_1(query):
    return f"Result for: {query}"

def dummy_tool_2(query):
    return f"Result for: {query} (alt)"

def always_none(query):
    return None

def always_number(query):
    return 42

def forbidden_tool(query):
    return "personal_data: forbidden info"

if __name__ == "__main__":
    middleware = ClaudeConstraintMiddleware(confidence_threshold=0.85)

    # Confidence threshold test
    spec = ToolCallSpec(
        tool_name="dummy_tool_1",
        min_confidence=0.9,
        output_must=[lambda x: x is not None],
        output_forbidden=["personal_data"],
        detect_strategy="confidence_threshold",
        detect_threshold=0.8,
    )
    print(middleware.call(spec, tool_fn=dummy_tool_1, query="quantum"))

    # Range detection
    spec2 = ToolCallSpec(
        tool_name="always_number",
        detect_strategy="range",
        valid_range=(10, 50),
    )
    print(middleware.call(spec2, tool_fn=always_number, query="test"))

    # Semantic detection
    spec3 = ToolCallSpec(
        tool_name="forbidden_tool",
        detect_strategy="semantic",
        forbidden_patterns=["personal_data"],
    )
    print(middleware.call(spec3, tool_fn=forbidden_tool, query="test"))

    # Consensus collapse
    spec4 = ToolCallSpec(tool_name="dummy_tool_1")
    print(middleware.consensus_call(spec4, [dummy_tool_1, dummy_tool_2], query="consensus"))
