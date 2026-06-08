import pytest

from medgraphia.llm.gateway import CompletionRequest, LiteLLMGateway, LLMProvider


@pytest.mark.asyncio
async def test_gateway_model_id_prefixing():
    """Verify that different providers get the correct litellm prefix."""
    # Anthropic
    gw_ant = LiteLLMGateway(LLMProvider.ANTHROPIC, "claude-3-5-sonnet")
    assert gw_ant._model_id == "anthropic/claude-3-5-sonnet"

    # DeepSeek
    gw_ds = LiteLLMGateway(LLMProvider.DEEPSEEK, "deepseek-chat")
    assert gw_ds._model_id == "deepseek/deepseek-chat"

    # Gemini
    gw_gem = LiteLLMGateway(LLMProvider.GEMINI, "gemini-1.5-pro")
    assert gw_gem._model_id == "gemini/gemini-1.5-pro"

    # OpenAI (No prefix)
    gw_oa = LiteLLMGateway(LLMProvider.OPENAI, "gpt-4o")
    assert gw_oa._model_id == "gpt-4o"


@pytest.mark.asyncio
async def test_gateway_model_id_override_prefixing():
    """Verify that per-request model_id overrides also get prefixed correctly."""
    gw = LiteLLMGateway(LLMProvider.ANTHROPIC, "claude-3-haiku")

    # Override with another anthropic model
    req = CompletionRequest(
        system_prompt="sys", user_prompt="user", model_id="claude-3-5-sonnet", mock_response="hi"
    )

    resp = await gw.acomplete(req)
    # The internal logic should have prefixed it
    assert resp.model_used == "anthropic/claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_gateway_mock_completion():
    """Verify that mock_response bypasses actual API calls."""
    gw = LiteLLMGateway(LLMProvider.OPENAI, "gpt-4o")

    req = CompletionRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello World'",
        mock_response="Hello World",
    )

    resp = await gw.acomplete(req)
    assert resp.ok
    assert resp.text == "Hello World"
    assert resp.model_used == "gpt-4o"


@pytest.mark.asyncio
async def test_gateway_astream_mock():
    """Verify that streaming also works with mock_response."""
    gw = LiteLLMGateway(LLMProvider.OLLAMA, "qwen2")

    req = CompletionRequest(
        system_prompt="sys", user_prompt="user", mock_response="Streamed response", stream=True
    )

    chunks = []
    async for chunk in gw.astream(req):
        chunks.append(chunk)

    full_text = "".join(chunks)
    assert full_text == "Streamed response"


@pytest.mark.asyncio
async def test_gateway_json_parsing():
    """Test the robust JSON parsing helper."""
    from medgraphia.llm.gateway import _parse_json_safe

    # Case 1: Markdown fence
    raw_1 = 'Here is the result: ```json\n{"status": "ok", "count": 5}\n```'
    parsed_1 = _parse_json_safe(raw_1)
    assert parsed_1["status"] == "ok"
    assert parsed_1["count"] == 5

    # Case 2: Trailing comma (common LLM error)
    raw_2 = '{"name": "test",}'
    parsed_2 = _parse_json_safe(raw_2)
    assert parsed_2["name"] == "test"
