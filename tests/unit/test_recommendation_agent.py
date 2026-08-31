from support_troubleshooting_agent.agents._resilience import invoke_with_retry
from support_troubleshooting_agent.agents.log_analysis import log_agent
from support_troubleshooting_agent.agents.rag_knowledge import rag_agent
from support_troubleshooting_agent.agents.recommendation import recommendation_agent


class _RetryingModel:
    def __init__(self):
        self.calls = 0

    def __ror__(self, other):
        return self

    def invoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 rate limit exceeded")
        return type("Resp", (), {"content": '{"ok": true}'})()


def test_invoke_with_retry_retries_transient_failures():
    model = _RetryingModel()
    prompt = object()

    result = invoke_with_retry(model, prompt, {"test": "payload"}, "sample")

    assert result.content == '{"ok": true}'
    assert model.calls == 2


def test_log_agent_handles_missing_log_files():
    result = log_agent({})

    assert result["current_step"] == "log_agent"
    assert result["log_analysis"]["summary"] == "No log data was provided."
    assert result["errors"][0]["user_message"]


def test_rag_agent_handles_empty_retrieval_results():
    result = rag_agent({"ticket_summary": "checkout is failing"}, retriever_fn=lambda query: [])

    assert result["current_step"] == "rag_agent"
    assert result["retrieved_documents"] == []
    assert result["errors"][0]["user_message"]


def test_recommendation_agent_returns_structured_json():
    state = {
        "root_cause": {
            "primary_cause": "Database latency regression",
            "confidence": 0.8,
            "supporting_evidence": ["slow query 821ms", "HTTP 504s during checkout"],
        }
    }

    result = recommendation_agent(state)

    assert "recommendations" in result
    assert result["current_step"] == "recommendation_agent"

    recommendations = result["recommendations"]
    assert isinstance(recommendations, dict)

    for key in [
        "recommended_actions",
        "validation_steps",
        "escalation_guidance",
        "preventative_measures",
    ]:
        assert key in recommendations
        assert isinstance(recommendations[key], list)
        assert recommendations[key]
