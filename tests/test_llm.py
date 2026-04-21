from __future__ import annotations

from arietta_voice.config import ModelConfig
from arietta_voice.llm import BedrockChatModel, GemmaChatModel, _bedrock_messages, build_chat_model


def test_build_chat_model_selects_provider() -> None:
    local = build_chat_model(ModelConfig(chat_provider="local", chat_model="mlx-community/gemma-4"))
    bedrock = build_chat_model(
        ModelConfig(
            chat_provider="bedrock",
            chat_model="us.anthropic.claude-sonnet-4-6",
            bedrock_region="us-west-2",
        )
    )

    assert isinstance(local, GemmaChatModel)
    assert isinstance(bedrock, BedrockChatModel)


def test_bedrock_messages_split_system_and_chat_roles() -> None:
    system, messages = _bedrock_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    )

    assert system == [{"text": "You are helpful."}]
    assert messages == [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]


def test_bedrock_chat_model_generates_text_from_converse() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.request = None

        def converse(self, **kwargs):
            self.request = kwargs
            return {
                "output": {
                    "message": {
                        "content": [
                            {"text": "Hello from Bedrock."},
                        ]
                    }
                }
            }

    fake_client = FakeClient()
    model = BedrockChatModel(
        "us.anthropic.claude-sonnet-4-6",
        region_name="us-west-2",
        client_factory=lambda: fake_client,
    )

    response = model.generate(
        [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=111,
        temperature=0.25,
    )

    assert response == "Hello from Bedrock."
    assert fake_client.request is not None
    assert fake_client.request["modelId"] == "us.anthropic.claude-sonnet-4-6"
    assert fake_client.request["system"] == [{"text": "System prompt"}]
    assert fake_client.request["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]
    assert fake_client.request["inferenceConfig"] == {"temperature": 0.25, "maxTokens": 111}
