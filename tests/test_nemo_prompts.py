from pathlib import Path

import yaml
from nemoguardrails import RailsConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEMO_CONFIG_DIR = PROJECT_ROOT / "configs" / "nemo"

PROMPTS_PATH = NEMO_CONFIG_DIR / "prompts.yml"


def load_prompts() -> list[dict]:
    with PROMPTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    assert isinstance(data, dict)
    assert isinstance(
        data.get("prompts"),
        list,
    )

    return data["prompts"]


def get_prompt(
    task_name: str,
) -> dict:

    prompts = load_prompts()

    for prompt in prompts:
        if prompt.get("task") == task_name:
            return prompt

    raise AssertionError(f"Prompt для task=" f"{task_name!r} не найден.")


def test_nemo_config_loads_successfully() -> None:
    config = RailsConfig.from_path(str(NEMO_CONFIG_DIR))

    assert config is not None


def test_self_check_input_uses_chat_messages() -> None:
    prompt = get_prompt("self_check_input")

    assert "content" not in prompt

    messages = prompt["messages"]

    assert len(messages) == 2

    assert messages[0]["type"] == "system"

    assert messages[1]["type"] == "user"

    system_content = messages[0]["content"]

    user_content = messages[1]["content"]

    assert user_content.strip() == "{{ user_input }}"

    assert "prompt injection" in system_content.casefold()

    assert "при сомнении разрешай запрос" in system_content.casefold()

    assert prompt["max_tokens"] == 4


def test_self_check_output_uses_chat_messages() -> None:
    prompt = get_prompt("self_check_output")

    assert "content" not in prompt

    messages = prompt["messages"]

    assert len(messages) == 2

    assert messages[0]["type"] == "system"

    assert messages[1]["type"] == "user"

    user_content = messages[1]["content"]

    assert "{{ user_input }}" in user_content

    assert "{{ bot_response }}" in user_content

    assert prompt["max_tokens"] == 4
