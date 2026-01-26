from pathlib import Path


PROMPT_DIR = Path(__file__).parent / "prompts"


def get_prompt(name: str) -> str:
    """Get prompt from the agent prompt dir"""
    prompt_path = PROMPT_DIR / f"{name}.md"
    with open(prompt_path, "r") as file:
        template = file.read()

    return template
