from __future__ import annotations


def render_prompt_template(template: str, **kwargs: str) -> str:
    return str(template).format(**kwargs)
