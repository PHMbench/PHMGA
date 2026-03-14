"""Backward-compatible prompt exports.

`templates.py` is no longer the formal prompt system. The real contracts live in
the dedicated prompt modules. This wrapper exists only so older imports fail
softly while the repo migrates to the new prompt layout.
"""

from .execute_prompt import EXECUTE_PROMPT_TEMPLATE as EXECUTOR_PROMPT
from .plan_prompt import PLAN_PROMPT_TEMPLATE as PLANNER_PROMPT
from .reflect_prompt import REFLECT_PROMPT_TEMPLATE as REFLECTOR_PROMPT
from .report_prompt import REPORT_PROMPT_TEMPLATE as REPORT_PROMPT
