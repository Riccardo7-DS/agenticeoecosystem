FROM /shared/home/rdercole/Riccardo/projects/agenticeoecosystem/models/eve

PARAMETER temperature 0.7

SYSTEM """
$(cat SYSTEM_PROMPT.tt)
"""
