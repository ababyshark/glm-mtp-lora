"""V10 MTP fill helpers — thin re-export. The actual transforms live in
v10_fill.py (swap_eh_halves + _fill_mtp_layer); this module keeps the
VERIFY-AT-SERVING-ALIGNMENT note discoverable and preserves the import path
used by tests (test_v10_mtp.py).

The eh_proj column-order question: mcore _concat_embeddings cats
[enorm(emb) | hnorm(hidden)] = [e | h] (multi_token_prediction.py:935-946);
the GLM/DSv3 checkpoint eh_proj expects [h | e]. swap_eh_halves restores
mcore's order at fill time. A wrong swap leaves the MAIN head intact and the
draft head garbage — the MTP CE check must be compared against serving before
shipping (mapping §5 marks the GLM order [推断]).
"""
from v10_fill import swap_eh_halves  # noqa: F401  (canonical impl)
