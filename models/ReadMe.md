## Gate Attention
We add the gate attention at here instead of the dot product.
Here is the code that we updated. You can check the code from the `ima_gate3.py` and `ima_glove_gate3.py`.
```
# Unify every rule and weighted sum based on attention
    new_states = unifier(embedded_ctx_preds, initial_state=[state])
# (?, rules, dim)
    new_state = dot11([sim_vec, new_states])
# Apply gating
    gate = gating(new_state)
    outs.append(gate)
    new_state = gate2([new_state, state, gate])
    state = new_state
```

The original code for product is only one line.
```
# Unify every rule and weighted sum based on attention
    new_states = unifier(embedded_ctx_preds, initial_state=[state])
# (?, rules, dim)
    state = dot11([sim_vec, new_states])
```
