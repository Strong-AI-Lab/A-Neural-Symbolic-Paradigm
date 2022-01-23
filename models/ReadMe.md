## Gate Attention
We add the Gate Attention here instead of the dot product.
Here is the code that we updated. You can check the code from the `ima_gate3.py` and `ima_glove_gate3.py`. Also there is an another updating for the gate attention in the `ima_glove_gate4.py` which has a much lower initial loss compared to the `ima_glove_gate3.py`.
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

The original code for dot product is only one line.
```
# Unify every rule and weighted sum based on attention
    new_states = unifier(embedded_ctx_preds, initial_state=[state])
# (?, rules, dim)
    state = dot11([sim_vec, new_states])
```


Here is our model architecture.

<img width="800" alt="IMA_GloVe_GA" src="https://user-images.githubusercontent.com/23516191/147908506-05866a83-b3a5-49fb-add5-164007776727.PNG">
