score(guess, answer) -> pattern where pattern is 'G'|'Y'|'-' repeated N.

filter_candidates(history, words, N) -> list[str]

validate_guess(word, allowed, N) -> bool

max_turns default (6 for N=5; document how it scales if you choose to vary it).

Determinism: same inputs → same outputs; no hidden state.