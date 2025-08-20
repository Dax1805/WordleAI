BaseSolver: properties id, name, version; method next_guess(state) -> str.

state = { turn, history: [(guess, pattern)], candidates, allowed, N, rng }

Initial solver IDs you’ll implement:

random_consistent

letter_freq

entropy
