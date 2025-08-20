run_id, solver, N, answer, success, guesses, time_ms, guess_1, patt_1, ... guess_6, patt_6, wordlist_answers_sha, wordlist_allowed_sha, seed
And a manifest JSON saved next to the CSV:

{ run_id, timestamp, git_commit, config, wordlists: {N: {answers_sha, allowed_sha}}, sample_size }