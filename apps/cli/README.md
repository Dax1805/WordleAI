wordleai run --solver entropy --N 5 --sample 1000 --seed 42 --guess-space allowed

wordleai sweep --solvers letter_freq,entropy --N 5,6 --sample 500 --seed 123

wordleai first-guess --solver entropy --N 5 --top 20 --guess-space allowed