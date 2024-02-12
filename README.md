## Mon, Feb 12, 2024
Just ran ` python run.py --env_mode delay --seed seed --train_len 100000` for seed=0,...,9.

Produced pretty good results. Tricks that seemd to work:
- Emulate trading-bot (0.001 step size, frequent updates, only apply cost of buying during selling, scaling down rewards by 0.01)
- Avoided some things from trading-bot (did not take different and apply sigmoid. Instead just showed raw values -- no scaling -- and some hidden data points)
- Oddly, `--env_mode qlearn` works if you change the reward by negative (alg'm seems to accidently do reward minimization?)
