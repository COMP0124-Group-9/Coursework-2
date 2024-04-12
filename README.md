# Coursework-2

Produced for COMP0124 Multi-agent Artificial Intelligence for an investigation into the beahviour of two MARL techniques in the 'Atari Warlords' game environment.

All required dependencies are listed in `requirements.txt`

Warlords ROM can be acquired using AutoROM by running the `AutoROM` command in the root directory once dependencies are installed.

Training for Q-Learning is performed by running ``main.py``. Agents can be switched between `Agent` (an agent to be trained on next run), `RandomAgent` (an agent choosing actions randomly), and `TrainedAgent` (an agent which uses the last trained model but performs no model updates) by changing the class specifier in line 2 of ``def main()`` (line 34 overall in file).

To switch to MADPPG implementation, switch `game.run()` to `game.run_madppg()` in line 52 of `main.py`.

To adjust Q-Learning parameters, change settings in the `Agent` class constructor in `Agent.py`.

To adjust MADPPG parameters, change settings in the `Game` class constructor in `Game.py`.
