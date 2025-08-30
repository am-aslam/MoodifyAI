# AI Flappy Bird — DQN Agent (PyTorch)

This repo implements a **Deep Q-Network (DQN)** agent that learns to play a simplified **Flappy Bird** built with `pygame`.
It uses feature-based state (no raw pixels) for fast training.

## 🔧 Tech
- Python 3.10+
- PyTorch
- pygame
- numpy
- matplotlib (for optional reward plots)

## 📦 Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## ▶️ Train
```bash
python -m flappy.train
```
- Checkpoints are saved to `models/`.
- Training logs are written to `runs/`.
- Use `CTRL+C` to stop; the latest checkpoint is preserved.

### Key hyperparameters
Edit `flappy/config.py`:
- `EPISODES`, `MAX_STEPS`
- `LR`, `GAMMA`, `BATCH_SIZE`, `EPS_START`, `EPS_END`, `EPS_DECAY`
- `TARGET_UPDATE`

## 🎮 Watch the Trained Agent
```bash
python -m flappy.play --model models/best.pt
```
You can also specify a different checkpoint:
```bash
python -m flappy.play --model models/ckpt_last.pt
```

## 🧠 State Representation
State vector (normalized):
1. Bird y position
2. Bird vertical velocity
3. Horizontal distance to next pipe
4. Next pipe top y
5. Next pipe bottom y

## 🏆 Rewards
- +1 per step survived
- +100 for passing a pipe
- -100 on collision
- -0.05 per flap (to discourage spamming)

## 🗂️ Structure
```
ai-flappy-dqn/
├── flappy/
│   ├── __init__.py
│   ├── config.py
│   ├── dqn.py
│   ├── game.py
│   ├── train.py
│   ├── play.py
│   └── utils.py
├── models/
├── runs/
├── requirements.txt
└── README.md
```

## ❗ Notes
- Training speed depends on CPU/GPU. Pixel-based RL is intentionally avoided for practicality.
- You can tweak physics or visuals in `game.py` (gravity, flap impulse, pipe gap, spawn rate).
- This is a learning-focused implementation; feel free to optimize further.
