import argparse
import torch
import numpy as np
from .game import FlappyBirdEnv
from .dqn import QNetwork
from . import config

def play():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best.pt')
    args = parser.parse_args()

    env = FlappyBirdEnv(render=True)
    state_dim = 5
    action_dim = 2

    device = 'cuda' if config.DEVICE=='cuda' and torch.cuda.is_available() else 'cpu'
    net = QNetwork(state_dim, action_dim).to(device)

    try:
        ckpt = torch.load(args.model, map_location=device)
        net.load_state_dict(ckpt['model'])
        print(f"Loaded model from {args.model}")
    except Exception as e:
        print(f"Warning: could not load model '{args.model}': {e}. Using random weights.")

    state = env.reset()
    done = False
    while True:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = net(s)
            action = int(torch.argmax(q, dim=1).item())
        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()

if __name__ == '__main__':
    play()
