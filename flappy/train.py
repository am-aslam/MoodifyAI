
import os
import time
import signal
import argparse
import numpy as np
import torch
from .game import FlappyBirdEnv
from .dqn import DQNAgent
from . import config

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=config.EPISODES)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=config.SEED)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = FlappyBirdEnv(render=False, seed=args.seed)
    state_dim = 5
    action_dim = 2

    agent = DQNAgent(state_dim, action_dim)
    best_score = -1
    total_steps = 0
    returns_window = []

    def handle_sigint(signum, frame):
        ckpt_path = os.path.join(args.save_dir, 'ckpt_last.pt')
        torch.save({'model': agent.online.state_dict(), 'steps': agent.steps_done}, ckpt_path)
        print(f"\nSaved last checkpoint to {ckpt_path}. Exiting...")
        env.close()
        exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    for ep in range(1, args.episodes+1):
        state = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        while not done and steps < config.MAX_STEPS:
            eps = agent.compute_eps()
            action = agent.select_action(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, float(done))
            loss = agent.optimize()
            state = next_state
            ep_return += reward
            steps += 1
            total_steps += 1

            if total_steps % config.SAVE_EVERY == 0:
                ckpt_path = os.path.join(args.save_dir, f'ckpt_{total_steps}.pt')
                torch.save({'model': agent.online.state_dict(), 'steps': total_steps}, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        returns_window.append(ep_return)
        if len(returns_window) > 50:
            returns_window.pop(0)
        avg_return = sum(returns_window) / len(returns_window)

        if info.get('score', 0) > best_score:
            best_score = info.get('score', 0)
            best_path = os.path.join(args.save_dir, 'best.pt')
            torch.save({'model': agent.online.state_dict(), 'steps': total_steps}, best_path)

        print(f"Episode {ep:4d} | steps={steps:4d} | return={ep_return:8.1f} | avg50={avg_return:8.1f} | score={info.get('score',0):3d} | eps={eps:0.3f}")

    last_path = os.path.join(args.save_dir, 'ckpt_last.pt')
    torch.save({'model': agent.online.state_dict(), 'steps': total_steps}, last_path)
    print(f"Training complete. Last checkpoint: {last_path}")
    env.close()

if __name__ == '__main__':
    train()
