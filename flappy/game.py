import random
from dataclasses import dataclass
import pygame
import numpy as np

from . import config

@dataclass
class Pipe:
    x: float
    gap_y: float  
    passed: bool = False

class FlappyBirdEnv:
    def __init__(self, render: bool = False, seed: int = config.SEED):
        self.width = config.WIDTH
        self.height = config.HEIGHT
        self.fps = config.FPS
        self.clock = None
        self.render_mode = render
        self._rng = random.Random(seed)

        self.bird_x = config.BIRD_X
        self.reset()

        # Init pygame
        pygame.init()
        if self.render_mode:
            # Normal visible game window
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy Bird RL")
            self.clock = pygame.time.Clock()
        else:
            # Headless surface (for Streamlit)
            self.screen = pygame.Surface((self.width, self.height))

    def reset(self):
        """Reset the game environment."""
        self.bird_y = self.height / 2
        self.bird_vy = 0.0
        self.frames = 0
        self.score = 0
        self.pipes = []
        self.spawn_counter = 0
        self.alive = True
        for i in range(3):
            self._spawn_pipe(initial=True, offset=i * 120 + 200)
        return self._get_state()

    def _spawn_pipe(self, initial=False, offset=None):
        """Create a new pipe."""
        gap_center = self._rng.uniform(140, self.height - 140)
        x = self.width + (offset if offset is not None else 0)
        self.pipes.append(Pipe(x=x, gap_y=gap_center, passed=False))

    def _next_pipe(self):
        """Get the next pipe in front of the bird."""
        for p in self.pipes:
            if p.x + config.PIPE_WIDTH >= self.bird_x:
                return p
        return self.pipes[0]

    def _get_state(self):
        """Return the state representation for RL."""
        p = self._next_pipe()
        y = (self.bird_y / self.height) * 2 - 1
        vy = max(-10, min(10, self.bird_vy)) / 10.0
        dx = (p.x - self.bird_x) / self.width 
        top = (p.gap_y - config.PIPE_GAP / 2) / self.height * 2 - 1
        bottom = (p.gap_y + config.PIPE_GAP / 2) / self.height * 2 - 1
        return np.array([y, vy, dx, top, bottom], dtype=np.float32)

    def step(self, action: int):
        """Perform one step in the game."""
        reward = 0.0
        done = False

        # Bird action
        if action == 1:
            self.bird_vy = config.FLAP_IMPULSE
            reward += config.REWARD_FLAP

        # Bird physics
        self.bird_vy += config.GRAVITY
        self.bird_y += self.bird_vy

        # Move pipes
        for p in self.pipes:
            p.x -= config.PIPE_SPEED

        # Spawn new pipes
        self.spawn_counter += 1
        if self.spawn_counter >= config.PIPE_INTERVAL:
            self.spawn_counter = 0
            self._spawn_pipe()

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if p.x + config.PIPE_WIDTH > -5]

        # Check collisions
        if self.bird_y <= 0 or self.bird_y >= self.height:
            done = True

        p = self._next_pipe()
        within_x = (self.bird_x + 12) >= p.x and (self.bird_x - 12) <= p.x + config.PIPE_WIDTH
        gap_top = p.gap_y - config.PIPE_GAP / 2
        gap_bottom = p.gap_y + config.PIPE_GAP / 2
        within_gap = (self.bird_y - 12) >= gap_top and (self.bird_y + 12) <= gap_bottom
        if within_x and not within_gap:
            done = True

        # Scoring
        if p.x + config.PIPE_WIDTH < self.bird_x and not p.passed:
            p.passed = True
            self.score += 1
            reward += config.REWARD_PASS_PIPE

        # Step reward
        reward += config.REWARD_STEP
        if done:
            reward += config.REWARD_DEAD

        self.frames += 1
        if self.frames >= 10000:
            done = True  

        # Render every step
        self.render()

        return self._get_state(), reward, done, {"score": self.score}

    def render(self):
        """Draw the game screen."""
        if self.screen is None:
            return

        # Handle quit events only if visible window
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        # Background
        self.screen.fill((135, 206, 235))  
        pygame.draw.rect(self.screen, (222, 184, 135), (0, self.height - 80, self.width, 80))

        # Pipes
        for p in self.pipes:
            top = p.gap_y - config.PIPE_GAP / 2
            bottom = p.gap_y + config.PIPE_GAP / 2
            pygame.draw.rect(self.screen, (34, 139, 34), (p.x, 0, config.PIPE_WIDTH, top))
            pygame.draw.rect(self.screen, (34, 139, 34), (p.x, bottom, config.PIPE_WIDTH, self.height - bottom))

        # Bird
        pygame.draw.circle(self.screen, (255, 215, 0), (int(self.bird_x), int(self.bird_y)), 12)

        # Flip buffer only if visible window
        if self.render_mode:
            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.fps)

    def close(self):
        if self.render_mode:
            pygame.quit()
