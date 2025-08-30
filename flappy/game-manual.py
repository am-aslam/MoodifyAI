import pygame
from flappy.game import FlappyBirdEnv

if __name__ == "__main__":
    env = FlappyBirdEnv(render=True)

    state = env.reset()
    done = False
    while True:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1

        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()
