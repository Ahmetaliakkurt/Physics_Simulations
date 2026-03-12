import pygame
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

WIDTH, HEIGHT = 1200, 800
FPS = 90

WHITE = (255, 255, 255)
BLACK = (20, 20, 30)
RED = (255, 50, 50)
BLUE = (50, 150, 255)
GREEN = (0, 255, 100)
DANGER_ZONE_COLOR = (80, 0, 0)

PLANE_SPEED = 6
PLANE_TURN_RATE = 12
MISSILE_SPEED = 8.0
MISSILE_TURN_RATE = 3.5
WALL_MARGIN = 100

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.996
MIN_EPSILON = 0.01

TARGET_LOSS = 0.2
LOSS_WINDOW_SIZE = 50
MAX_EPISODES_LIMIT = 400
MAX_STEPS_PER_EPISODE = 1500

class Plane:
    def __init__(self):
        self.reset()
        self.rect = pygame.Rect(self.x, self.y, 20, 20)

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.angle = 0
        self.speed = PLANE_SPEED

    def move(self, action):
        if action == 0: self.angle -= PLANE_TURN_RATE
        elif action == 2: self.angle += PLANE_TURN_RATE
        
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.rect.center = (self.x, self.y)

    def draw(self, screen):
        rad = math.radians(self.angle)
        nose = (self.x + 20 * math.cos(rad), self.y + 20 * math.sin(rad))
        left = (self.x + 15 * math.cos(rad + 2.5), self.y + 15 * math.sin(rad + 2.5))
        right = (self.x + 15 * math.cos(rad - 2.5), self.y + 15 * math.sin(rad - 2.5))
        pygame.draw.polygon(screen, BLUE, [nose, left, right])

class Missile:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 100
        self.y = 100
        self.angle = 45
        self.speed = MISSILE_SPEED

    def move(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        desired_rad = math.atan2(dy, dx)
        desired_angle = math.degrees(desired_rad)
        
        diff = (desired_angle - self.angle + 180) % 360 - 180
        
        if diff > MISSILE_TURN_RATE:
            self.angle += MISSILE_TURN_RATE
        elif diff < -MISSILE_TURN_RATE:
            self.angle -= MISSILE_TURN_RATE
        else:
            self.angle = desired_angle
            
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), 6)
        rad = math.radians(self.angle)
        tail_x = self.x - 10 * math.cos(rad)
        tail_y = self.y - 10 * math.sin(rad)
        pygame.draw.line(screen, RED, (self.x, self.y), (tail_x, tail_y), 2)

def get_state(plane, missile):
    dx = missile.x - plane.x
    dy = missile.y - plane.y
    dist = math.sqrt(dx**2 + dy**2)
    
    if dist < 150: dist_state = 0
    elif dist < 400: dist_state = 1
    else: dist_state = 2

    rad_to_missile = math.atan2(dy, dx)
    angle_to_missile = math.degrees(rad_to_missile)
    rel_angle = (angle_to_missile - plane.angle) % 360
    angle_state = int(rel_angle // 45)

    is_near_wall = 0
    if (plane.x < WALL_MARGIN or plane.x > WIDTH - WALL_MARGIN or 
        plane.y < WALL_MARGIN or plane.y > HEIGHT - WALL_MARGIN):
        is_near_wall = 1
    
    return (dist_state, angle_state, is_near_wall)

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Dogfight: High Penalty Mode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)

    q_table = np.zeros((3, 8, 2, 3))
    global EPSILON
    
    score_history = []
    loss_history = []
    loss_window = deque(maxlen=LOSS_WINDOW_SIZE) 
    
    episode = 0
    converged = False

    while not converged:
        episode += 1
        plane = Plane()
        missile = Missile()
        
        alive = True
        score = 0
        total_loss = 0
        steps = 0
        
        while alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state = get_state(plane, missile)
            dist_idx, angle_idx, wall_idx = state

            if random.uniform(0, 1) < EPSILON:
                action = random.randint(0, 2)
            else:
                action = np.argmax(q_table[dist_idx, angle_idx, wall_idx])

            plane.move(action)
            missile.move(plane.x, plane.y)
            score += 1
            steps += 1

            next_state = get_state(plane, missile)
            n_dist, n_angle, n_wall = next_state
            
            distance = math.sqrt((plane.x - missile.x)**2 + (plane.y - missile.y)**2)
            
            reward = 0
            
            if plane.x < 10 or plane.x > WIDTH-10 or plane.y < 10 or plane.y > HEIGHT-10:
                reward = -5000
                alive = False
            elif distance < 20:
                reward = -5000
                alive = False
            elif steps >= MAX_STEPS_PER_EPISODE:
                reward = 5000
                alive = False 
            else:
                reward += 1
                
                if wall_idx == 1: reward -= 50
                
                if dist_idx == 0 and action != 1:
                    reward += 10

            old_val = q_table[dist_idx, angle_idx, wall_idx, action]
            if not alive:
                next_max = 0
            else:
                next_max = np.max(q_table[n_dist, n_angle, n_wall])
            
            target = reward + DISCOUNT_FACTOR * next_max
            td_error = target - old_val
            total_loss += abs(td_error)
            
            new_val = old_val + LEARNING_RATE * td_error
            q_table[dist_idx, angle_idx, wall_idx, action] = new_val

            if episode % 1 == 0: 
                screen.fill(BLACK)
                pygame.draw.rect(screen, DANGER_ZONE_COLOR, (0, 0, WIDTH, HEIGHT), 5) 
                pygame.draw.rect(screen, GREEN, (WALL_MARGIN, WALL_MARGIN, WIDTH-2*WALL_MARGIN, HEIGHT-2*WALL_MARGIN), 1)
                plane.draw(screen)
                missile.draw(screen)
                
                avg_loss_disp = sum(loss_window)/len(loss_window) if len(loss_window) > 0 else 0
                info = f"Ep: {episode} | Score: {score} | Loss: {avg_loss_disp:.2f} | Eps: {EPSILON:.2f}"
                screen.blit(font.render(info, True, WHITE), (15, 15))
                pygame.display.update()
                clock.tick(FPS)         
        avg_loss = total_loss / steps if steps > 0 else 0
        loss_window.append(avg_loss)
        score_history.append(score)
        loss_history.append(avg_loss)

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY

        if len(loss_window) == LOSS_WINDOW_SIZE:
            current_avg_loss = sum(loss_window) / LOSS_WINDOW_SIZE
            if current_avg_loss < TARGET_LOSS:
                print(f"--- CONVERGED at Episode {episode}! ---")
                converged = True
        
        if episode >= MAX_EPISODES_LIMIT:
            converged = True

    pygame.quit()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, color='orange', alpha=0.6, label='Loss')
    if len(loss_history) >= LOSS_WINDOW_SIZE:
        ma = np.convolve(loss_history, np.ones(LOSS_WINDOW_SIZE)/LOSS_WINDOW_SIZE, mode='valid')
        plt.plot(range(LOSS_WINDOW_SIZE-1, len(loss_history)), ma, color='red', label='Moving Avg')
    plt.legend(); plt.grid(True); plt.title("Loss Convergence")

    plt.subplot(1, 2, 2)
    plt.plot(score_history)
    plt.title("Survival Score")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_simulation()