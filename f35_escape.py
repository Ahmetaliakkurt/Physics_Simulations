import pygame
import math
import random
import numpy as np

# --- AYARLAR ---
WIDTH, HEIGHT = 1400, 900
FPS = 60

# Renkler
WHITE = (255, 255, 255)
BLACK = (10, 10, 20)
RED = (255, 0, 0)      # Kilitli
GRAY = (80, 80, 80)    # Arayışta
BLUE = (70, 130, 180)
ORANGE = (255, 165, 0)
GREEN = (0, 255, 0)

# Fizik
MAX_SPEED = 6.0
ACCELERATION = 0.2
NORMAL_TURN = 5
HIGH_G_TURN = 20    # Çok sert dönüş (Ani kırılma)

MISSILE_SPEED_SEARCH = 4.0
MISSILE_SPEED_LOCKED = 9.0  # Çok hızlı
MISSILE_TURN_RATE = 3.5     # KRİTİK: Füze artık sınırlı dönebiliyor (Hantallaştı)

LOCK_ON_RANGE = 400

# RL (Q-Learning)
# State arttığı için tabloyu büyütmemiz lazım
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99 # Geleceğe daha çok önem versin
EPSILON = 1.0
EPSILON_DECAY = 0.998 
MIN_EPSILON = 0.01

class F35:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.angle = 0 
        self.speed = MAX_SPEED

    def move(self, action):
        # 0:Sol, 1:Düz, 2:Sağ, 3:High-G Break
        current_turn = NORMAL_TURN
        
        # Enerji Yönetimi (Hızlanma)
        if self.speed < MAX_SPEED:
            self.speed += ACCELERATION

        if action == 0:
            self.angle -= current_turn
        elif action == 2:
            self.angle += current_turn
        elif action == 3: # High-G
            # Eğer yeterince hızlıysa manevrayı yap
            if self.speed > 3.0:
                self.angle += HIGH_G_TURN # Ani dönüş
                self.speed = 2.5 # Hız (Enerji) kaybı
            else:
                # Enerji yoksa normal dön
                self.angle += current_turn

        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        
        # Clamp (Duvardan çıkamasın ama duvara çarpınca ölsün kontrolü dışarıda)
        # Görsel taşmayı engellemek için clamp:
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

    def draw(self, screen):
        rad = math.radians(self.angle)
        nose = (self.x + 20 * math.cos(rad), self.y + 20 * math.sin(rad))
        left = (self.x + 15 * math.cos(rad + 2.4), self.y + 15 * math.sin(rad + 2.4))
        right = (self.x + 15 * math.cos(rad - 2.4), self.y + 15 * math.sin(rad - 2.4))
        
        color = BLUE if self.speed > 3.0 else ORANGE
        pygame.draw.polygon(screen, color, [nose, left, right])

class SmartMissile:
    def __init__(self):
        self.respawn()
        self.locked = False
        self.angle = 0 # Radyan değil Derece tutalım kolaylık için

    def respawn(self):
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top': self.x, self.y = random.randint(0, WIDTH), 0
        elif side == 'bottom': self.x, self.y = random.randint(0, WIDTH), HEIGHT
        elif side == 'left': self.x, self.y = 0, random.randint(0, HEIGHT)
        elif side == 'right': self.x, self.y = WIDTH, random.randint(0, HEIGHT)
        self.speed = MISSILE_SPEED_SEARCH
        self.angle = random.randint(0, 360)

    def move(self, target):
        dx = target.x - self.x
        dy = target.y - self.y
        dist = math.sqrt(dx**2 + dy**2)

        # 1. Kilitlenme Kontrolü
        if dist < LOCK_ON_RANGE:
            self.locked = True
            self.speed = MISSILE_SPEED_LOCKED
            target_x, target_y = target.x, target.y
        else:
            self.locked = False
            self.speed = MISSILE_SPEED_SEARCH
            target_x, target_y = target.x, target.y # Yine de takip etsin ama yavaş

        # 2. Güdüm (DÖNÜŞ LİMİTLİ)
        desired_rad = math.atan2(target_y - self.y, target_x - self.x)
        desired_angle = math.degrees(desired_rad)
        
        # Açılar arası farkı bul (-180 ile 180 arası)
        diff = (desired_angle - self.angle + 180) % 360 - 180
        
        # Dönüşü sınırla (Inertia)
        if diff > MISSILE_TURN_RATE:
            self.angle += MISSILE_TURN_RATE
        elif diff < -MISSILE_TURN_RATE:
            self.angle -= MISSILE_TURN_RATE
        else:
            self.angle += diff # Küçük fark varsa tam dön

        # Hareket
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

    def draw(self, screen):
        color = RED if self.locked else GRAY
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 6)

def get_state(plane, missile):
    # 1. Mesafe
    dx = missile.x - plane.x
    dy = missile.y - plane.y
    dist = math.sqrt(dx**2 + dy**2)
    
    if dist < 150: dist_idx = 0      # ÇOK YAKIN (Dodge Zamanı)
    elif dist < LOCK_ON_RANGE: dist_idx = 1 # KİLİTLİ
    else: dist_idx = 2               # UZAK

    # 2. Füzenin Konumu (Bana göre nerede?)
    rad_to_missile = math.atan2(dy, dx)
    angle_to_missile = math.degrees(rad_to_missile)
    rel_angle = (angle_to_missile - plane.angle) % 360
    pos_angle_idx = int(rel_angle // 90) # 4 Yön (Ön, Sağ, Arka, Sol) - Basitleştirdik

    # 3. Füzenin Bakış Açısı (YENİ - ÇOK ÖNEMLİ)
    # Füze bana mı bakıyor, yoksa başka yere mi?
    # Füzenin açısı ile 'füzeden uçağa olan açı' farkı
    rad_missile_to_plane = math.atan2(-dy, -dx)
    deg_missile_to_plane = math.degrees(rad_missile_to_plane)
    
    missile_heading_diff = (missile.angle - deg_missile_to_plane + 180) % 360 - 180
    
    # Eğer fark azsa füze tam üzerime geliyor demektir.
    if abs(missile_heading_diff) < 20:
        heading_idx = 0 # TEHLİKE: Üzerime kitli
    else:
        heading_idx = 1 # OVERSHOOT: Yanımdan/Arkamdan geçiyor (Fırsat)

    return (dist_idx, pos_angle_idx, heading_idx)

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("F-35 Evasion: Turn Limit & Heading Awareness")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    # Q-Table: [Mesafe(3)][Konum(4)][Heading(2)][Action(4)]
    # Toplam 96 state - Yönetilebilir ve hızlı öğrenir
    q_table = np.zeros((3, 4, 2, 4))

    global EPSILON
    episode = 0
    high_score = 0

    running = True
    while running:
        plane = F35()
        missile = SmartMissile()
        alive = True
        score = 0
        episode += 1
        
        while alive and running:
            state = get_state(plane, missile)
            d_idx, p_idx, h_idx = state

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            if random.uniform(0, 1) < EPSILON:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[d_idx, p_idx, h_idx])

            plane.move(action)
            missile.move(plane)
            score += 1

            # Next State
            next_state = get_state(plane, missile)
            nd_idx, np_idx, nh_idx = next_state
            
            # Gerçek Mesafe
            dist = math.sqrt((plane.x - missile.x)**2 + (plane.y - missile.y)**2)

            # --- ÖDÜL SİSTEMİ ---
            reward = 1 # Yaşama ödülü

            # 1. Kilitlenme Cezası (Agresifleşmesi için)
            if missile.locked:
                reward -= 2 # Kilitli kalmak kötü, kurtulmaya çalış!

            # 2. OVERSHOOT ÖDÜLÜ (YENİ)
            # Füze yakınken (d_idx 0 veya 1) ve füze bana bakmıyorsa (h_idx 1)
            # Bu demektir ki füzeyi boşa düşürdüm!
            if (d_idx == 0 or d_idx == 1) and h_idx == 1:
                reward += 10 # Harika hareket!

            # 3. Ölüm ve Duvar
            if dist < 20:
                reward = -1000
                alive = False
            if plane.x < 10 or plane.x > WIDTH-10 or plane.y < 10 or plane.y > HEIGHT-10:
                reward = -1000
                alive = False

            # Q Update
            old_val = q_table[d_idx, p_idx, h_idx, action]
            next_max = 0 if not alive else np.max(q_table[nd_idx, np_idx, nh_idx])
            
            q_table[d_idx, p_idx, h_idx, action] = (1 - LEARNING_RATE) * old_val + \
                                                   LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)

            # --- ÇİZİM ---
            screen.fill(BLACK)
            
            # Kilitlenme Alanı
            if missile.locked:
                pygame.draw.circle(screen, (30, 0, 0), (int(plane.x), int(plane.y)), LOCK_ON_RANGE, 1)
            
            plane.draw(screen)
            missile.draw(screen)
            
            # Bilgi
            info = f"EP: {episode} | SCORE: {score} | EPS: {EPSILON:.2f}"
            heading_str = "LOCKED ON YOU!" if h_idx == 0 else "OVERSHOOTING/MISSING"
            heading_color = RED if h_idx == 0 else GREEN
            
            screen.blit(font.render(info, True, WHITE), (10, 10))
            screen.blit(font.render(f"Missile Status: {heading_str}", True, heading_color), (10, 30))

            pygame.display.update()
            clock.tick(FPS)

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
        if score > high_score:
            high_score = score

    pygame.quit()

if __name__ == "__main__":
    run_simulation()