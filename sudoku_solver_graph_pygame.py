import numpy as np
import random
import time
import pygame
import os
import cv2  # Video oluşturmak için eklendi

pygame.init()
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 700
SQUARE_SIZE = SCREEN_WIDTH // 9
FONT = pygame.font.SysFont('Arial', 40)
STATUS_FONT = pygame.font.SysFont('Arial', 24)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Graf Sudoku Çözücü")

# --- VİDEO KAYDI İÇİN EKLENDİ ---
FRAMES_DIR = "sudoku_frames"
FRAME_COUNT = 0
# ---------------------------------

initial_board_mask = None

def draw_grid(screen, board, highlight_cell=None):
    screen.fill(WHITE)
    
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        
        pygame.draw.line(screen, BLACK, (0, i * SQUARE_SIZE), (SCREEN_WIDTH, i * SQUARE_SIZE), thickness)
        pygame.draw.line(screen, BLACK, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, SCREEN_WIDTH), thickness)
        
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                text_color = BLUE if not initial_board_mask[i, j] else BLACK
                
                if highlight_cell and highlight_cell == (i, j):
                    pygame.draw.rect(screen, YELLOW, (j * SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                    text_color = BLACK
                    
                num_text = FONT.render(str(board[i][j]), True, text_color)
                screen.blit(num_text, (j * SQUARE_SIZE + 15, i * SQUARE_SIZE + 5))
                
    pygame.draw.rect(screen, BLACK, (0, SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_HEIGHT - SCREEN_WIDTH), 1)

def update_status(action):
    # --- VİDEO KAYDI İÇİN GÜNCELLENDİ ---
    global FRAME_COUNT  # Global sayacı kullan
    
    status_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT - SCREEN_WIDTH))
    status_surface.fill(GRAY)
    
    text = STATUS_FONT.render(action, True, BLACK)
    status_surface.blit(text, (10, 10))
    
    screen.blit(status_surface, (0, SCREEN_WIDTH))

    # Mevcut ekran görüntüsünü bir dosyaya kaydet
    #frame_filename = os.path.join(FRAMES_DIR, f"frame_{FRAME_COUNT:06d}.png")
    #pygame.image.save(screen, frame_filename)
    #FRAME_COUNT += 1
    # -------------------------------------
    
    pygame.display.flip() # Ekranı güncelle

def find_empty_cell(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)
    return None

def is_valid_placement(board, num, row, col):
    if num in board[row]: return False
    if num in board[:, col]: return False

    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3

    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            if board[r][c] == num: return False
    return True

def fill_board(board):
    find = find_empty_cell(board)
    if not find: return True
    row, col = find
    numbers = list(range(1, 10))
    random.shuffle(numbers) 
    for num in numbers:
        if is_valid_placement(board, num, row, col):
            board[row][col] = num
            if fill_board(board): return True
            board[row][col] = 0
    return False

def generate_sudoku(clues=25):
    global initial_board_mask
    board = np.zeros((9, 9), dtype=int)
    fill_board(board)
    
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    
    cells_to_remove = 81 - clues
    puzzle_board = board.copy()
    
    initial_board_mask = np.ones((9, 9), dtype=bool) 
    
    for r, c in cells:
        if cells_to_remove <= 0: break
        puzzle_board[r][c] = 0
        initial_board_mask[r, c] = False
        cells_to_remove -= 1
        
    return puzzle_board

def solve_sudoku_animated_pygame(board, delay=0):
    find = find_empty_cell(board)
    if not find:
        draw_grid(screen, board)
        update_status("ÇÖZÜM BULUNDU! (Graf Tamamen Renklendirildi)")
        return True
    
    row, col = find

    for color in range(1, 10):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if is_valid_placement(board, color, row, col):
            
            board[row][col] = color
            draw_grid(screen, board, (row, col))
            update_status(f"DENENİYOR: ({row+1}, {col+1}) hücresine {color} atandı.")
            pygame.time.wait(delay) 

            if solve_sudoku_animated_pygame(board, delay):
                return True

            board[row][col] = 0 
            draw_grid(screen, board, (row, col))
            #update_status(f"GERİ İZLEME: ({row+1}, {col+1}) hücresi 0'a sıfırlandı.")
            pygame.time.wait(delay)

    return False

# --- VİDEO OLUŞTURMAK İÇİN YENİ FONKSİYON ---
def compile_frames_to_video(frames_dir, output_file, fps=60):
    """
    Kaydedilen kareleri bir video dosyasında birleştirir.
    """
    print(f"Kareler videoya dönüştürülüyor... Hedef: {output_file}")
    
    images = [img for img in os.listdir(frames_dir) if img.endswith(".png")]
    if not images:
        print("Video oluşturulamadı, hiç kare bulunamadı.")
        return
        
    # Görüntüleri ada göre sırala (frame_000000.png, frame_000001.png, ...)
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # İlk görüntüden videonun boyutlarını al
    frame_path = os.path.join(frames_dir, images[0])
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Hata: İlk kare okunamadı: {frame_path}")
        return
        
    height, width, layers = frame.shape

    # Video yazıcıyı oluştur
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # veya 'XVID'
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Her kareyi videoya yaz
    for image_name in images:
        img_path = os.path.join(frames_dir, image_name)
        img = cv2.imread(img_path)
        if img is not None:
            video.write(img)
        else:
            print(f"Uyarı: Kare okunamadı: {img_path}")

    video.release()
    cv2.destroyAllWindows()
    print(f"Video başarıyla '{output_file}' olarak kaydedildi.")
# ------------------------------------------

def main():
    # --- VİDEO KAYDI İÇİN EKLENDİ ---
    # Kareleri kaydetmek için klasör oluştur
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
    # ---------------------------------
        
    puzzle = generate_sudoku(clues=25) 
    
    ANIMATION_DELAY_MS = 0
    
    draw_grid(screen, puzzle)
    update_status("Başlangıç Bulmacası (Graf Renklendirmeye Hazır)...")
    pygame.time.wait(2000)

    solve_board = puzzle.copy()
    start_time = time.time()
    
    solved = solve_sudoku_animated_pygame(solve_board, delay=ANIMATION_DELAY_MS)
    
    end_time = time.time()
    
    if solved:
        final_message = f"Çözüldü! Süre: {end_time - start_time:.4f}s"
    else:
        final_message = "Çözüm Bulunamadı."
    
    # --- VİDEO KAYDI İÇİN GÜNCELLENMİŞ BİTİŞ DÖNGÜSÜ ---
    # Son mesajı 5 saniye boyunca göster ve kareleri kaydet, sonra otomatik çık
    end_time_display = time.time()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Her döngüde son durumu güncelle (ve kareyi kaydet)
        update_status(final_message) 
        
        # 5 saniye geçtiyse veya kullanıcı pencereyi kapattıysa çık
        if (time.time() - end_time_display > 5.0):
            running = False
            
        pygame.time.wait(100) # CPU'yu yormamak için kısa bir bekleme
    # ----------------------------------------------------
    
    pygame.quit()

    # --- VİDEO KAYDI İÇİN EKLENDİ ---
    # Pygame kapandıktan sonra kareleri videoya dönüştür
    compile_frames_to_video(FRAMES_DIR, "sudoku_cozumu.mp4", fps=60)
    # ---------------------------------

if __name__ == "__main__":
    main()