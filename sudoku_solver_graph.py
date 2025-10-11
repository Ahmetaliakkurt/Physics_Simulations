import numpy as np
import random
import time

def find_empty_cell(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)
    return None

def is_valid_placement(board, num, row, col):

    if num in board[row]:
        return False

    if num in board[:, col]:
        return False

    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3

    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            if board[r][c] == num:
                return False

    return True

def print_board(board, title="Tahta"):
    print(f"\n### {title} ###")
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - ")

        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")
def fill_board(board):
    find = find_empty_cell(board)
    if not find:
        return True
    
    row, col = find
    numbers = list(range(1, 10))
    random.shuffle(numbers) 

    for num in numbers:
        if is_valid_placement(board, num, row, col):
            board[row][col] = num
            
            if fill_board(board):
                return True

            board[row][col] = 0 

    return False

def generate_sudoku(clues=40):
    board = np.zeros((9, 9), dtype=int)
    fill_board(board)
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    
    cells_to_remove = 81 - clues # Yaklaşık silinecek hücre sayısı
    
    puzzle_board = board.copy()
    
    for r, c in cells:
        if cells_to_remove <= 0:
            break
            
        temp = puzzle_board[r][c]
        puzzle_board[r][c] = 0
        cells_to_remove -= 1
        
    return puzzle_board

def solve_sudoku_graph_coloring(board):
    find = find_empty_cell(board)
    if not find:
        return True
    row, col = find
    for color in range(1, 10):
        if is_valid_placement(board, color, row, col):
            board[row][col] = color
            if solve_sudoku_graph_coloring(board):
                return True
            board[row][col] = 0 # Geri İzleme
    return False


start_time = time.time()
new_puzzle = generate_sudoku(clues=30) 
generation_time = time.time() - start_time

print(f"Oluşturma Süresi: {generation_time:.4f} saniye\n")
print("=" * 60)
print_board(new_puzzle, title="Rastgele Oluşturulmuş Sudoku")
print("=" * 60)

solve_board = new_puzzle.copy()
start_time = time.time()

if solve_sudoku_graph_coloring(solve_board):
    solve_time = time.time() - start_time
    print(f"Çözüm Süresi: {solve_time:.4f} saniye\n")
    print_board(solve_board, title="Graf Teorisi ile Çözümü")