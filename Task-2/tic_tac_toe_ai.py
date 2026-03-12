import math

# Create the board
board = [" " for _ in range(9)]

# Print the board
def print_board():
    print()
    print(board[0] + " | " + board[1] + " | " + board[2])
    print("--+---+--")
    print(board[3] + " | " + board[4] + " | " + board[5])
    print("--+---+--")
    print(board[6] + " | " + board[7] + " | " + board[8])
    print()

# Check winner
def check_winner(player):
    win_conditions = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

# Check draw
def is_draw():
    return " " not in board

# Minimax algorithm
def minimax(depth, is_maximizing):

    if check_winner("O"):
        return 1

    if check_winner("X"):
        return -1

    if is_draw():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                score = minimax(depth + 1, False)
                board[i] = " "
                best_score = max(score, best_score)
        return best_score

    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                score = minimax(depth + 1, True)
                board[i] = " "
                best_score = min(score, best_score)
        return best_score

# AI move
def ai_move():
    best_score = -math.inf
    move = 0

    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(0, False)
            board[i] = " "

            if score > best_score:
                best_score = score
                move = i

    board[move] = "O"

# Human move
def human_move():
    move = int(input("Enter position (1-9): ")) - 1

    if board[move] != " ":
        print("Invalid move! Try again.")
        human_move()
    else:
        board[move] = "X"

# Game loop
print("Tic Tac Toe AI")
print("You are X | AI is O")

while True:

    print_board()

    human_move()

    if check_winner("X"):
        print_board()
        print("You Win!")
        break

    if is_draw():
        print_board()
        print("Draw!")
        break

    ai_move()

    if check_winner("O"):
        print_board()
        print("AI Wins!")
        break

    if is_draw():
        print_board()
        print("Draw!")
        break