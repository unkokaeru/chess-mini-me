"""
The main driver file. This will handle user input and displaying the current GameState object
"""

# Imports

import pygame as p
import ChessEngine
from pygame import Color


# Global variables

WIDTH = HEIGHT = 800
DIMENSION = 8  # dimensions of a chess board are 8x8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 30
IMAGES: dict = {}


def load_images() -> None:
    """
    Initalise a global dictionary of images
    :return: None
    """

    # List of pieces
    pieces = [
        "bR",
        "bN",
        "bB",
        "bQ",
        "bK",
        "bP",
        "wR",
        "wN",
        "wB",
        "wQ",
        "wK",
        "wP",
    ]  # TODO: use this to populate the board

    # Load the images
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(
            p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE)
        )


def main() -> None:
    """
    The main driver for the code. Handles user input and updating the graphics
    :return: None
    """

    # Initialize pygame
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))

    # Initialize game state
    gamestate = ChessEngine.GameState()
    load_images()
    running = True
    sq_selected: tuple = ()
    player_clicks: list[tuple] = []
    valid_moves = gamestate.get_valid_moves()
    move_made = False
    animate = False
    game_over = False
    colors = [p.Color("white"), p.Color("gray")]

    # Main game loop
    while running:
        for event in p.event.get():
            # Quit handlers
            if event.type == p.QUIT:
                running = False

            # Mouse handlers
            elif (
                event.type == p.MOUSEBUTTONDOWN
            ):  # TODO: add drag and drop functionality
                if not game_over:
                    location = p.mouse.get_pos()
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    if sq_selected == (
                        row,
                        col,
                    ):
                        sq_selected = ()
                        player_clicks = []
                    else:
                        sq_selected = (row, col)
                        player_clicks.append(sq_selected)
                    if len(player_clicks) == 2:
                        move = ChessEngine.Move(
                            player_clicks[0], player_clicks[1], gamestate.board
                        )
                        for i in range(len(valid_moves)):
                            if move == valid_moves[i]:
                                gamestate.make_move(valid_moves[i])
                                move_made = True
                                animate = True
                                sq_selected = ()
                                player_clicks = []
                        if not move_made:
                            player_clicks = [sq_selected]

            # Key handlers
            elif event.type == p.KEYDOWN:
                if event.key == p.K_z:
                    gamestate.undo_move()
                    move_made = True
                    animate = False
                    valid_moves = gamestate.get_valid_moves()
                if event.key == p.K_r:
                    gamestate = ChessEngine.GameState()
                    valid_moves = gamestate.get_valid_moves()
                    sq_selected = ()
                    player_clicks = []
                    move_made = False
                    animate = False

        # Update the graphics
        if move_made:
            if animate:
                animate_move(
                    gamestate.moveLog[-1], screen, gamestate.board, clock, colors
                )
            valid_moves = gamestate.get_valid_moves()
            move_made = False
            animate = False
        draw_gamestate(screen, gamestate, valid_moves, sq_selected, colors)

        # Checkmate and stalemate handlers
        if gamestate.checkmate:
            game_over = True
            if gamestate.whiteToMove:
                draw_text(screen, "Black wins by checkmate")
            else:
                draw_text(screen, "White wins by checkmate")

        # Refresh the screen
        clock.tick(MAX_FPS)
        p.display.flip()


def highlight_squares(
    screen: p.Surface,
    gamestate: ChessEngine.GameState,
    valid_moves: list,
    sq_selected: tuple,
) -> None:  # TODO: add highlighting for last move
    """
    Highlight square selected and moves for piece selected
    :param screen: the screen to draw on
    :param gamestate: the current gamestate
    :param valid_moves: a list of valid moves
    :param sq_selected: the square selected
    :return: None
    """

    if sq_selected != ():
        row, col = sq_selected

        if gamestate.board[row][col][0] == ("w" if gamestate.whiteToMove else "b"):
            # Highlight the selected square
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(p.Color("blue"))

            # Highlight moves from that square
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
            s.fill(p.Color("yellow"))
            for move in valid_moves:
                if move.startRow == row and move.startCol == col:
                    screen.blit(s, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))


def draw_gamestate(
    screen: p.Surface,
    gamestate: ChessEngine.GameState,
    valid_moves: list,
    sq_selected: tuple,
    colors: list[Color],
) -> None:
    """
    Responsible for all the graphics within a current gamestate
    :param screen: the screen to draw on
    :param gamestate: the gamestate to draw
    :param valid_moves: a list of valid moves
    :param sq_selected: the square selected
    :return: None
    """

    # Draw the board
    draw_board(screen, colors)

    # Highlight squares
    highlight_squares(screen, gamestate, valid_moves, sq_selected)

    # Draw pieces
    draw_pieces(screen, gamestate.board)


def draw_board(screen: p.Surface, colors: list[Color]) -> None:
    """
    Draw the squares on the board
    :param screen: the screen to draw on
    :return: None
    """

    # Draw the squares on the board
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = colors[((row + col) % 2)]
            p.draw.rect(
                screen, color, p.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            )


def draw_pieces(screen: p.Surface, board: list) -> None:
    """
    Draw the pieces on the board using the current GameState.board
    :param screen: the screen to draw on
    :param board: the board to draw
    :return: None
    """

    # Draw the pieces on the board
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != "--":
                screen.blit(
                    IMAGES[piece],
                    p.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                )


def draw_text(
    screen: p.Surface, text: str
) -> None:  # TODO: Add a background to the text
    """
    Draw text on the screen
    :param screen: the screen to draw on
    :param text: the text to draw
    :return: None
    """

    # Draw the text on the screen
    font = p.font.SysFont("Helvitca", 32, True, False)
    text_object = font.render(text, 0, p.Color("Black"))
    text_location = p.Rect(0, 0, WIDTH, HEIGHT).move(
        WIDTH / 2 - text_object.get_width() / 2,
        HEIGHT / 2 - text_object.get_height() / 2,
    )
    screen.blit(text_object, text_location)


def animate_move(
    move: ChessEngine.Move,
    screen: p.Surface,
    board: list,
    clock: p.time.Clock,
    colors: list[Color],
) -> None:
    """
    Animate a move
    :param move: the move to animate
    :param screen: the screen to draw on
    :param board: the board to draw
    :param clock: the clock to keep track of time
    :return: None
    """

    # Calculate the frames per square and the total number of frames
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    frames_per_square = 5  # frames to move one square (arbitrary) -> change later to be proportional to the distance
    frame_count = (
        abs(dR) + abs(dC)
    ) * frames_per_square  # total number of frames for a move

    # Animate the move
    for frame in range(frame_count + 1):
        r, c = (
            move.startRow + dR * frame / frame_count,
            move.startCol + dC * frame / frame_count,
        )

        # Re-draw the entire board TODO: only redraw the squares that changed
        draw_board(screen, colors)
        draw_pieces(screen, board)

        # Erase the piece moved from its ending square
        color = colors[(move.endRow + move.endCol) % 2]
        end_square = p.Rect(
            move.endCol * SQ_SIZE, move.endRow * SQ_SIZE, SQ_SIZE, SQ_SIZE
        )
        p.draw.rect(screen, color, end_square)

        # Draw captured piece onto rectangle
        if move.pieceCaptured != "--":
            screen.blit(IMAGES[move.pieceCaptured], end_square)

        # Draw the moving piece
        screen.blit(
            IMAGES[move.pieceMoved], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        )
        p.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
