"""
The main driver file. This will handle user input and displaying the current GameState object
"""

# Imports

import pygame as p
from pygame import Surface
from engine import Move, GameState
import movefinder
from pygame import Color

# Types

COORDS = tuple[int, int]
IMAGE = Surface


def load_images(SQ_SIZE: int, IMAGES: dict[str, IMAGE]) -> None:
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

    # TODO: add a main menu, settings, and a game over screen
    # TODO: add asynchronous programming
    # TODO: refactor code to be more modular

    # Initalise constants
    WIDTH = HEIGHT = 800
    DIMENSION = 8  # dimensions of a chess board are 8x8
    SQ_SIZE = HEIGHT // DIMENSION
    MAX_FPS = 30

    # Initalise pygame
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))

    # Initalise game state
    gamestate = GameState()
    images: dict[str, IMAGE] = {}
    load_images(SQ_SIZE, images)
    running = True
    sq_selected: tuple | COORDS = ()
    player_clicks: list[tuple | COORDS] = []
    valid_moves = gamestate.get_valid_moves()
    move_made = False
    animate = False
    game_over = False
    colors = [p.Color("white"), p.Color("gray")]

    # Initalise players (if False, AI is playing)
    player_white_human = True
    player_black_human = False

    # Main game loop
    while running:
        # Human turn handlerss
        human_turn = (gamestate.white_to_move and player_white_human) or (
            not gamestate.white_to_move and player_black_human
        )

        for event in p.event.get():
            # Quit handlers
            if event.type == p.QUIT:
                running = False

            # Mouse handlers
            elif (
                event.type == p.MOUSEBUTTONDOWN
            ):  # TODO: add drag and drop functionality
                if not game_over and human_turn:
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
                        move = Move(player_clicks[0], player_clicks[1], gamestate.board)
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
                # Undo move
                if event.key == p.K_z:
                    gamestate.undo_move()
                    move_made = True
                    animate = False
                    valid_moves = gamestate.get_valid_moves()

                # Reset board
                if event.key == p.K_r:
                    gamestate = GameState()
                    valid_moves = gamestate.get_valid_moves()
                    sq_selected = ()
                    player_clicks = []
                    move_made = False
                    animate = False

        # AI move handlers
        if not game_over and not human_turn:
            ai_move = movefinder.find_best_move(gamestate, valid_moves)
            gamestate.make_move(ai_move, human_turn)
            move_made = True
            animate = True

        # Update the graphics
        if move_made:
            if animate:
                animate_move(
                    gamestate.move_log[-1],
                    screen,
                    gamestate.board,
                    clock,
                    colors,
                    DIMENSION,
                    SQ_SIZE,
                    images,
                )
            valid_moves = gamestate.get_valid_moves()
            move_made = False
            animate = False
        draw_gamestate(
            screen,
            gamestate,
            valid_moves,
            sq_selected,
            colors,
            SQ_SIZE,
            DIMENSION,
            images,
        )

        # Checkmate and stalemate handlers
        if gamestate.checkmate:
            game_over = True
            if gamestate.white_to_move:
                draw_text(screen, "Black wins by checkmate", WIDTH, HEIGHT)
            else:
                draw_text(screen, "White wins by checkmate", WIDTH, HEIGHT)

        if gamestate.stalemate:
            game_over = True
            draw_text(screen, "Stalemate", WIDTH, HEIGHT)

        # Refresh the screen
        clock.tick(MAX_FPS)
        p.display.flip()


def highlight_squares(
    screen: IMAGE,
    gamestate: GameState,
    valid_moves: list[Move],
    sq_selected: tuple | COORDS,
    SQ_SIZE: int,
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

        if gamestate.board[row][col][0] == ("w" if gamestate.white_to_move else "b"):
            # Highlight the selected square
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(p.Color("blue"))

            # Highlight moves from that square
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
            s.fill(p.Color("yellow"))
            for move in valid_moves:
                if move.start_row == row and move.start_col == col:
                    screen.blit(s, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))


def draw_gamestate(
    screen: IMAGE,
    gamestate: GameState,
    valid_moves: list[Move],
    sq_selected: tuple | COORDS,
    colors: list[Color],
    SQ_SIZE: int,
    DIMENSION: int,
    IMAGES: dict[str, IMAGE],
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
    draw_board(screen, colors, DIMENSION, SQ_SIZE)

    # Highlight squares
    highlight_squares(screen, gamestate, valid_moves, sq_selected, SQ_SIZE)

    # Draw pieces
    draw_pieces(screen, gamestate.board, DIMENSION, SQ_SIZE, IMAGES)


def draw_board(
    screen: IMAGE, colors: list[Color], DIMENSION: int, SQ_SIZE: int
) -> None:
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


def draw_pieces(
    screen: IMAGE,
    board: list[list[str]],
    DIMENSION: int,
    SQ_SIZE: int,
    IMAGES: dict[str, IMAGE],
) -> None:
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
    screen: IMAGE, text: str, WIDTH: int, HEIGHT: int
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
    move: Move,
    screen: IMAGE,
    board: list[list[str]],
    clock: p.time.Clock,
    colors: list[Color],
    DIMENSION: int,
    SQ_SIZE: int,
    IMAGES: dict[str, IMAGE],
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
    dR = move.end_row - move.start_row
    dC = move.end_col - move.start_col
    frames_per_square = 5  # frames to move one square (arbitrary) -> change later to be proportional to the distance
    frame_count = (
        abs(dR) + abs(dC)
    ) * frames_per_square  # total number of frames for a move

    # Animate the move
    for frame in range(frame_count + 1):
        r, c = (
            move.start_row + dR * frame / frame_count,
            move.start_col + dC * frame / frame_count,
        )

        # Re-draw the entire board TODO: only redraw the squares that changed
        draw_board(screen, colors, DIMENSION, SQ_SIZE)
        draw_pieces(screen, board, DIMENSION, SQ_SIZE, IMAGES)

        # Erase the piece moved from its ending square
        color = colors[(move.end_row + move.end_col) % 2]
        end_square = p.Rect(
            move.end_col * SQ_SIZE, move.end_row * SQ_SIZE, SQ_SIZE, SQ_SIZE
        )
        p.draw.rect(screen, color, end_square)

        # Draw captured piece onto rectangle
        if move.piece_captured != "--":
            screen.blit(IMAGES[move.piece_captured], end_square)

        # Draw the moving piece
        screen.blit(
            IMAGES[move.piece_moved], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        )
        p.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
