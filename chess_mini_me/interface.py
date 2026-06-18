"""The graphical interface for the chess game, built with Pygame.

This module turns the rules in :mod:`chess_mini_me.engine`, the search in
:mod:`chess_mini_me.move_finder` and the learned opponent in
:mod:`chess_mini_me.cloner` into a playable game. It is the only module that
depends on Pygame, so importing the engine for tests or scripting never
requires a display.

The window has a board on the left and a side panel on the right showing the
move list, a status line and an action button (resign during play, save the
game as PGN once it is over). Responsibilities are gathered into a single
:class:`ChessInterface` object so the window, images, fonts and geometry are
configured once and shared.
"""

from __future__ import annotations

import datetime
import pathlib

import pygame

from chess_mini_me import constants, opponent, pgn
from chess_mini_me.engine import GameState, Move
from chess_mini_me.training import (
    BLACK_WIN_VALUE,
    DRAW_VALUE,
    WHITE_WIN_VALUE,
    StyleRecorder,
    default_store,
)

# A square on the board, or ``None`` when nothing is selected.
Square = tuple[int, int]

# Colours used by the interface. Pygame's API spells "color" the American way,
# so that spelling appears only where its names are passed to Pygame.
LIGHT_SQUARE_COLOUR = pygame.Color(238, 238, 210)
DARK_SQUARE_COLOUR = pygame.Color(118, 150, 86)
SELECTED_SQUARE_COLOUR = pygame.Color(246, 246, 105)
LAST_MOVE_COLOUR = pygame.Color(205, 210, 106)
LEGAL_MOVE_COLOUR = pygame.Color(106, 168, 79)
BACKGROUND_COLOUR = pygame.Color(49, 46, 43)
PANEL_BACKGROUND_COLOUR = pygame.Color(38, 36, 33)
TEXT_COLOUR = pygame.Color(235, 235, 235)
MUTED_TEXT_COLOUR = pygame.Color(170, 165, 158)
PANEL_COLOUR = pygame.Color(70, 66, 60)
HIGHLIGHTED_PANEL_COLOUR = pygame.Color(120, 110, 95)
ACTION_BUTTON_COLOUR = pygame.Color(150, 80, 70)
OVERLAY_COLOUR = pygame.Color(0, 0, 0)


class ChessInterface:
    """Run the chess game window, from the main menu to game over."""

    def __init__(self) -> None:
        """Create the window and load the resources shared across the game."""
        pygame.init()
        self.board_size = self._determine_board_size()
        self.square_size = self.board_size // constants.BOARD_DIMENSION
        self.board_size = self.square_size * constants.BOARD_DIMENSION

        self.panel_width = int(self.square_size * 3.4)
        self.window_width = self.board_size + self.panel_width
        self.window_height = self.board_size

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption(constants.WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        self.large_font = pygame.font.SysFont("helvetica", self.square_size // 2, True)
        self.medium_font = pygame.font.SysFont("helvetica", self.square_size // 3, True)
        self.small_font = pygame.font.SysFont("helvetica", self.square_size // 4)
        self.move_font = pygame.font.SysFont("couriernew", self.square_size // 4)

        self.piece_images = self._load_piece_images()
        self.store = default_store()
        self.games_directory = self.store.directory / "games"

    # ------------------------------------------------------------------
    # Set-up helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_board_size() -> int:
        """Return a board size in pixels that fits the available display.

        Returns:
            The side length of the square board, in pixels.
        """
        try:
            display_height = pygame.display.Info().current_h
        except pygame.error:
            display_height = 0
        if display_height <= 0:
            return constants.FALLBACK_BOARD_SIZE_PIXELS
        return max(
            constants.BOARD_DIMENSION,
            display_height - constants.WINDOW_MARGIN_PIXELS,
        )

    def _load_piece_images(self) -> dict[str, pygame.Surface]:
        """Load and scale every piece image to the current square size.

        Returns:
            A mapping from piece code (such as ``wQ``) to its scaled image.

        Raises:
            FileNotFoundError: If an expected piece image is missing.
        """
        image_directory = (
            pathlib.Path(__file__).resolve().parent.parent
            / constants.IMAGE_DIRECTORY_NAME
        )
        images: dict[str, pygame.Surface] = {}
        for colour in constants.PIECE_COLOURS:
            for piece_type in constants.PIECE_TYPES:
                piece = colour + piece_type
                path = image_directory / (piece + constants.IMAGE_FILE_EXTENSION)
                if not path.exists():
                    raise FileNotFoundError(f"Missing piece image: {path}")
                image = pygame.image.load(str(path))
                images[piece] = pygame.transform.smoothscale(
                    image, (self.square_size, self.square_size)
                )
        return images

    # ------------------------------------------------------------------
    # Top-level flow
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Show the menu and play games until the player chooses to quit."""
        while True:
            player_setup = self._show_main_menu()
            if player_setup is None:
                break
            should_continue = self._play_game(*player_setup)
            if not should_continue:
                break
        pygame.quit()

    def _show_main_menu(self) -> tuple[str, str] | None:
        """Display the main menu and collect the choice of players.

        Returns:
            A tuple ``(white_player, black_player)`` of player-type names, or
            ``None`` if the player closed the window.
        """
        white_player = opponent.PLAYER_HUMAN
        black_player = opponent.PLAYER_COMPUTER

        while True:
            buttons = self._draw_main_menu(white_player, black_player)
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    action = self._button_action_at(buttons, event.pos)
                    if action == "toggle_white":
                        white_player = opponent.next_player_type(white_player)
                    elif action == "toggle_black":
                        black_player = opponent.next_player_type(black_player)
                    elif action == "learn_lichess":
                        self._run_lichess_learning()
                    elif action == "start":
                        return white_player, black_player
                    elif action == "quit":
                        return None

    def _play_game(self, white_player: str, black_player: str) -> bool:
        """Play a single game.

        Args:
            white_player: The player type controlling White.
            black_player: The player type controlling Black.

        Returns:
            True to return to the main menu, or False to quit the program.
        """
        gamestate = GameState()
        valid_moves = gamestate.get_valid_moves()
        controller = opponent.OpponentController(self.store)
        recorder = StyleRecorder()
        player_labels = {
            constants.WHITE: opponent.PLAYER_LABELS[white_player],
            constants.BLACK: opponent.PLAYER_LABELS[black_player],
        }

        selected_square: Square | None = None
        is_dragging = False
        drag_position = (0, 0)
        has_learned_from_game = False
        saved_pgn_name: str | None = None
        san_cache: list[str] = []

        while True:
            player_to_move = white_player if gamestate.white_to_move else black_player
            human_to_move = player_to_move == opponent.PLAYER_HUMAN
            game_over = gamestate.is_game_over()
            action_rect = self._action_button_rect()

            if len(san_cache) != len(gamestate.move_log):
                san_cache = pgn.build_san_moves(gamestate)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        return True
                    if event.key == pygame.K_q:
                        return False
                    if event.key == pygame.K_z and gamestate.move_log and not game_over:
                        gamestate.undo_move()
                        valid_moves = gamestate.get_valid_moves()
                        selected_square, is_dragging = None, False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    # The panel's action button works regardless of whose turn
                    # it is: resign during play, or save the game once it ends.
                    if action_rect.collidepoint(event.pos):
                        if game_over:
                            saved_pgn_name = self._save_game(gamestate, player_labels)
                        elif human_to_move:
                            mover = (
                                constants.WHITE
                                if gamestate.white_to_move
                                else constants.BLACK
                            )
                            gamestate.resign(mover)
                        continue

                if game_over or not human_to_move:
                    continue

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not self._is_on_board(event.pos):
                        continue
                    clicked = self._square_at(event.pos)
                    if selected_square is None:
                        if self._belongs_to_mover(gamestate, clicked):
                            selected_square = clicked
                            is_dragging = True
                            drag_position = event.pos
                    else:
                        moved, valid_moves = self._attempt_move(
                            gamestate, valid_moves, selected_square, clicked, recorder
                        )
                        if moved:
                            selected_square, is_dragging = None, False
                        elif self._belongs_to_mover(gamestate, clicked):
                            selected_square, is_dragging = clicked, True
                            drag_position = event.pos
                        else:
                            selected_square, is_dragging = None, False

                elif event.type == pygame.MOUSEMOTION and is_dragging:
                    drag_position = event.pos

                elif event.type == pygame.MOUSEBUTTONUP and is_dragging:
                    is_dragging = False
                    if self._is_on_board(event.pos):
                        released = self._square_at(event.pos)
                        if released != selected_square:
                            moved, valid_moves = self._attempt_move(
                                gamestate, valid_moves, selected_square, released,
                                recorder,
                            )
                            selected_square = None

            # Let a computer or Mini-Me player reply, or resign, on its turn.
            if not game_over and not human_to_move:
                if controller.should_resign(player_to_move, gamestate):
                    mover = (
                        constants.WHITE if gamestate.white_to_move else constants.BLACK
                    )
                    gamestate.resign(mover)
                else:
                    move, promotion_piece = controller.choose_move(
                        player_to_move, gamestate, valid_moves
                    )
                    if move is not None:
                        self._animate_move(gamestate, move)
                        gamestate.make_move(move, promotion_piece)
                        valid_moves = gamestate.get_valid_moves()

            self._draw_position(
                gamestate, valid_moves, selected_square, is_dragging, drag_position
            )
            self._draw_panel(gamestate, san_cache, player_to_move, saved_pgn_name)
            if gamestate.is_game_over():
                self._draw_game_over_banner(gamestate)
                if not has_learned_from_game:
                    self._learn_from_game(gamestate, recorder, controller)
                    has_learned_from_game = True
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND)

    # ------------------------------------------------------------------
    # Move handling
    # ------------------------------------------------------------------

    def _attempt_move(
        self,
        gamestate: GameState,
        valid_moves: list[Move],
        start_square: Square,
        end_square: Square,
        recorder: StyleRecorder | None = None,
    ) -> tuple[bool, list[Move]]:
        """Make the move from one square to another when it is legal.

        Args:
            gamestate: The game state to update.
            valid_moves: The legal moves for the side to move.
            start_square: The square the piece is moving from.
            end_square: The square the piece is moving to.
            recorder: An optional recorder that captures the human's move.

        Returns:
            A tuple ``(was_made, valid_moves)`` with refreshed moves if made.
        """
        intended_move = Move(start_square, end_square, gamestate.board)
        for legal_move in valid_moves:
            if legal_move == intended_move:
                promotion_piece = constants.QUEEN
                if legal_move.is_pawn_promotion:
                    promotion_piece = self._choose_promotion(
                        legal_move.piece_moved[0]
                    )
                if recorder is not None:
                    recorder.record(gamestate, legal_move, promotion_piece)
                gamestate.make_move(legal_move, promotion_piece)
                return True, gamestate.get_valid_moves()
        return False, valid_moves

    @staticmethod
    def _belongs_to_mover(gamestate: GameState, square: Square) -> bool:
        """Return whether a square holds a piece of the side to move.

        Args:
            gamestate: The current game state.
            square: The (row, column) square to inspect.

        Returns:
            True if the square holds a piece belonging to the side to move.
        """
        piece = gamestate.board[square[0]][square[1]]
        if piece == constants.EMPTY_SQUARE:
            return False
        mover_colour = constants.WHITE if gamestate.white_to_move else constants.BLACK
        return piece[0] == mover_colour

    def _is_on_board(self, position: tuple[int, int]) -> bool:
        """Return whether a pixel position lies over the board, not the panel.

        Args:
            position: The (x, y) pixel position.

        Returns:
            True if the position is within the board area.
        """
        return 0 <= position[0] < self.board_size and 0 <= position[1] < self.board_size

    def _square_at(self, position: tuple[int, int]) -> Square:
        """Convert a pixel position into a board square.

        Args:
            position: The (x, y) pixel position.

        Returns:
            The (row, column) square containing that position.
        """
        column = position[0] // self.square_size
        row = position[1] // self.square_size
        return row, column

    def _choose_promotion(self, colour: str) -> str:
        """Ask the player which piece a promoting pawn should become.

        Args:
            colour: The colour of the promoting pawn (``"w"`` or ``"b"``).

        Returns:
            The chosen piece type (``"Q"``, ``"R"``, ``"B"`` or ``"N"``).
        """
        choices = (
            constants.QUEEN,
            constants.ROOK,
            constants.BISHOP,
            constants.KNIGHT,
        )
        panel_width = self.square_size * len(choices)
        panel_left = (self.board_size - panel_width) // 2
        panel_top = (self.board_size - self.square_size) // 2
        option_rectangles = {
            piece_type: pygame.Rect(
                panel_left + index * self.square_size,
                panel_top,
                self.square_size,
                self.square_size,
            )
            for index, piece_type in enumerate(choices)
        }

        while True:
            overlay = pygame.Surface((self.board_size, self.board_size))
            overlay.set_alpha(160)
            overlay.fill(OVERLAY_COLOUR)
            self.screen.blit(overlay, (0, 0))
            for piece_type, rectangle in option_rectangles.items():
                pygame.draw.rect(self.screen, PANEL_COLOUR, rectangle)
                self.screen.blit(self.piece_images[colour + piece_type], rectangle)
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return constants.QUEEN
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for piece_type, rectangle in option_rectangles.items():
                        if rectangle.collidepoint(event.pos):
                            return piece_type

    # ------------------------------------------------------------------
    # Saving and learning
    # ------------------------------------------------------------------

    def _save_game(
        self, gamestate: GameState, player_labels: dict[str, str]
    ) -> str:
        """Save the finished game as a PGN file and return its name.

        Args:
            gamestate: The finished game.
            player_labels: A mapping from colour to a display name.

        Returns:
            The file name the game was saved as.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"game-{timestamp}.pgn"
        pgn.save_pgn(
            gamestate,
            self.games_directory / file_name,
            white_name=player_labels[constants.WHITE],
            black_name=player_labels[constants.BLACK],
        )
        return file_name

    @staticmethod
    def _game_result_value(gamestate: GameState) -> float:
        """Return the finished game's result from White's perspective.

        Args:
            gamestate: The finished game state.

        Returns:
            ``1`` for a White win, ``-1`` for a Black win, ``0`` for a draw.
        """
        result = gamestate.result_string()
        if result == "1-0":
            return WHITE_WIN_VALUE
        if result == "0-1":
            return BLACK_WIN_VALUE
        return DRAW_VALUE

    def _learn_from_game(
        self,
        gamestate: GameState,
        recorder: StyleRecorder,
        controller: "opponent.OpponentController",
    ) -> None:
        """Teach the Mini-Me from the human moves of the finished game.

        Args:
            gamestate: The finished game state.
            recorder: The recorder holding the human's moves.
            controller: The opponent controller to refresh afterwards.
        """
        if not recorder.has_moves():
            return
        self._draw_message_overlay(
            ["Mini-Me is learning your style ..."], self.medium_font
        )
        pygame.display.flip()
        try:
            opponent.learn_from_finished_game(
                self.store, recorder, self._game_result_value(gamestate)
            )
            controller.reload_mini_me()
        except ImportError:
            self._draw_message_overlay(
                ["Install PyTorch to let the Mini-Me learn."], self.small_font
            )
            pygame.display.flip()
            pygame.time.wait(1500)

    def _run_lichess_learning(self) -> None:
        """Prompt for a Lichess username and train the Mini-Me from it."""
        username = self._text_input_prompt("Lichess username (Enter to confirm):")
        if not username:
            return
        from chess_mini_me import lichess
        from chess_mini_me.training import learn_from_examples

        try:
            self._draw_message_overlay(
                [f"Downloading {username}'s games ..."], self.medium_font
            )
            pygame.display.flip()
            pgn_text = lichess.download_games(username)

            self._draw_message_overlay(["Analysing playing style ..."], self.medium_font)
            pygame.display.flip()
            games = lichess.analyse_games(pgn_text, username)
            examples = lichess.extract_examples(pgn_text, username)
            if len(examples) == 0:
                self._show_temporary_message(
                    ["No games were found for that player."], self.medium_font
                )
                return

            summary_lines = lichess.summarise_style(games).splitlines()
            self._draw_message_overlay(
                ["Training the Mini-Me ...", ""] + summary_lines[:6], self.small_font
            )
            pygame.display.flip()
            learn_from_examples(self.store, examples, epochs=8)
            self._show_temporary_message(
                [f"Mini-Me trained on {len(examples)} of", f"{username}'s moves."],
                self.medium_font,
            )
        except ValueError as error:
            self._show_temporary_message([str(error)], self.medium_font)
        except ImportError as error:
            self._show_temporary_message(
                ["This feature needs extra libraries:", str(error)], self.small_font
            )

    def _text_input_prompt(self, prompt_text: str) -> str | None:
        """Collect a single line of text from the player.

        Args:
            prompt_text: The prompt to display above the input box.

        Returns:
            The entered text, or ``None`` if the player cancelled.
        """
        entered_text = ""
        while True:
            self.screen.fill(BACKGROUND_COLOUR)
            self._blit_centred_text(
                prompt_text, self.medium_font, self.window_height // 2 - self.square_size,
                baseline_is_centre=True, width=self.window_width,
            )
            box = pygame.Rect(
                self.window_width // 6, self.window_height // 2,
                self.window_width * 2 // 3, self.square_size,
            )
            pygame.draw.rect(self.screen, PANEL_COLOUR, box, border_radius=6)
            self._blit_centred_text(
                entered_text or "_", self.medium_font, box.centery,
                baseline_is_centre=True, width=self.window_width,
            )
            self._blit_centred_text(
                "Enter to confirm, Escape to cancel", self.small_font,
                self.window_height // 2 + self.square_size * 2,
                baseline_is_centre=True, width=self.window_width,
            )
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return entered_text.strip()
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_BACKSPACE:
                        entered_text = entered_text[:-1]
                    elif event.unicode.isprintable():
                        entered_text += event.unicode

    def _show_temporary_message(
        self, lines: list[str], font: pygame.font.Font, milliseconds: int = 2200
    ) -> None:
        """Show a message over the board for a short time.

        Args:
            lines: The lines of text to show.
            font: The font to render with.
            milliseconds: How long to display the message.
        """
        self._draw_message_overlay(lines, font)
        pygame.display.flip()
        pygame.time.wait(milliseconds)

    # ------------------------------------------------------------------
    # Board drawing
    # ------------------------------------------------------------------

    def _draw_position(
        self,
        gamestate: GameState,
        valid_moves: list[Move],
        selected_square: Square | None,
        is_dragging: bool,
        drag_position: tuple[int, int],
    ) -> None:
        """Draw the board, highlights and pieces for the current position.

        Args:
            gamestate: The game state to draw.
            valid_moves: The legal moves, used to highlight destinations.
            selected_square: The currently selected square, if any.
            is_dragging: Whether a piece is being dragged with the mouse.
            drag_position: The current mouse position while dragging.
        """
        self._draw_board()
        self._highlight_last_move(gamestate)
        self._highlight_selection(gamestate, valid_moves, selected_square)
        skip_square = selected_square if is_dragging else None
        self._draw_pieces(gamestate, skip_square)
        if is_dragging and selected_square is not None:
            self._draw_dragged_piece(gamestate, selected_square, drag_position)

    def _draw_board(self) -> None:
        """Draw the alternating light and dark board squares."""
        for row in range(constants.BOARD_DIMENSION):
            for column in range(constants.BOARD_DIMENSION):
                colour = (
                    LIGHT_SQUARE_COLOUR
                    if (row + column) % 2 == 0
                    else DARK_SQUARE_COLOUR
                )
                pygame.draw.rect(self.screen, colour, self._square_rect(row, column))

    def _draw_pieces(
        self, gamestate: GameState, skip_square: Square | None
    ) -> None:
        """Draw every piece except one optionally skipped square.

        Args:
            gamestate: The game state holding the board.
            skip_square: A square not to draw, used for the dragged piece.
        """
        for row in range(constants.BOARD_DIMENSION):
            for column in range(constants.BOARD_DIMENSION):
                if (row, column) == skip_square:
                    continue
                piece = gamestate.board[row][column]
                if piece != constants.EMPTY_SQUARE:
                    self.screen.blit(
                        self.piece_images[piece], self._square_rect(row, column)
                    )

    def _draw_dragged_piece(
        self,
        gamestate: GameState,
        selected_square: Square,
        drag_position: tuple[int, int],
    ) -> None:
        """Draw the piece being dragged centred on the mouse.

        Args:
            gamestate: The game state holding the board.
            selected_square: The square the dragged piece came from.
            drag_position: The current mouse position.
        """
        piece = gamestate.board[selected_square[0]][selected_square[1]]
        if piece == constants.EMPTY_SQUARE:
            return
        half = self.square_size // 2
        self.screen.blit(
            self.piece_images[piece],
            (drag_position[0] - half, drag_position[1] - half),
        )

    def _highlight_last_move(self, gamestate: GameState) -> None:
        """Tint the start and end squares of the most recent move.

        Args:
            gamestate: The game state holding the move log.
        """
        if not gamestate.move_log:
            return
        last_move = gamestate.move_log[-1]
        for square in (
            (last_move.start_row, last_move.start_column),
            (last_move.end_row, last_move.end_column),
        ):
            self._tint_square(square[0], square[1], LAST_MOVE_COLOUR, 140)

    def _highlight_selection(
        self,
        gamestate: GameState,
        valid_moves: list[Move],
        selected_square: Square | None,
    ) -> None:
        """Highlight the selected piece and the squares it may move to.

        Args:
            gamestate: The current game state.
            valid_moves: The legal moves for the side to move.
            selected_square: The currently selected square, if any.
        """
        if selected_square is None or not self._belongs_to_mover(
            gamestate, selected_square
        ):
            return
        row, column = selected_square
        self._tint_square(row, column, SELECTED_SQUARE_COLOUR, 160)
        for move in valid_moves:
            if move.start_row == row and move.start_column == column:
                self._tint_square(
                    move.end_row, move.end_column, LEGAL_MOVE_COLOUR, 130
                )

    def _tint_square(
        self, row: int, column: int, colour: pygame.Color, alpha: int
    ) -> None:
        """Draw a translucent colour over a single square.

        Args:
            row: The square's row.
            column: The square's column.
            colour: The tint colour.
            alpha: The opacity, from 0 (clear) to 255 (solid).
        """
        overlay = pygame.Surface((self.square_size, self.square_size))
        overlay.set_alpha(alpha)
        overlay.fill(colour)
        self.screen.blit(overlay, (column * self.square_size, row * self.square_size))

    def _square_rect(self, row: int, column: int) -> pygame.Rect:
        """Return the pixel rectangle of a board square.

        Args:
            row: The square's row.
            column: The square's column.

        Returns:
            The rectangle covering the square.
        """
        return pygame.Rect(
            column * self.square_size,
            row * self.square_size,
            self.square_size,
            self.square_size,
        )

    def _animate_move(self, gamestate: GameState, move: Move) -> None:
        """Glide a piece from its start to its end square.

        Args:
            gamestate: The game state before the move is made.
            move: The move to animate.
        """
        delta_row = move.end_row - move.start_row
        delta_column = move.end_column - move.start_column
        total_frames = (
            abs(delta_row) + abs(delta_column)
        ) * constants.ANIMATION_FRAMES_PER_SQUARE
        if total_frames == 0:
            return

        for frame in range(total_frames + 1):
            progress = frame / total_frames
            current_row = move.start_row + delta_row * progress
            current_column = move.start_column + delta_column * progress

            self._draw_board()
            self._highlight_last_move(gamestate)
            self._draw_pieces(gamestate, skip_square=None)

            end_rect = self._square_rect(move.end_row, move.end_column)
            end_colour = (
                LIGHT_SQUARE_COLOUR
                if (move.end_row + move.end_column) % 2 == 0
                else DARK_SQUARE_COLOUR
            )
            pygame.draw.rect(self.screen, end_colour, end_rect)
            if move.piece_captured != constants.EMPTY_SQUARE:
                self.screen.blit(self.piece_images[move.piece_captured], end_rect)
            start_rect = self._square_rect(move.start_row, move.start_column)
            start_colour = (
                LIGHT_SQUARE_COLOUR
                if (move.start_row + move.start_column) % 2 == 0
                else DARK_SQUARE_COLOUR
            )
            pygame.draw.rect(self.screen, start_colour, start_rect)

            self.screen.blit(
                self.piece_images[move.piece_moved],
                (current_column * self.square_size, current_row * self.square_size),
            )
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND * 2)

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def _action_button_rect(self) -> pygame.Rect:
        """Return the rectangle of the panel's action button.

        Returns:
            The rectangle for the resign or save button.
        """
        padding = self.square_size // 4
        height = int(self.square_size * 0.7)
        return pygame.Rect(
            self.board_size + padding,
            self.window_height - height - padding,
            self.panel_width - 2 * padding,
            height,
        )

    def _draw_panel(
        self,
        gamestate: GameState,
        san_moves: list[str],
        player_to_move: str,
        saved_pgn_name: str | None,
    ) -> None:
        """Draw the side panel: move list, status and action button.

        Args:
            gamestate: The current game state.
            san_moves: The game's moves in SAN.
            player_to_move: The player type whose turn it is.
            saved_pgn_name: The name of the saved PGN file, if any.
        """
        panel = pygame.Rect(self.board_size, 0, self.panel_width, self.window_height)
        pygame.draw.rect(self.screen, PANEL_BACKGROUND_COLOUR, panel)
        padding = self.square_size // 4
        left = self.board_size + padding

        title = self.medium_font.render("Moves", True, TEXT_COLOUR)
        self.screen.blit(title, (left, padding))

        self._draw_move_list(san_moves, left, padding + title.get_height() + padding)
        self._draw_panel_status(
            gamestate, player_to_move, saved_pgn_name, left
        )
        self._draw_action_button(gamestate)

    def _draw_move_list(self, san_moves: list[str], left: int, top: int) -> None:
        """Draw the most recent moves as a numbered list.

        Args:
            san_moves: The moves in SAN.
            left: The left pixel coordinate to draw at.
            top: The top pixel coordinate to draw at.
        """
        row_height = self.move_font.get_height() + 4
        bottom = self.window_height - int(self.square_size * 1.6)
        visible_rows = max(1, (bottom - top) // row_height)

        pairs = [
            (
                index // 2 + 1,
                san_moves[index],
                san_moves[index + 1] if index + 1 < len(san_moves) else "",
            )
            for index in range(0, len(san_moves), 2)
        ]
        for offset, (number, white_move, black_move) in enumerate(
            pairs[-visible_rows:]
        ):
            line = f"{number:>3}. {white_move:<7} {black_move}"
            rendered = self.move_font.render(line, True, TEXT_COLOUR)
            self.screen.blit(rendered, (left, top + offset * row_height))

    def _draw_panel_status(
        self,
        gamestate: GameState,
        player_to_move: str,
        saved_pgn_name: str | None,
        left: int,
    ) -> None:
        """Draw the status line above the action button.

        Args:
            gamestate: The current game state.
            player_to_move: The player type whose turn it is.
            saved_pgn_name: The name of the saved PGN file, if any.
            left: The left pixel coordinate to draw at.
        """
        status_top = self.window_height - int(self.square_size * 1.5)
        if gamestate.is_game_over():
            message = gamestate.outcome_description()
        else:
            mover = "White" if gamestate.white_to_move else "Black"
            label = opponent.PLAYER_LABELS[player_to_move]
            message = f"{mover} to move ({label})"
        self.screen.blit(
            self.small_font.render(message, True, MUTED_TEXT_COLOUR), (left, status_top)
        )
        if saved_pgn_name is not None:
            self.screen.blit(
                self.small_font.render(f"Saved {saved_pgn_name}", True, LEGAL_MOVE_COLOUR),
                (left, status_top + self.small_font.get_height() + 2),
            )

    def _draw_action_button(self, gamestate: GameState) -> None:
        """Draw the resign or save button depending on the game state.

        Args:
            gamestate: The current game state.
        """
        rectangle = self._action_button_rect()
        label = "Save as PGN" if gamestate.is_game_over() else "Resign"
        pygame.draw.rect(self.screen, ACTION_BUTTON_COLOUR, rectangle, border_radius=6)
        rendered = self.small_font.render(label, True, TEXT_COLOUR)
        self.screen.blit(
            rendered,
            (
                rectangle.centerx - rendered.get_width() // 2,
                rectangle.centery - rendered.get_height() // 2,
            ),
        )

    def _draw_game_over_banner(self, gamestate: GameState) -> None:
        """Draw the result of a finished game over the board.

        Args:
            gamestate: The finished game state.
        """
        self._draw_message_overlay(
            [gamestate.outcome_description(), "Press R for menu or Q to quit"],
            self.large_font,
            secondary_font=self.small_font,
        )

    def _draw_message_overlay(
        self,
        lines: list[str],
        font: pygame.font.Font,
        secondary_font: pygame.font.Font | None = None,
    ) -> None:
        """Draw lines of text centred over a dimmed board.

        Args:
            lines: The lines to draw; the first uses ``font`` and the rest use
                ``secondary_font`` when given.
            font: The font for the first line.
            secondary_font: The font for the remaining lines, if different.
        """
        overlay = pygame.Surface((self.board_size, self.board_size))
        overlay.set_alpha(170)
        overlay.fill(OVERLAY_COLOUR)
        self.screen.blit(overlay, (0, 0))

        line_height = font.get_height() + self.square_size // 6
        total_height = line_height * len(lines)
        top = (self.board_size - total_height) // 2
        for index, line in enumerate(lines):
            chosen = font if index == 0 or secondary_font is None else secondary_font
            rendered = chosen.render(line, True, TEXT_COLOUR)
            left = (self.board_size - rendered.get_width()) // 2
            self.screen.blit(rendered, (left, top + index * line_height))

    # ------------------------------------------------------------------
    # Main menu drawing
    # ------------------------------------------------------------------

    def _draw_main_menu(
        self, white_player: str, black_player: str
    ) -> dict[str, pygame.Rect]:
        """Draw the main menu and return its clickable regions.

        Args:
            white_player: The player type currently controlling White.
            black_player: The player type currently controlling Black.

        Returns:
            A mapping from action name to the rectangle that triggers it.
        """
        self.screen.fill(BACKGROUND_COLOUR)
        self._blit_centred_text(
            constants.WINDOW_TITLE, self.large_font, self.square_size // 2,
            width=self.window_width,
        )

        mini_me_ready = self.store.model_path.exists()
        status = (
            "A trained Mini-Me is ready to play."
            if mini_me_ready
            else "No Mini-Me yet: play it or train from Lichess."
        )
        rendered_status = self.small_font.render(status, True, MUTED_TEXT_COLOUR)
        self.screen.blit(
            rendered_status,
            ((self.window_width - rendered_status.get_width()) // 2,
             self.square_size + self.square_size // 3),
        )

        button_width = self.window_width // 2
        button_height = int(self.square_size * 0.8)
        button_left = (self.window_width - button_width) // 2
        first_top = int(self.square_size * 2.2)
        spacing = button_height + self.square_size // 5

        rows = (
            ("toggle_white", f"White: {opponent.PLAYER_LABELS[white_player]}", False),
            ("toggle_black", f"Black: {opponent.PLAYER_LABELS[black_player]}", False),
            ("learn_lichess", "Learn from Lichess", False),
            ("start", "Start game", True),
            ("quit", "Quit", True),
        )
        buttons: dict[str, pygame.Rect] = {}
        for index, (action, label, emphasised) in enumerate(rows):
            rectangle = pygame.Rect(
                button_left, first_top + index * spacing, button_width, button_height
            )
            colour = HIGHLIGHTED_PANEL_COLOUR if emphasised else PANEL_COLOUR
            pygame.draw.rect(self.screen, colour, rectangle, border_radius=8)
            self._blit_centred_text(
                label, self.medium_font, rectangle.centery, baseline_is_centre=True,
                width=self.window_width,
            )
            buttons[action] = rectangle
        return buttons

    @staticmethod
    def _button_action_at(
        buttons: dict[str, pygame.Rect], position: tuple[int, int]
    ) -> str | None:
        """Return the action of the button at a position, if any.

        Args:
            buttons: A mapping from action name to clickable rectangle.
            position: The (x, y) pixel position of a click.

        Returns:
            The matching action name, or ``None`` when nothing was clicked.
        """
        for action, rectangle in buttons.items():
            if rectangle.collidepoint(position):
                return action
        return None

    def _blit_centred_text(
        self,
        text: str,
        font: pygame.font.Font,
        vertical_centre: int,
        baseline_is_centre: bool = False,
        width: int | None = None,
    ) -> None:
        """Draw horizontally centred text at a given vertical position.

        Args:
            text: The text to draw.
            font: The font to render with.
            vertical_centre: The vertical pixel coordinate to draw at.
            baseline_is_centre: When True, centre the text vertically on
                ``vertical_centre``; otherwise treat it as the top edge.
            width: The width to centre within; defaults to the board width.
        """
        centre_width = self.board_size if width is None else width
        rendered = font.render(text, True, TEXT_COLOUR)
        left = (centre_width - rendered.get_width()) // 2
        top = (
            vertical_centre - rendered.get_height() // 2
            if baseline_is_centre
            else vertical_centre
        )
        self.screen.blit(rendered, (left, top))


def run_game() -> None:
    """Create and run the chess interface. The program's entry point."""
    ChessInterface().run()
