"""The graphical interface for the chess game, built with Pygame.

This module turns the rules in :mod:`chess_mini_me.engine`, the search in
:mod:`chess_mini_me.move_finder` and the learned opponents in
:mod:`chess_mini_me.cloner` into a playable game. It is the only module that
depends on Pygame, so importing the engine for tests or scripting never
requires a display.

The window has a board on the left and a side panel on the right showing the
move list, a status line and an action button. Longer operations such as
downloading a player's games or training a Mini-Me run on a background thread
and show a live progress screen, and results are shown on a notice screen that
stays until the player dismisses it.
"""

from __future__ import annotations

import datetime
import pathlib
import threading

import pygame

from chess_mini_me import constants, dialogs, opponent, pgn
from chess_mini_me.engine import GameState, Move
from chess_mini_me.training import (
    BLACK_WIN_VALUE,
    DRAW_VALUE,
    LEARNING_PROFILE_NAME,
    WHITE_WIN_VALUE,
    StyleRecorder,
    list_profile_names,
    store_for_profile,
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
FOCUSED_FIELD_COLOUR = pygame.Color(96, 110, 90)
ACTION_BUTTON_COLOUR = pygame.Color(150, 80, 70)
PRIMARY_BUTTON_COLOUR = pygame.Color(90, 130, 90)
PROGRESS_BAR_COLOUR = pygame.Color(120, 170, 110)
OVERLAY_COLOUR = pygame.Color(0, 0, 0)

# The smallest a board square may shrink to when the window is made small.
MINIMUM_SQUARE_SIZE = 28

# The side panel's width as a multiple of one board square.
PANEL_SQUARE_COUNT = 3.4


class _BackgroundProgress:
    """Shared state for a task running on a background thread.

    The worker thread updates the textual status, the optional completion
    fraction and any detail lines; the main thread reads them to draw the
    progress screen. Only simple attribute assignments cross the threads, which
    is safe under Python's global interpreter lock.
    """

    def __init__(self) -> None:
        """Start in an in-progress state with no detail."""
        self.status = "Working ..."
        self.fraction: float | None = None
        self.detail_lines: list[str] = []
        self.done = False
        self.result: object | None = None
        self.error: BaseException | None = None


class ChessInterface:
    """Run the chess game window, from the main menu to game over."""

    def __init__(self) -> None:
        """Create the window and load the resources shared across the game."""
        pygame.init()
        self.clock = pygame.time.Clock()
        pygame.display.set_caption(constants.WINDOW_TITLE)

        # The window is resizable; the board, panel, fonts and piece images are
        # all derived from the current window size and recomputed on a resize.
        initial_width, initial_height = self._initial_window_size()
        self._configure_layout(initial_width, initial_height)

        self.games_directory = pathlib.Path.cwd() / "saved_games"
        self.selected_profile = self._initial_profile()

    # ------------------------------------------------------------------
    # Set-up helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_window_size() -> tuple[int, int]:
        """Return a sensible initial window size for the current display.

        Returns:
            A ``(width, height)`` target that the layout is then fitted into.
        """
        try:
            info = pygame.display.Info()
            width, height = info.current_w, info.current_h
        except pygame.error:
            width, height = 0, 0
        if width <= 0 or height <= 0:
            size = constants.FALLBACK_BOARD_SIZE_PIXELS
            return int(size * (1 + PANEL_SQUARE_COUNT / constants.BOARD_DIMENSION)), size
        margin = constants.WINDOW_MARGIN_PIXELS
        return width - margin, height - margin

    def _configure_layout(self, target_width: int, target_height: int) -> None:
        """Recompute the layout to fill a target window size.

        The board is square and the panel is a fixed multiple of a board
        square, so a single square size determines everything. It is chosen as
        the largest that fits the target width and height, then the window,
        fonts and piece images are rebuilt around it.

        Args:
            target_width: The width to fit into, in pixels.
            target_height: The height to fit into, in pixels.
        """
        square_from_height = target_height // constants.BOARD_DIMENSION
        square_from_width = int(
            target_width / (constants.BOARD_DIMENSION + PANEL_SQUARE_COUNT)
        )
        self.square_size = max(
            MINIMUM_SQUARE_SIZE, min(square_from_height, square_from_width)
        )
        self.board_size = self.square_size * constants.BOARD_DIMENSION
        self.panel_width = int(self.square_size * PANEL_SQUARE_COUNT)
        self.window_width = self.board_size + self.panel_width
        self.window_height = self.board_size

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )

        self.large_font = pygame.font.SysFont("helvetica", self.square_size // 2, True)
        self.medium_font = pygame.font.SysFont("helvetica", self.square_size // 3, True)
        self.small_font = pygame.font.SysFont("helvetica", self.square_size // 4)
        self.move_font = pygame.font.SysFont("couriernew", self.square_size // 4)

        self.piece_images = self._load_piece_images()

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

    @staticmethod
    def _initial_profile() -> str | None:
        """Return the Mini-Me profile to select first, if any exist.

        Returns:
            A profile name, preferring the learned 'my-style' profile, or
            ``None`` when no profiles have been created yet.
        """
        names = list_profile_names()
        if not names:
            return None
        if LEARNING_PROFILE_NAME in names:
            return LEARNING_PROFILE_NAME
        return names[0]

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

    def _show_main_menu(self) -> tuple[str, str, str | None] | None:
        """Display the main menu and collect the choice of players.

        Returns:
            A tuple ``(white_player, black_player, mini_me_profile)``, or
            ``None`` if the player closed the window.
        """
        white_player = opponent.PLAYER_HUMAN
        black_player = opponent.PLAYER_COMPUTER

        while True:
            buttons = self._draw_main_menu(white_player, black_player)
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND)

            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    self._configure_layout(event.w, event.h)
                    continue
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    action = self._button_action_at(buttons, event.pos)
                    if action == "toggle_white":
                        white_player = opponent.next_player_type(white_player)
                    elif action == "toggle_black":
                        black_player = opponent.next_player_type(black_player)
                    elif action == "cycle_profile":
                        self._cycle_profile()
                    elif action == "learn_lichess":
                        self._run_lichess_learning()
                    elif action == "start":
                        return white_player, black_player, self.selected_profile
                    elif action == "quit":
                        return None

    def _cycle_profile(self) -> None:
        """Advance the selected Mini-Me profile to the next saved one."""
        names = list_profile_names()
        if not names:
            self.selected_profile = None
            return
        if self.selected_profile not in names:
            self.selected_profile = names[0]
            return
        position = names.index(self.selected_profile)
        self.selected_profile = names[(position + 1) % len(names)]

    def _play_game(
        self, white_player: str, black_player: str, mini_me_profile: str | None
    ) -> bool:
        """Play a single game.

        Args:
            white_player: The player type controlling White.
            black_player: The player type controlling Black.
            mini_me_profile: The Mini-Me profile to play against, if any.

        Returns:
            True to return to the main menu, or False to quit the program.
        """
        gamestate = GameState()
        valid_moves = gamestate.get_valid_moves()
        opponent_store = store_for_profile(mini_me_profile or LEARNING_PROFILE_NAME)
        controller = opponent.OpponentController(opponent_store)
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
                if event.type == pygame.VIDEORESIZE:
                    self._configure_layout(event.w, event.h)
                    continue
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
                pygame.display.flip()
                if not has_learned_from_game:
                    self._learn_from_game(gamestate, recorder)
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
        while True:
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
                if event.type == pygame.VIDEORESIZE:
                    self._configure_layout(event.w, event.h)
                    continue
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
    ) -> str | None:
        """Save the finished game as a PGN file chosen by the player.

        Args:
            gamestate: The finished game.
            player_labels: A mapping from colour to a display name.

        Returns:
            The saved file's name, or ``None`` if the player cancelled.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        suggested = f"chess-mini-me-{timestamp}.pgn"
        path, cancelled = dialogs.ask_save_pgn_path(suggested, self.games_directory)
        if cancelled or path is None:
            return None
        pgn.save_pgn(
            gamestate,
            path,
            white_name=player_labels[constants.WHITE],
            black_name=player_labels[constants.BLACK],
        )
        self._show_notice("Game saved", [f"Saved to:", str(path)])
        return path.name

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
        self, gamestate: GameState, recorder: StyleRecorder
    ) -> None:
        """Teach the 'my-style' Mini-Me from the human moves of the game.

        Args:
            gamestate: The finished game state.
            recorder: The recorder holding the human's moves.
        """
        if not recorder.has_moves():
            return

        examples = recorder.finalise(self._game_result_value(gamestate))
        epochs = 4
        store = store_for_profile(LEARNING_PROFILE_NAME)

        def worker(progress: _BackgroundProgress) -> int:
            """Train the learning profile, reporting progress per epoch."""
            from chess_mini_me.training import learn_from_examples

            def report(epoch: int, loss: float) -> None:
                progress.fraction = (epoch + 1) / epochs
                progress.status = f"Learning your style (epoch {epoch + 1}/{epochs})"
                progress.detail_lines = [f"training loss {loss:.4f}"]

            learn_from_examples(
                store, examples, epochs=epochs, progress_callback=report
            )
            return len(examples)

        progress = self._run_with_progress("Mini-Me is learning", worker)
        if isinstance(progress.error, ImportError):
            self._show_notice(
                "PyTorch is needed to learn",
                [
                    "The Mini-Me learns with PyTorch, which is not installed.",
                    "",
                    "Install it, then play again to teach your Mini-Me:",
                    "    pip install -r requirements-mini-me.txt",
                    "or  pip install torch --index-url",
                    "        https://download.pytorch.org/whl/cpu",
                ],
            )
            return
        if progress.error is not None:
            self._show_notice("Learning failed", [str(progress.error)])
            return

        moves_learned = progress.result
        self._refresh_selected_profile_after_learning()
        self._show_notice(
            "Mini-Me updated",
            [
                f"Learned from {moves_learned} of your moves.",
                "",
                f"Select the '{LEARNING_PROFILE_NAME}' profile in the menu",
                "to play against your own clone.",
            ],
        )

    def _refresh_selected_profile_after_learning(self) -> None:
        """Select the learning profile if no profile was selected before."""
        if self.selected_profile is None:
            self.selected_profile = LEARNING_PROFILE_NAME

    def _run_lichess_learning(self) -> None:
        """Collect Lichess settings, then download, analyse and train."""
        settings = self._lichess_config_screen()
        if settings is None:
            return

        username = settings["username"]
        profile_name = settings["profile"]
        maximum_games = settings["games"]
        rated_only = settings["rated"]
        epochs = 8

        def worker(progress: _BackgroundProgress) -> dict[str, object]:
            """Download, analyse and train, reporting progress as it goes."""
            from chess_mini_me import lichess
            from chess_mini_me.training import learn_from_examples

            progress.status = f"Downloading up to {maximum_games} games ..."
            progress.detail_lines = [f"from lichess.org/@/{username}"]
            pgn_text = lichess.download_games(
                username, max_games=maximum_games, rated_only=rated_only
            )

            progress.status = "Analysing playing style ..."
            progress.detail_lines = []
            games = lichess.analyse_games(pgn_text, username)
            examples = lichess.extract_examples(pgn_text, username)
            if len(examples) == 0:
                return {"summary": ["No games were found for that player."], "trained": 0}

            def report(epoch: int, loss: float) -> None:
                progress.fraction = (epoch + 1) / epochs
                progress.status = f"Training the Mini-Me (epoch {epoch + 1}/{epochs})"
                progress.detail_lines = [f"training loss {loss:.4f}"]

            learn_from_examples(
                store_for_profile(profile_name), examples, epochs=epochs,
                progress_callback=report,
            )
            return {
                "summary": lichess.summarise_style(games).splitlines(),
                "trained": len(examples),
            }

        progress = self._run_with_progress(
            f"Building Mini-Me from {username}", worker
        )

        if isinstance(progress.error, ImportError):
            self._show_notice(
                "Extra libraries are needed",
                [
                    "Importing from Lichess needs a few extra libraries:",
                    "    pip install -r requirements-mini-me.txt",
                    "",
                    str(progress.error),
                ],
            )
            return
        if isinstance(progress.error, ValueError):
            self._show_notice("Could not import games", [str(progress.error)])
            return
        if progress.error is not None:
            self._show_notice("Import failed", [str(progress.error)])
            return

        outcome = progress.result
        trained = outcome["trained"]
        if trained == 0:
            self._show_notice("No games found", outcome["summary"])
            return

        self.selected_profile = profile_name
        self._show_notice(
            f"Mini-Me '{profile_name}' is ready",
            [f"Trained on {trained} of {username}'s moves.", ""]
            + list(outcome["summary"]),
        )

    # ------------------------------------------------------------------
    # Background progress and notices
    # ------------------------------------------------------------------

    def _run_with_progress(
        self, title: str, worker
    ) -> _BackgroundProgress:
        """Run a worker on a background thread while showing progress.

        Args:
            title: The heading shown on the progress screen.
            worker: A callable taking a :class:`_BackgroundProgress` to update.

        Returns:
            The completed :class:`_BackgroundProgress`, holding the result or
            the error raised by the worker.
        """
        progress = _BackgroundProgress()

        def run() -> None:
            try:
                progress.result = worker(progress)
            except BaseException as error:  # noqa: BLE001 - reported to the user
                progress.error = error
            finally:
                progress.done = True

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        quit_requested = False
        frame = 0
        while not progress.done:
            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    self._configure_layout(event.w, event.h)
                    continue
                if event.type == pygame.QUIT:
                    quit_requested = True
            self._draw_progress_screen(title, progress, frame)
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND)
            frame += 1

        if quit_requested:
            # Let the enclosing loop see the close request and exit cleanly.
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return progress

    def _draw_progress_screen(
        self, title: str, progress: _BackgroundProgress, frame: int
    ) -> None:
        """Draw the progress screen for a running background task.

        Args:
            title: The heading to show.
            progress: The current progress state.
            frame: The current frame counter, used for the spinner.
        """
        self.screen.fill(BACKGROUND_COLOUR)
        centre_y = self.window_height // 3
        self._blit_centred_text(
            title, self.large_font, centre_y, baseline_is_centre=True,
            width=self.window_width,
        )

        dots = "." * (1 + (frame // 8) % 3)
        self._blit_centred_text(
            progress.status + dots, self.medium_font, centre_y + self.square_size,
            baseline_is_centre=True, width=self.window_width,
        )

        bar = pygame.Rect(
            self.window_width // 4, centre_y + self.square_size * 2,
            self.window_width // 2, self.square_size // 2,
        )
        pygame.draw.rect(self.screen, PANEL_COLOUR, bar, border_radius=6)
        if progress.fraction is not None:
            filled = bar.copy()
            filled.width = int(bar.width * max(0.0, min(1.0, progress.fraction)))
            pygame.draw.rect(self.screen, PROGRESS_BAR_COLOUR, filled, border_radius=6)
        else:
            # An indeterminate sweep when there is no measurable fraction.
            sweep_width = bar.width // 4
            position = (frame * 6) % (bar.width + sweep_width) - sweep_width
            sweep = pygame.Rect(
                bar.left + max(0, position), bar.top,
                min(sweep_width, bar.width), bar.height,
            )
            pygame.draw.rect(self.screen, PROGRESS_BAR_COLOUR, sweep, border_radius=6)

        for index, line in enumerate(progress.detail_lines[:3]):
            self._blit_centred_text(
                line, self.small_font,
                bar.bottom + self.square_size // 2 + index * self.small_font.get_height(),
                baseline_is_centre=True, width=self.window_width,
            )

    def _show_notice(self, title: str, lines: list[str]) -> None:
        """Show a message that stays until the player dismisses it.

        Args:
            title: The heading of the notice.
            lines: The body lines of the notice.
        """
        while True:
            close_button = pygame.Rect(
                (self.window_width - self.square_size * 2) // 2,
                self.window_height - int(self.square_size * 1.4),
                self.square_size * 2,
                self.square_size,
            )
            self.screen.fill(BACKGROUND_COLOUR)
            self._blit_centred_text(
                title, self.large_font, self.square_size,
                width=self.window_width,
            )
            top = self.square_size * 3
            for index, line in enumerate(lines):
                rendered = self.small_font.render(line, True, TEXT_COLOUR)
                self.screen.blit(
                    rendered,
                    (
                        (self.window_width - rendered.get_width()) // 2,
                        top + index * (self.small_font.get_height() + 4),
                    ),
                )
            pygame.draw.rect(
                self.screen, PRIMARY_BUTTON_COLOUR, close_button, border_radius=8
            )
            self._blit_centred_text(
                "Close", self.medium_font, close_button.centery,
                baseline_is_centre=True, width=self.window_width,
            )
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND)

            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    self._configure_layout(event.w, event.h)
                    continue
                if event.type == pygame.QUIT:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    return
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_SPACE
                ):
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and close_button.collidepoint(
                    event.pos
                ):
                    return

    # ------------------------------------------------------------------
    # Lichess configuration screen
    # ------------------------------------------------------------------

    def _lichess_config_screen(self) -> dict[str, object] | None:
        """Collect the settings for a Lichess import.

        Returns:
            A settings dictionary with ``username``, ``profile``, ``games`` and
            ``rated`` keys, or ``None`` if the player cancelled.
        """
        username = ""
        profile = ""
        games = 200
        rated_only = True
        focus = "username"
        game_steps = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]

        while True:
            controls = self._draw_lichess_config(
                username, profile, games, rated_only, focus
            )
            pygame.display.flip()
            self.clock.tick(constants.MAXIMUM_FRAMES_PER_SECOND)

            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    self._configure_layout(event.w, event.h)
                    continue
                if event.type == pygame.QUIT:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if controls["username"].collidepoint(event.pos):
                        focus = "username"
                    elif controls["profile"].collidepoint(event.pos):
                        focus = "profile"
                    elif controls["fewer"].collidepoint(event.pos):
                        games = self._step_value(games, game_steps, -1)
                    elif controls["more"].collidepoint(event.pos):
                        games = self._step_value(games, game_steps, 1)
                    elif controls["rated"].collidepoint(event.pos):
                        rated_only = not rated_only
                    elif controls["start"].collidepoint(event.pos):
                        if username.strip():
                            return {
                                "username": username.strip(),
                                "profile": (profile.strip() or username.strip()),
                                "games": games,
                                "rated": rated_only,
                            }
                    elif controls["cancel"].collidepoint(event.pos):
                        return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_TAB:
                        focus = "profile" if focus == "username" else "username"
                    elif event.key == pygame.K_RETURN:
                        if username.strip():
                            return {
                                "username": username.strip(),
                                "profile": (profile.strip() or username.strip()),
                                "games": games,
                                "rated": rated_only,
                            }
                    elif event.key == pygame.K_BACKSPACE:
                        if focus == "username":
                            username = username[:-1]
                        else:
                            profile = profile[:-1]
                    elif event.unicode.isprintable():
                        if focus == "username":
                            username += event.unicode
                        else:
                            profile += event.unicode

    @staticmethod
    def _step_value(value: int, steps: list[int], direction: int) -> int:
        """Return the next value when stepping through a list of choices.

        Args:
            value: The current value.
            steps: The ordered list of allowed values.
            direction: ``-1`` to step down or ``1`` to step up.

        Returns:
            The neighbouring allowed value, clamped to the ends.
        """
        if value in steps:
            index = steps.index(value)
        else:
            index = min(range(len(steps)), key=lambda i: abs(steps[i] - value))
        index = max(0, min(len(steps) - 1, index + direction))
        return steps[index]

    def _draw_lichess_config(
        self, username: str, profile: str, games: int, rated_only: bool, focus: str
    ) -> dict[str, pygame.Rect]:
        """Draw the Lichess configuration screen and return its controls.

        Args:
            username: The current username text.
            profile: The current profile-name text.
            games: The chosen number of games.
            rated_only: Whether only rated games are imported.
            focus: Which text field currently has focus.

        Returns:
            A mapping from control name to its clickable rectangle.
        """
        self.screen.fill(BACKGROUND_COLOUR)
        self._blit_centred_text(
            "Train a Mini-Me from Lichess", self.large_font, self.square_size // 2,
            width=self.window_width,
        )

        left = self.window_width // 6
        field_width = self.window_width * 2 // 3
        field_height = self.square_size
        row_gap = int(field_height * 1.5)
        top = int(self.square_size * 2)
        controls: dict[str, pygame.Rect] = {}

        username_box = pygame.Rect(left, top, field_width, field_height)
        self._draw_field("Lichess username", username or "_", username_box,
                         focus == "username")
        controls["username"] = username_box

        profile_box = pygame.Rect(left, top + row_gap, field_width, field_height)
        placeholder = profile or (username or "(defaults to username)")
        self._draw_field("Save as profile", placeholder, profile_box,
                         focus == "profile")
        controls["profile"] = profile_box

        games_top = top + 2 * row_gap
        self.screen.blit(
            self.small_font.render("Games to import", True, MUTED_TEXT_COLOUR),
            (left, games_top - self.small_font.get_height() - 2),
        )
        fewer = pygame.Rect(left, games_top, field_height, field_height)
        more = pygame.Rect(left + field_width - field_height, games_top,
                           field_height, field_height)
        self._draw_text_button(fewer, "-")
        self._draw_text_button(more, "+")
        value_box = pygame.Rect(
            fewer.right + 8, games_top, more.left - fewer.right - 16, field_height
        )
        pygame.draw.rect(self.screen, PANEL_COLOUR, value_box, border_radius=6)
        self._blit_text_in_rect(str(games), self.medium_font, value_box)
        controls["fewer"] = fewer
        controls["more"] = more

        rated_top = games_top + row_gap
        rated_box = pygame.Rect(left, rated_top, field_width, field_height)
        self._draw_text_button(
            rated_box, f"Rated games only: {'Yes' if rated_only else 'No'}"
        )
        controls["rated"] = rated_box

        button_width = field_width // 2 - 8
        button_top = rated_top + row_gap
        start = pygame.Rect(left, button_top, button_width, field_height)
        cancel = pygame.Rect(left + field_width - button_width, button_top,
                             button_width, field_height)
        pygame.draw.rect(self.screen, PRIMARY_BUTTON_COLOUR, start, border_radius=8)
        self._blit_text_in_rect("Start", self.medium_font, start)
        pygame.draw.rect(self.screen, PANEL_COLOUR, cancel, border_radius=8)
        self._blit_text_in_rect("Cancel", self.medium_font, cancel)
        controls["start"] = start
        controls["cancel"] = cancel
        return controls

    def _draw_field(
        self, label: str, value: str, rectangle: pygame.Rect, focused: bool
    ) -> None:
        """Draw a labelled text field.

        Args:
            label: The label shown above the field.
            value: The text shown inside the field.
            rectangle: The field rectangle.
            focused: Whether the field currently has focus.
        """
        self.screen.blit(
            self.small_font.render(label, True, MUTED_TEXT_COLOUR),
            (rectangle.left, rectangle.top - self.small_font.get_height() - 2),
        )
        colour = FOCUSED_FIELD_COLOUR if focused else PANEL_COLOUR
        pygame.draw.rect(self.screen, colour, rectangle, border_radius=6)
        rendered = self.medium_font.render(value, True, TEXT_COLOUR)
        self.screen.blit(
            rendered,
            (rectangle.left + 10, rectangle.centery - rendered.get_height() // 2),
        )

    def _draw_text_button(self, rectangle: pygame.Rect, label: str) -> None:
        """Draw a simple labelled button.

        Args:
            rectangle: The button rectangle.
            label: The button text.
        """
        pygame.draw.rect(self.screen, PANEL_COLOUR, rectangle, border_radius=6)
        self._blit_text_in_rect(label, self.medium_font, rectangle)

    def _blit_text_in_rect(
        self, text: str, font: pygame.font.Font, rectangle: pygame.Rect
    ) -> None:
        """Draw text centred inside a rectangle.

        Args:
            text: The text to draw.
            font: The font to render with.
            rectangle: The rectangle to centre within.
        """
        rendered = font.render(text, True, TEXT_COLOUR)
        self.screen.blit(
            rendered,
            (
                rectangle.centerx - rendered.get_width() // 2,
                rectangle.centery - rendered.get_height() // 2,
            ),
        )

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
        self._draw_panel_status(gamestate, player_to_move, saved_pgn_name, left)
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
        self._blit_text_in_rect(label, self.small_font, rectangle)

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
            constants.WINDOW_TITLE, self.large_font, self.square_size // 3,
            width=self.window_width,
        )

        profile_label = self.selected_profile or "none yet"
        status = f"Mini-Me profile: {profile_label}"
        rendered_status = self.small_font.render(status, True, MUTED_TEXT_COLOUR)
        self.screen.blit(
            rendered_status,
            ((self.window_width - rendered_status.get_width()) // 2, self.square_size),
        )

        button_width = self.window_width // 2
        button_height = int(self.square_size * 0.72)
        button_left = (self.window_width - button_width) // 2
        first_top = int(self.square_size * 1.7)
        spacing = button_height + self.square_size // 6

        rows = (
            ("toggle_white", f"White: {opponent.PLAYER_LABELS[white_player]}", False),
            ("toggle_black", f"Black: {opponent.PLAYER_LABELS[black_player]}", False),
            ("cycle_profile", f"Mini-Me: {profile_label}", False),
            ("learn_lichess", "Train from Lichess", False),
            ("start", "Start game", True),
            ("quit", "Quit", True),
        )
        buttons: dict[str, pygame.Rect] = {}
        for index, (action, label, emphasised) in enumerate(rows):
            rectangle = pygame.Rect(
                button_left, first_top + index * spacing, button_width, button_height
            )
            colour = PRIMARY_BUTTON_COLOUR if emphasised else PANEL_COLOUR
            pygame.draw.rect(self.screen, colour, rectangle, border_radius=8)
            self._blit_text_in_rect(label, self.medium_font, rectangle)
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
