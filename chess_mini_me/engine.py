"""The chess engine: game state, move generation and the move log.

This module is responsible for the rules of chess. It stores the current
position, generates the legal moves for the side to move (including pins,
checks, castling, en passant and promotion) and supports making and undoing
moves. It deliberately has no dependency on the graphical interface.
"""

from __future__ import annotations

from typing import Any

from chess_mini_me import constants


class CastleRights:
    """The castling rights available to each side.

    Each flag records whether one specific castling move is still legal,
    ignoring temporary obstructions such as a square being attacked.
    """

    def __init__(
        self,
        white_king_side: bool,
        black_king_side: bool,
        white_queen_side: bool,
        black_queen_side: bool,
    ) -> None:
        """Store the four castling-rights flags.

        Args:
            white_king_side: Whether White may still castle king-side.
            black_king_side: Whether Black may still castle king-side.
            white_queen_side: Whether White may still castle queen-side.
            black_queen_side: Whether Black may still castle queen-side.
        """
        self.white_king_side = white_king_side
        self.black_king_side = black_king_side
        self.white_queen_side = white_queen_side
        self.black_queen_side = black_queen_side

    def copy(self) -> "CastleRights":
        """Return an independent copy of these castling rights.

        Returns:
            A new ``CastleRights`` holding the same four flags.
        """
        return CastleRights(
            self.white_king_side,
            self.black_king_side,
            self.white_queen_side,
            self.black_queen_side,
        )


class Move:
    """A single move and the information needed to make or undo it."""

    def __init__(
        self,
        start_square: tuple[int, int],
        end_square: tuple[int, int],
        board: list[list[str]],
        is_en_passant_move: bool = False,
        is_castle_move: bool = False,
    ) -> None:
        """Record a move from one square to another on the given board.

        Args:
            start_square: The (row, column) the piece moves from.
            end_square: The (row, column) the piece moves to.
            board: The board, used to read the moved and captured pieces.
            is_en_passant_move: Whether this move is an en passant capture.
            is_castle_move: Whether this move is a castling move.
        """
        self.start_row, self.start_column = start_square
        self.end_row, self.end_column = end_square

        self.piece_moved = board[self.start_row][self.start_column]
        self.piece_captured = board[self.end_row][self.end_column]

        self.is_en_passant_move = is_en_passant_move
        self.is_castle_move = is_castle_move

        # A pawn reaching the far rank is promoted.
        self.is_pawn_promotion = (
            self.piece_moved == constants.WHITE + constants.PAWN
            and self.end_row == 0
        ) or (
            self.piece_moved == constants.BLACK + constants.PAWN
            and self.end_row == constants.BOARD_DIMENSION - 1
        )

        # A unique identifier built from the start and end coordinates, used to
        # compare moves quickly.
        self.move_id = (
            self.start_row * 1000
            + self.start_column * 100
            + self.end_row * 10
            + self.end_column
        )

    def __eq__(self, other: Any) -> bool:
        """Compare two moves by their start and end coordinates.

        Args:
            other: The object to compare against.

        Returns:
            True if ``other`` is a ``Move`` with the same identifier.
        """
        if isinstance(other, Move):
            return self.move_id == other.move_id
        return False

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``.

        Returns:
            The move identifier, allowing moves to be used in sets and dicts.
        """
        return self.move_id

    def get_chess_notation(self) -> str:
        """Return the move in simple coordinate notation, such as ``e2e4``.

        Returns:
            The start and end squares as a file-and-rank string.
        """
        return self._square_name(
            self.start_row, self.start_column
        ) + self._square_name(self.end_row, self.end_column)

    @staticmethod
    def _square_name(row: int, column: int) -> str:
        """Return the file-and-rank name of a single square.

        Args:
            row: The board row index (0 is the eighth rank).
            column: The board column index (0 is the a-file).

        Returns:
            The square name, for example ``e4``.
        """
        return constants.COLUMNS_TO_FILES[column] + str(
            constants.BOARD_DIMENSION - row
        )


class GameState:
    """The complete state of a game of chess at one moment in time."""

    def __init__(self) -> None:
        """Set up the board and game state for the start of a new game."""
        self.board: list[list[str]] = [
            list(rank) for rank in constants.STARTING_BOARD
        ]

        # Map each piece type to the method that generates its moves. This
        # avoids a long chain of conditionals when generating moves.
        self.move_generators = {
            constants.PAWN: self._generate_pawn_moves,
            constants.ROOK: self._generate_rook_moves,
            constants.KNIGHT: self._generate_knight_moves,
            constants.BISHOP: self._generate_bishop_moves,
            constants.QUEEN: self._generate_queen_moves,
            constants.KING: self._generate_king_moves,
        }

        self.white_to_move = True
        self.move_log: list[Move] = []

        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)

        self.in_check = False
        self.checkmate = False
        self.stalemate = False

        # Pins and checks discovered for the side to move, each stored as
        # (row, column, direction_row, direction_column).
        self.pins: list[tuple[int, int, int, int]] = []
        self.checks: list[tuple[int, int, int, int]] = []

        # The square a pawn may capture into by en passant, if any.
        self.en_passant_target: tuple[int, int] | tuple[()] = ()

        self.current_castling_rights = CastleRights(True, True, True, True)
        self.castle_rights_log = [self.current_castling_rights.copy()]

    # ------------------------------------------------------------------
    # Making and undoing moves
    # ------------------------------------------------------------------

    def make_move(self, move: Move, promotion_piece: str = constants.QUEEN) -> None:
        """Apply a move to the board and update all derived state.

        Args:
            move: The move to make.
            promotion_piece: The piece type a promoting pawn becomes. The
                graphical interface supplies the player's choice; the move
                finder uses the default queen.
        """
        self.board[move.start_row][move.start_column] = constants.EMPTY_SQUARE
        self.board[move.end_row][move.end_column] = move.piece_moved
        self.move_log.append(move)
        self.white_to_move = not self.white_to_move

        if move.piece_moved == constants.WHITE + constants.KING:
            self.white_king_location = (move.end_row, move.end_column)
        elif move.piece_moved == constants.BLACK + constants.KING:
            self.black_king_location = (move.end_row, move.end_column)

        # The pawn captured by en passant sits on the moving pawn's start row,
        # in the destination column, so it is not removed by the move above.
        if move.is_en_passant_move:
            self.board[move.start_row][move.end_column] = constants.EMPTY_SQUARE

        # Record whether the next player may answer with en passant.
        is_two_square_pawn_advance = (
            move.piece_moved[1] == constants.PAWN
            and abs(move.start_row - move.end_row) == 2
        )
        if is_two_square_pawn_advance:
            self.en_passant_target = (
                (move.start_row + move.end_row) // 2,
                move.start_column,
            )
        else:
            self.en_passant_target = ()

        if move.is_pawn_promotion:
            self.board[move.end_row][move.end_column] = (
                move.piece_moved[0] + promotion_piece
            )

        if move.is_castle_move:
            self._move_castling_rook(move, undo=False)

        self._update_castle_rights(move)
        self.castle_rights_log.append(self.current_castling_rights.copy())

    def undo_move(self) -> None:
        """Reverse the most recent move and restore the previous state."""
        if not self.move_log:
            return

        move = self.move_log.pop()
        self.board[move.start_row][move.start_column] = move.piece_moved
        self.board[move.end_row][move.end_column] = move.piece_captured
        self.white_to_move = not self.white_to_move

        if move.piece_moved == constants.WHITE + constants.KING:
            self.white_king_location = (move.start_row, move.start_column)
        elif move.piece_moved == constants.BLACK + constants.KING:
            self.black_king_location = (move.start_row, move.start_column)

        if move.is_en_passant_move:
            # The destination square was empty, so clear it, and restore the
            # captured pawn to the moving pawn's start row.
            self.board[move.end_row][move.end_column] = constants.EMPTY_SQUARE
            captured_pawn = (
                constants.BLACK + constants.PAWN
                if move.piece_moved == constants.WHITE + constants.PAWN
                else constants.WHITE + constants.PAWN
            )
            self.board[move.start_row][move.end_column] = captured_pawn

        if move.is_castle_move:
            self._move_castling_rook(move, undo=True)

        # Restore the previous castling rights.
        self.castle_rights_log.pop()
        self.current_castling_rights = self.castle_rights_log[-1].copy()

        # Undoing a move can never leave the side to move mated or stalemated.
        self.checkmate = False
        self.stalemate = False

    def _move_castling_rook(self, move: Move, undo: bool) -> None:
        """Move the rook that accompanies a castling king move.

        Args:
            move: The castling move whose rook should be moved.
            undo: When True the rook is returned to its corner; otherwise it is
                placed beside the king.
        """
        is_king_side = move.end_column - move.start_column == 2
        row = move.end_row
        if is_king_side:
            corner_column = move.end_column + 1
            beside_king_column = move.end_column - 1
        else:
            corner_column = move.end_column - 2
            beside_king_column = move.end_column + 1

        source, destination = (
            (beside_king_column, corner_column)
            if undo
            else (corner_column, beside_king_column)
        )
        self.board[row][destination] = self.board[row][source]
        self.board[row][source] = constants.EMPTY_SQUARE

    def _update_castle_rights(self, move: Move) -> None:
        """Revoke castling rights affected by a move.

        Castling rights are lost when the king moves, when a rook moves from
        its starting corner, or when a rook is captured on its starting corner.

        Args:
            move: The move that has just been made.
        """
        rights = self.current_castling_rights

        def revoke_for_square(row: int, column: int) -> None:
            """Revoke the rights tied to a rook starting on a given square."""
            if (row, column) == (7, 0):
                rights.white_queen_side = False
            elif (row, column) == (7, 7):
                rights.white_king_side = False
            elif (row, column) == (0, 0):
                rights.black_queen_side = False
            elif (row, column) == (0, 7):
                rights.black_king_side = False

        if move.piece_moved == constants.WHITE + constants.KING:
            rights.white_king_side = False
            rights.white_queen_side = False
        elif move.piece_moved == constants.BLACK + constants.KING:
            rights.black_king_side = False
            rights.black_queen_side = False
        elif move.piece_moved[1] == constants.ROOK:
            revoke_for_square(move.start_row, move.start_column)

        # A rook captured on its home square can also remove castling rights.
        if move.piece_captured[1:] == constants.ROOK:
            revoke_for_square(move.end_row, move.end_column)

    # ------------------------------------------------------------------
    # Legal move generation
    # ------------------------------------------------------------------

    def get_valid_moves(self) -> list[Move]:
        """Return every legal move for the side to move.

        The result accounts for pins, checks, castling and en passant, and the
        ``checkmate`` and ``stalemate`` flags are refreshed as a side effect.

        Returns:
            The list of legal moves; empty when the game has ended.
        """
        # Flags are recomputed from scratch on every call so that they are
        # never left stale by the move finder's deep search.
        self.checkmate = False
        self.stalemate = False

        self.in_check, self.pins, self.checks = self._find_pins_and_checks()

        if self.white_to_move:
            king_row, king_column = self.white_king_location
        else:
            king_row, king_column = self.black_king_location

        if self.in_check:
            moves = self._get_moves_while_in_check(king_row, king_column)
        else:
            moves = self.get_all_possible_moves()
            self._generate_castle_moves(king_row, king_column, moves)

        if not moves:
            if self.in_check:
                self.checkmate = True
            else:
                self.stalemate = True

        return moves

    def _get_moves_while_in_check(
        self, king_row: int, king_column: int
    ) -> list[Move]:
        """Return the legal moves available when the king is in check.

        With a single check the king may move, or the checking piece may be
        captured or blocked. With a double check only the king may move.

        Args:
            king_row: The row of the king of the side to move.
            king_column: The column of the king of the side to move.

        Returns:
            The legal moves that leave the king safe.
        """
        if len(self.checks) > 1:
            moves: list[Move] = []
            self._generate_king_moves(king_row, king_column, moves)
            return moves

        moves = self.get_all_possible_moves()
        check_row, check_column, direction_row, direction_column = self.checks[0]
        checking_piece = self.board[check_row][check_column]

        # The squares that would resolve the check by capture or block.
        if checking_piece[1] == constants.KNIGHT:
            valid_squares = {(check_row, check_column)}
        else:
            valid_squares = set()
            for distance in range(1, constants.BOARD_DIMENSION):
                square = (
                    king_row + direction_row * distance,
                    king_column + direction_column * distance,
                )
                valid_squares.add(square)
                if square == (check_row, check_column):
                    break

        # Keep king moves and any move that lands on a resolving square.
        return [
            move
            for move in moves
            if move.piece_moved[1] == constants.KING
            or (move.end_row, move.end_column) in valid_squares
        ]

    def get_all_possible_moves(self) -> list[Move]:
        """Return every pseudo-legal move for the side to move.

        Pseudo-legal moves obey the movement rules of each piece but may leave
        the king in check; callers must filter those out.

        Returns:
            The list of pseudo-legal moves.
        """
        moves: list[Move] = []
        for row in range(constants.BOARD_DIMENSION):
            for column in range(constants.BOARD_DIMENSION):
                piece = self.board[row][column]
                if piece == constants.EMPTY_SQUARE:
                    continue
                piece_colour = piece[0]
                belongs_to_mover = (
                    piece_colour == constants.WHITE and self.white_to_move
                ) or (piece_colour == constants.BLACK and not self.white_to_move)
                if belongs_to_mover:
                    self.move_generators[piece[1]](row, column, moves)
        return moves

    def _find_pins_and_checks(
        self,
    ) -> tuple[bool, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
        """Find every piece pinned to the king and every piece checking it.

        Returns:
            A tuple of (in_check, pins, checks). Each pin and check is stored as
            (row, column, direction_row, direction_column).
        """
        pins: list[tuple[int, int, int, int]] = []
        checks: list[tuple[int, int, int, int]] = []
        in_check = False

        if self.white_to_move:
            ally_colour, enemy_colour = constants.WHITE, constants.BLACK
            king_row, king_column = self.white_king_location
        else:
            ally_colour, enemy_colour = constants.BLACK, constants.WHITE
            king_row, king_column = self.black_king_location

        # Scan outwards from the king along every straight-line direction.
        for direction in constants.ALL_DIRECTIONS:
            is_diagonal = direction in constants.DIAGONAL_DIRECTIONS
            possible_pin: tuple[int, int, int, int] | tuple[()] = ()

            for distance in range(1, constants.BOARD_DIMENSION):
                end_row = king_row + direction[0] * distance
                end_column = king_column + direction[1] * distance
                if not self._is_on_board(end_row, end_column):
                    break

                end_piece = self.board[end_row][end_column]
                if end_piece == constants.EMPTY_SQUARE:
                    continue

                piece_colour, piece_type = end_piece[0], end_piece[1]
                if piece_colour == ally_colour:
                    if piece_type == constants.KING:
                        # The king is transparent to this scan. When a tentative
                        # king move is tested its stored location is updated but
                        # the board is not, so the king's old square must not be
                        # allowed to block an attacking ray onto the new square.
                        continue
                    # The first friendly piece might be pinned; a second one
                    # behind it means there is no pin along this direction.
                    if possible_pin == ():
                        possible_pin = (
                            end_row,
                            end_column,
                            direction[0],
                            direction[1],
                        )
                        continue
                    break

                # An enemy piece: determine whether it attacks along this line.
                if self._enemy_attacks_along(
                    piece_type, is_diagonal, distance, direction, enemy_colour
                ):
                    if possible_pin == ():
                        in_check = True
                        checks.append(
                            (end_row, end_column, direction[0], direction[1])
                        )
                    else:
                        pins.append(possible_pin)
                break

        # Knights are handled separately because they jump rather than slide.
        for offset in constants.KNIGHT_OFFSETS:
            end_row = king_row + offset[0]
            end_column = king_column + offset[1]
            if not self._is_on_board(end_row, end_column):
                continue
            end_piece = self.board[end_row][end_column]
            if end_piece == enemy_colour + constants.KNIGHT:
                in_check = True
                checks.append((end_row, end_column, offset[0], offset[1]))

        return in_check, pins, checks

    @staticmethod
    def _enemy_attacks_along(
        piece_type: str,
        is_diagonal: bool,
        distance: int,
        direction: tuple[int, int],
        enemy_colour: str,
    ) -> bool:
        """Decide whether an enemy piece attacks the king along a direction.

        Args:
            piece_type: The type of the enemy piece encountered.
            is_diagonal: Whether the scan direction is diagonal.
            distance: How many squares away the enemy piece sits.
            direction: The (row, column) scan direction from the king.
            enemy_colour: The colour of the attacking side.

        Returns:
            True if the enemy piece gives check along this direction.
        """
        if piece_type == constants.QUEEN:
            return True
        if piece_type == constants.ROOK:
            return not is_diagonal
        if piece_type == constants.BISHOP:
            return is_diagonal
        if distance == 1 and piece_type == constants.KING:
            return True
        if distance == 1 and piece_type == constants.PAWN and is_diagonal:
            # A pawn attacks diagonally towards the opposing side. A white pawn
            # attacks upwards (towards row 0), so it checks a king that lies up
            # the board from it, meaning the scan direction has a positive row
            # component. The reverse holds for a black pawn.
            attacks_downward = direction[0] == 1
            return (enemy_colour == constants.WHITE and attacks_downward) or (
                enemy_colour == constants.BLACK and not attacks_downward
            )
        return False

    # ------------------------------------------------------------------
    # Per-piece move generation
    # ------------------------------------------------------------------

    def _get_pin(
        self, row: int, column: int
    ) -> tuple[bool, tuple[int, int] | tuple[()]]:
        """Return whether the piece at a square is pinned, and its pin line.

        A non-queen pin is consumed (removed from ``self.pins``) so that it is
        not applied twice within a single move-generation pass.

        Args:
            row: The row of the piece.
            column: The column of the piece.

        Returns:
            A tuple of (is_pinned, pin_direction); the direction is empty when
            the piece is not pinned.
        """
        for index in range(len(self.pins) - 1, -1, -1):
            pin = self.pins[index]
            if pin[0] == row and pin[1] == column:
                pin_direction = (pin[2], pin[3])
                if self.board[row][column][1] != constants.QUEEN:
                    self.pins.pop(index)
                return True, pin_direction
        return False, ()

    def _generate_pawn_moves(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append every pseudo-legal move for the pawn at a square.

        Args:
            row: The pawn's row.
            column: The pawn's column.
            moves: The list to append generated moves to.
        """
        is_pinned, pin_direction = self._get_pin(row, column)

        if self.white_to_move:
            advance, start_row, enemy_colour = -1, 6, constants.BLACK
        else:
            advance, start_row, enemy_colour = 1, 1, constants.WHITE

        # Forward advances.
        if self.board[row + advance][column] == constants.EMPTY_SQUARE and (
            not is_pinned or pin_direction == (advance, 0)
        ):
            moves.append(Move((row, column), (row + advance, column), self.board))
            two_squares_clear = (
                row == start_row
                and self.board[row + 2 * advance][column] == constants.EMPTY_SQUARE
            )
            if two_squares_clear:
                moves.append(
                    Move((row, column), (row + 2 * advance, column), self.board)
                )

        # Diagonal captures, including en passant.
        for column_step in (-1, 1):
            target_column = column + column_step
            if not 0 <= target_column < constants.BOARD_DIMENSION:
                continue
            if is_pinned and pin_direction != (advance, column_step):
                continue

            target_square = self.board[row + advance][target_column]
            if target_square[0] == enemy_colour:
                moves.append(
                    Move(
                        (row, column),
                        (row + advance, target_column),
                        self.board,
                    )
                )
            elif (row + advance, target_column) == self.en_passant_target:
                moves.append(
                    Move(
                        (row, column),
                        (row + advance, target_column),
                        self.board,
                        is_en_passant_move=True,
                    )
                )

    def _generate_sliding_moves(
        self,
        row: int,
        column: int,
        moves: list[Move],
        directions: tuple[tuple[int, int], ...],
    ) -> None:
        """Append the moves of a sliding piece (rook, bishop or queen).

        Args:
            row: The piece's row.
            column: The piece's column.
            moves: The list to append generated moves to.
            directions: The straight-line directions the piece may travel.
        """
        is_pinned, pin_direction = self._get_pin(row, column)
        enemy_colour = constants.BLACK if self.white_to_move else constants.WHITE

        for direction in directions:
            # A pinned piece may only move along the pinning line.
            if is_pinned and pin_direction not in (
                direction,
                (-direction[0], -direction[1]),
            ):
                continue

            for distance in range(1, constants.BOARD_DIMENSION):
                end_row = row + direction[0] * distance
                end_column = column + direction[1] * distance
                if not self._is_on_board(end_row, end_column):
                    break

                end_piece = self.board[end_row][end_column]
                if end_piece == constants.EMPTY_SQUARE:
                    moves.append(
                        Move((row, column), (end_row, end_column), self.board)
                    )
                elif end_piece[0] == enemy_colour:
                    moves.append(
                        Move((row, column), (end_row, end_column), self.board)
                    )
                    break
                else:
                    break

    def _generate_rook_moves(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append the rook's moves.

        Args:
            row: The rook's row.
            column: The rook's column.
            moves: The list to append generated moves to.
        """
        self._generate_sliding_moves(
            row, column, moves, constants.ORTHOGONAL_DIRECTIONS
        )

    def _generate_bishop_moves(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append the bishop's moves.

        Args:
            row: The bishop's row.
            column: The bishop's column.
            moves: The list to append generated moves to.
        """
        self._generate_sliding_moves(
            row, column, moves, constants.DIAGONAL_DIRECTIONS
        )

    def _generate_queen_moves(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append the queen's moves, combining rook and bishop directions.

        Args:
            row: The queen's row.
            column: The queen's column.
            moves: The list to append generated moves to.
        """
        self._generate_sliding_moves(
            row, column, moves, constants.ALL_DIRECTIONS
        )

    def _generate_knight_moves(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append the knight's moves.

        Args:
            row: The knight's row.
            column: The knight's column.
            moves: The list to append generated moves to.
        """
        is_pinned, _ = self._get_pin(row, column)
        if is_pinned:
            # A pinned knight can never move without exposing the king.
            return

        ally_colour = constants.WHITE if self.white_to_move else constants.BLACK
        for offset in constants.KNIGHT_OFFSETS:
            end_row = row + offset[0]
            end_column = column + offset[1]
            if not self._is_on_board(end_row, end_column):
                continue
            if self.board[end_row][end_column][0] != ally_colour:
                moves.append(
                    Move((row, column), (end_row, end_column), self.board)
                )

    def _generate_king_moves(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append the king's one-square moves that do not leave it in check.

        Args:
            row: The king's row.
            column: The king's column.
            moves: The list to append generated moves to.
        """
        ally_colour = constants.WHITE if self.white_to_move else constants.BLACK

        for direction in constants.ALL_DIRECTIONS:
            end_row = row + direction[0]
            end_column = column + direction[1]
            if not self._is_on_board(end_row, end_column):
                continue
            if self.board[end_row][end_column][0] == ally_colour:
                continue

            # Tentatively move the king and reject the move if it is attacked.
            original_location = (
                self.white_king_location
                if ally_colour == constants.WHITE
                else self.black_king_location
            )
            self._set_king_location(ally_colour, (end_row, end_column))
            in_check, _, _ = self._find_pins_and_checks()
            if not in_check:
                moves.append(
                    Move((row, column), (end_row, end_column), self.board)
                )
            self._set_king_location(ally_colour, original_location)

    def _set_king_location(
        self, colour: str, location: tuple[int, int]
    ) -> None:
        """Update the stored location of a king.

        Args:
            colour: The colour of the king to move.
            location: The new (row, column) of that king.
        """
        if colour == constants.WHITE:
            self.white_king_location = location
        else:
            self.black_king_location = location

    # ------------------------------------------------------------------
    # Castling
    # ------------------------------------------------------------------

    def _generate_castle_moves(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append any legal castling moves for the king at a square.

        Args:
            row: The king's row.
            column: The king's column.
            moves: The list to append generated moves to.
        """
        # Castling is only ever legal from the king's home square. Guarding on
        # this keeps the index arithmetic below in bounds even if the stored
        # castling rights and the king's position were ever to disagree.
        home_square = (7, 4) if self.white_to_move else (0, 4)
        if (row, column) != home_square:
            return

        if self._is_square_attacked(row, column):
            # Castling out of check is never allowed.
            return

        rights = self.current_castling_rights
        can_castle_king_side = (
            rights.white_king_side if self.white_to_move else rights.black_king_side
        )
        can_castle_queen_side = (
            rights.white_queen_side
            if self.white_to_move
            else rights.black_queen_side
        )

        if can_castle_king_side:
            self._generate_king_side_castle(row, column, moves)
        if can_castle_queen_side:
            self._generate_queen_side_castle(row, column, moves)

    def _generate_king_side_castle(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append the king-side castle if the squares are clear and safe.

        Args:
            row: The king's row.
            column: The king's column.
            moves: The list to append generated moves to.
        """
        squares_empty = (
            self.board[row][column + 1] == constants.EMPTY_SQUARE
            and self.board[row][column + 2] == constants.EMPTY_SQUARE
        )
        squares_safe = not self._is_square_attacked(
            row, column + 1
        ) and not self._is_square_attacked(row, column + 2)
        if squares_empty and squares_safe:
            moves.append(
                Move(
                    (row, column),
                    (row, column + 2),
                    self.board,
                    is_castle_move=True,
                )
            )

    def _generate_queen_side_castle(
        self, row: int, column: int, moves: list[Move]
    ) -> None:
        """Append the queen-side castle if the squares are clear and safe.

        Args:
            row: The king's row.
            column: The king's column.
            moves: The list to append generated moves to.
        """
        squares_empty = (
            self.board[row][column - 1] == constants.EMPTY_SQUARE
            and self.board[row][column - 2] == constants.EMPTY_SQUARE
            and self.board[row][column - 3] == constants.EMPTY_SQUARE
        )
        squares_safe = not self._is_square_attacked(
            row, column - 1
        ) and not self._is_square_attacked(row, column - 2)
        if squares_empty and squares_safe:
            moves.append(
                Move(
                    (row, column),
                    (row, column - 2),
                    self.board,
                    is_castle_move=True,
                )
            )

    def _is_square_attacked(self, row: int, column: int) -> bool:
        """Return whether the side not to move attacks a given square.

        Args:
            row: The row of the square to test.
            column: The column of the square to test.

        Returns:
            True if the opponent has a pseudo-legal move onto the square.
        """
        self.white_to_move = not self.white_to_move
        opponent_moves = self.get_all_possible_moves()
        self.white_to_move = not self.white_to_move
        return any(
            move.end_row == row and move.end_column == column
            for move in opponent_moves
        )

    @staticmethod
    def _is_on_board(row: int, column: int) -> bool:
        """Return whether a coordinate lies within the board.

        Args:
            row: The row index to test.
            column: The column index to test.

        Returns:
            True if both indices fall inside the board.
        """
        return (
            0 <= row < constants.BOARD_DIMENSION
            and 0 <= column < constants.BOARD_DIMENSION
        )
