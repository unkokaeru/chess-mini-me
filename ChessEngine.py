"""
The chess engine, responsible for storing the game state, determining valid moves and keeping a move log.
"""

# Imports

from __future__ import annotations
from typing import Any


class GameState:
    """
    Defines the current state of the chess game
    """

    def __init__(self) -> None:
        """
        Initialise the board
        :return: None
        """

        # The board is an 8x8 2d list, each element of the list has 2 characters.
        # The first character represents the colour of the piece, "b" or "w"
        # The second character represents the type of the piece, "K", "Q", "R", "B", "N" or "P"
        # "--" represents an empty space with no piece

        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
        ]

        self.moveFunctions = {
            "P": self.get_pawn_moves,
            "R": self.get_rook_moves,
            "N": self.get_knight_moves,
            "B": self.get_bishop_moves,
            "Q": self.get_queen_moves,
            "K": self.get_king_moves,
        }

        # Establish which player's turn it is (to begin)
        self.whiteToMove = True
        self.moveLog: list = []

        # Keep track of the king's location
        self.whiteKingLocation = (7, 4)
        self.blackKingLocation = (0, 4)

        # Keep track of check, checkmate and stalemate
        self.checkmate = False
        self.stalemate = False

        # Keep track of pins and checks
        self.pins: list = []
        self.checks: list = []

        # Keep track of en passant moves
        self.enPassantPossible: tuple = ()

        # Keep track of castling rights
        self.currentCastlingRights = CastleRights(True, True, True, True)
        self.castleRightsLog = [
            CastleRights(
                self.currentCastlingRights.wks,
                self.currentCastlingRights.bks,
                self.currentCastlingRights.wqs,
                self.currentCastlingRights.bqs,
            )
        ]

    def make_move(self, move: Move) -> None:
        """
        Make a move by changing the board accordingly
        :param move: Move
        :return: None
        """

        # Update the board: move the piece, add the move to the log, swap players
        self.board[move.startRow][move.startCol] = "--"
        self.board[move.endRow][move.endCol] = move.pieceMoved
        self.moveLog.append(move)
        self.whiteToMove = not self.whiteToMove

        # Update the king's location if moved
        if move.pieceMoved == "wK":
            self.whiteKingLocation = (move.endRow, move.endCol)
        elif move.pieceMoved == "bK":
            self.blackKingLocation = (move.endRow, move.endCol)

        # En passant move
        if move.isEnPassantMove:
            direction = (
                -1 if self.whiteToMove else 1
            )  # Determine direction based on whose turn it is
            rowOfCapturedPawn = (
                move.endRow + direction
            )  # Calculate the row of the captured pawn
            self.board[move.endRow][
                move.endCol
            ] = move.pieceMoved  # Move the attacking pawn
            self.board[move.startRow][
                move.startCol
            ] = "--"  # Remove the attacking pawn from its original position
            self.board[rowOfCapturedPawn][
                move.endCol
            ] = "--"  # Remove the captured pawn

        # Update en passant rights
        if move.pieceMoved[1] == "P" and abs(move.startRow - move.endRow) == 2:
            self.enPassantPossible = ((move.startRow + move.endRow) // 2, move.startCol)
        else:
            self.enPassantPossible = ()

        # Pawn promotion
        if move.isPawnPromotion:
            promoted_piece = input(
                "Promote to Q, R, B or N: "
            )  # TODO: add a GUI to select the piece TODO: add error handling
            self.board[move.endRow][move.endCol] = move.pieceMoved[0] + promoted_piece

        # Castle move
        if move.castle:
            # Kingside castle move
            if move.endCol - move.startCol == 2:
                # Moves the rook
                self.board[move.endRow][move.endCol - 1] = self.board[move.endRow][
                    move.endCol + 1
                ]

                # Erases the old rook
                self.board[move.endRow][move.endCol + 1] = "--"

            # Queenside castle move
            else:
                # Moves the rook
                self.board[move.endRow][move.endCol + 1] = self.board[move.endRow][
                    move.endCol - 2
                ]

                # Erases the old rook
                self.board[move.endRow][move.endCol - 2] = "--"

        # Update castling rights
        self.update_castle_rights(move)
        self.castleRightsLog.append(
            CastleRights(
                self.currentCastlingRights.wks,
                self.currentCastlingRights.bks,
                self.currentCastlingRights.wqs,
                self.currentCastlingRights.bqs,
            )
        )

    def undo_move(self) -> None:
        """
        Undo the last move made
        :return: None
        """

        # Check if there are any moves to undo
        if len(self.moveLog) != 0:
            # Undo the last move
            move = self.moveLog.pop()
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove

            # Update the king's location if moved
            if move.pieceMoved == "wK":
                self.whiteKingLocation = (move.endRow, move.endCol)
            elif move.pieceMoved == "bK":
                self.blackKingLocation = (move.endRow, move.endCol)

            # Undo en passant move
            if move.isEnPassantMove:
                # Place the attacking pawn back in its original position
                self.board[move.startRow][move.startCol] = move.pieceMoved

                # Clear the capture square
                self.board[move.endRow][move.endCol] = "--"

                # The captured pawn is restored to its original position which is one row behind the end row of the move for the attacking pawn
                captured_pawn_row = move.endRow + (1 if move.pieceMoved == "wP" else -1)

                # Restore the captured pawn
                self.board[captured_pawn_row][move.endCol] = (
                    "bP" if move.pieceMoved == "wP" else "wP"
                )

            # Undo castling move
            if move.castle:
                # Undo kingside castle move
                if move.endCol - move.startCol == 2:
                    # Moves the rook back
                    self.board[move.endRow][move.endCol + 1] = self.board[move.endRow][
                        move.endCol - 1
                    ]

                    # Erases the old rook
                    self.board[move.endRow][move.endCol - 1] = "--"  # erase old rook

                # Undo queenside castle move
                else:
                    # Moves the rook back
                    self.board[move.endRow][move.endCol - 2] = self.board[move.endRow][
                        move.endCol + 1
                    ]

                    # Erases the old rook
                    self.board[move.endRow][move.endCol + 1] = "--"

            # Undo castling rights
            newRights = self.castleRightsLog[-2]
            self.castleRightsLog.pop()
            self.currentCastlingRights = CastleRights(
                newRights.wks, newRights.bks, newRights.wqs, newRights.bqs
            )

    def update_castle_rights(self, move: Move) -> None:
        """
        Update the castle rights given the move
        :param move: Move
        :return: None
        """

        # If a king moves, the player can no longer castle
        if move.pieceMoved == "wK":
            self.currentCastlingRights.wks = False
            self.currentCastlingRights.wqs = False
        elif move.pieceMoved == "bK":
            self.currentCastlingRights.bks = False
            self.currentCastlingRights.bqs = False

        # If a rook moves, the player can no longer castle on that side
        elif move.pieceMoved == "wR":
            if move.startRow == 7:
                if move.startCol == 0:  # left rook
                    self.currentCastlingRights.wqs = False
                elif move.startCol == 7:  # right rook
                    self.currentCastlingRights.wks = False
        elif move.pieceMoved == "bR":
            if move.startRow == 0:
                if move.startCol == 0:  # left rook
                    self.currentCastlingRights.bqs = False
                elif move.startCol == 7:  # right rook
                    self.currentCastlingRights.bks = False

    def check_for_pins_and_checks(
        self,
    ) -> tuple[bool, list[tuple[int, int]], list[tuple[int, int, int, int]]]:
        """
        Determine if the current player is in check or pinned
        :return: In check, pins, checks
        """

        # Keep track of pins and checks
        pins = []
        checks = []
        in_check = False

        # Determine which king to check
        if self.whiteToMove:
            enemy_color = "b"
            ally_colour = "w"
            start_row = self.whiteKingLocation[0]
            start_col = self.whiteKingLocation[1]
        else:
            enemy_color = "w"
            ally_colour = "b"
            start_row = self.blackKingLocation[0]
            start_col = self.blackKingLocation[1]

        # Check outwards from king for pins and checks, keep track of pins
        directions = (
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )

        # Check for pins first
        for j in range(len(directions)):
            d = directions[j]
            possible_pin: tuple = ()

            # Check for pins and checks 1 square at a time
            for i in range(1, 8):
                end_row = start_row + d[0] * i
                end_col = start_col + d[1] * i

                # Check if the square is on the board
                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    end_piece = self.board[end_row][end_col]

                    # If there is a piece on the square
                    if end_piece[0] == ally_colour and end_piece[1] != "K":
                        # If this is the first allied piece, it could be pinned
                        if possible_pin == ():
                            possible_pin = (end_row, end_col, d[0], d[1])

                        # If there is a second allied piece, it cannot be a pin
                        else:
                            break

                    # If there is an enemy piece on the square
                    elif end_piece[0] == enemy_color:
                        type = end_piece[1]

                        # Check for 5 possibilities:
                        # 1. Orthogonally away from king and piece is a rook
                        # 2. Diagonally away from king and piece is a bishop
                        # 3. 1 square away diagonally from king and piece is a pawn
                        # 4. Any direction and piece is a queen
                        # 5. Any direction 1 square away and piece is a king

                        if (
                            (0 <= j <= 3 and type == "R")
                            or (4 <= j <= 7 and type == "B")
                            or (
                                i == 1
                                and type == "P"
                                and (
                                    (enemy_color == "w" and 6 <= j <= 7)
                                    or (enemy_color == "b" and 4 <= j <= 5)
                                )
                            )
                            or (type == "Q")
                            or (i == 1 and type == "K")
                        ):
                            # If it is not a pin, it is a check
                            if possible_pin == ():
                                in_check = True
                                checks.append((end_row, end_col, d[0], d[1]))
                                break
                            # Add the pin to the list
                            else:
                                pins.append(possible_pin)
                                break
                        # Enemy piece not applying check
                        else:
                            break
                # Off board
                else:
                    break

        # Check for knight checks
        knight_moves = (
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        )

        for m in knight_moves:
            end_row = start_row + m[0]
            end_col = start_col + m[1]

            # Check if the square is on the board
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]

                # If the enemy knight is attacking the king
                if end_piece[0] == enemy_color and end_piece[1] == "N":
                    in_check = True
                    checks.append((end_row, end_col, m[0], m[1]))

        return in_check, pins, checks

    def get_valid_moves(self) -> list[Move]:  # implement a faster version later
        """
        Gets a list of valid moves for the current board and player
        :return: list[Move]
        """

        # Initialise the list of valid moves
        moves: list = []

        # Check if the player is in check or pinned
        is_in_check, pins, checks = self.check_for_pins_and_checks()

        # Assign the return values to the corresponding variables
        self.is_in_check = is_in_check
        self.pins = pins
        self.checks = checks

        # Store the king's location depending on whose turn it is
        if self.whiteToMove:
            king_row = self.whiteKingLocation[0]
            king_col = self.whiteKingLocation[1]
        else:
            king_row = self.blackKingLocation[0]
            king_col = self.blackKingLocation[1]

        # If the player is in check, limit the moves to only those that get the player out of check
        if self.is_in_check is True:
            # If there is only one check, block the check or move the king
            if len(self.checks) == 1:
                moves = self.get_all_possible_moves()
                check = self.checks[0]
                check_row = check[0]
                check_col = check[1]
                piece_checking = self.board[check_row][check_col]
                valid_squares = []

                # If the piece checking the king is a knight, the king must move
                if piece_checking[1] == "N":
                    valid_squares = [(check_row, check_col)]

                # Otherwise, the piece checking the king can be blocked or captured
                else:
                    for i in range(1, 8):
                        valid_square = (
                            king_row + check[2] * i,
                            king_col + check[3] * i,
                        )
                        valid_squares.append(valid_square)
                        if (
                            valid_square[0] == check_row
                            and valid_square[1] == check_col
                        ):
                            break

                # Get rid of any moves that don't block the check or move the king
                for i in range(len(moves) - 1, -1, -1):
                    if moves[i].pieceMoved[1] != "K":
                        if (
                            moves[i].endRow,
                            moves[i].endCol,
                        ) not in valid_squares:
                            moves.remove(moves[i])

            # If there are multiple checks, the king must move
            else:
                self.get_king_moves(king_row, king_col, moves)

        # If the player is not in check, all moves are valid
        else:
            moves = self.get_all_possible_moves()

        # Add castling moves
        self.get_castle_moves(king_row, king_col, moves)

        # Check if the player is in checkmate or stalemate
        if len(moves) == 0:
            if self.in_check is True:
                self.checkmate = True
            else:
                self.stalemate = True

        return moves

    def in_check(self) -> bool:
        """
        Determine if the current player is in check
        :return: bool
        """

        # Determine which king is in check
        if self.whiteToMove:
            return self.square_under_attack(
                self.whiteKingLocation[0], self.whiteKingLocation[1]
            )
        else:
            return self.square_under_attack(
                self.blackKingLocation[0], self.blackKingLocation[1]
            )

    def square_under_attack(self, row: int, col: int) -> bool:
        """
        Determine if the enemy can attack the square at row, col
        :param row: int
        :param col: int
        :return: bool
        """

        # Switch to opponent's turn
        self.whiteToMove = not self.whiteToMove

        # Get the opponent's moves
        opponent_moves = self.get_all_possible_moves()

        # Switch back to original turn
        self.whiteToMove = not self.whiteToMove

        # Check if the opponent can attack the square
        for move in opponent_moves:
            if move.endRow == row and move.endCol == col:
                return True

        # If the opponent cannot attack the square
        return False

    def get_all_possible_moves(self) -> list[Move]:
        """
        Gets a list of all possible moves
        :return: list[Move]
        """

        # Initialise the list of moves
        moves: list = []

        # Iterate through each square on the board
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                turn = self.board[row][col][0]

                # If it is the player's turn, get the moves for that piece
                if (turn == "w" and self.whiteToMove) or (
                    turn == "b" and not self.whiteToMove
                ):
                    # Get the piece at the square
                    piece = self.board[row][col][1]

                    # Use the appropriate function to get the moves for that piece
                    self.moveFunctions[piece](row, col, moves)

        return moves

    def get_pawn_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the pawn moves for the pawn located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # Initialise variables
        piece_pinned = False
        pin_direction: tuple = ()

        pawnPromotion = False

        # Check if the pawn is pinned
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == row and self.pins[i][1] == col:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                if self.board[row][col][1] != "Q":
                    self.pins.remove(self.pins[i])
                break

        # Determine the direction the pawn should move
        if self.whiteToMove:
            move_amount = -1
            start_row = 6
            back_row = 0
            enemy_color = "b"
        else:
            move_amount = 1
            start_row = 1
            back_row = 7
            enemy_color = "w"

        # Check for valid moves
        if self.board[row + move_amount][col] == "--":
            # Check if the pawn is pinned
            if not piece_pinned or pin_direction == (move_amount, 0):
                # Check for pawn promotion
                if row + move_amount == back_row:
                    pawnPromotion = True

                # Allow the pawn to move 1 square
                moves.append(
                    Move(
                        (row, col),
                        (row + move_amount, col),
                        self.board,
                        pawnPromotion,
                    )
                )

                # Check if the pawn is on its starting square
                if row == start_row and self.board[row + 2 * move_amount][col] == "--":
                    # Allow the pawn to move 2 squares
                    moves.append(
                        Move(
                            (row, col),
                            (row + 2 * move_amount, col),
                            self.board,
                        )
                    )

        # Check for captures to the left
        if col - 1 >= 0:
            # Check if the pawn is pinned
            if not piece_pinned or pin_direction == (move_amount, -1):
                # Check if there is an enemy piece to the left
                if self.board[row + move_amount][col - 1][0] == enemy_color:
                    # Check for pawn promotion
                    if row + move_amount == back_row:
                        pawnPromotion = True

                    # Allow the pawn to capture to the left
                    moves.append(
                        Move(
                            (row, col),
                            (row + move_amount, col - 1),
                            self.board,
                            pawnPromotion,
                        )
                    )

        # Check for captures to the right
        if col + 1 <= 7:
            # Check if the pawn is pinned
            if not piece_pinned or pin_direction == (move_amount, 1):
                # Check if there is an enemy piece to the right
                if self.board[row + move_amount][col + 1][0] == enemy_color:
                    # Check for pawn promotion
                    if row + move_amount == back_row:
                        pawnPromotion = True

                    # Allow the pawn to capture to the right
                    moves.append(
                        Move(
                            (row, col),
                            (row + move_amount, col + 1),
                            self.board,
                            pawnPromotion,
                        )
                    )

        # Check for en passant
        if self.enPassantPossible:
            ep_row, ep_col = self.enPassantPossible

            # Check if the pawn is on the same row as the en passant pawn
            if self.whiteToMove:
                if row == 3 and (ep_col == col - 1 or ep_col == col + 1):
                    moves.append(
                        Move((row, col), (row - 1, ep_col), self.board, enPassant=True)
                    )
            else:  # Black to move
                if row == 4 and (ep_col == col - 1 or ep_col == col + 1):
                    moves.append(
                        Move((row, col), (row + 1, ep_col), self.board, enPassant=True)
                    )

    def get_rook_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the rook moves for the rook located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # Initialise variables
        piece_pinned = False
        pin_direction: tuple = ()

        # Check if the rook is pinned
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == row and self.pins[i][1] == col:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                if self.board[row][col][1] != "Q":
                    self.pins.remove(self.pins[i])
                break

        # Check in all orthogonal directions
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1))

        # Check for valid moves
        enemy_color = "b" if self.whiteToMove else "w"

        for d in directions:
            for i in range(1, 8):
                end_row = row + d[0] * i
                end_col = col + d[1] * i

                # Check if the square is on the board
                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    # Check if the rook is pinned
                    if (
                        not piece_pinned
                        or pin_direction == d
                        or pin_direction
                        == (
                            -d[0],
                            -d[1],
                        )
                    ):
                        # Check if the square is empty
                        end_piece = self.board[end_row][end_col]

                        if end_piece == "--":
                            moves.append(
                                Move((row, col), (end_row, end_col), self.board)
                            )

                        # Check if the square contains an enemy piece
                        elif end_piece[0] == enemy_color:
                            moves.append(
                                Move((row, col), (end_row, end_col), self.board)
                            )
                            break

                        # Check if the square contains a friendly piece
                        else:
                            break

                # Off board
                else:
                    break

    def get_knight_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the knight moves for the knight located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # Initialise variables
        piece_pinned = False

        # Check if the knight is pinned
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == row and self.pins[i][1] == col:
                piece_pinned = True
                self.pins.remove(self.pins[i])
                break

        # Check for valid moves
        knight_moves = (
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        )

        ally_colour = "w" if self.whiteToMove else "b"

        for m in knight_moves:
            end_row = row + m[0]
            end_col = col + m[1]  #
            # Check if the square is on the board
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                # Check if the knight is pinned
                if not piece_pinned:
                    end_piece = self.board[end_row][end_col]

                    # Check if the square is empty
                    if end_piece[0] != ally_colour:
                        moves.append(Move((row, col), (end_row, end_col), self.board))

    def get_bishop_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the bishop moves for the bishop located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # Initialise variables
        piece_pinned = False
        pin_direction: tuple = ()

        # Check if the bishop is pinned
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == row and self.pins[i][1] == col:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                if self.board[row][col][1] != "Q":
                    self.pins.remove(self.pins[i])
                break

        # Check in all diagonal directions
        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))

        # Check for valid moves
        enemy_color = "b" if self.whiteToMove else "w"

        for d in directions:
            for i in range(1, 8):
                end_row = row + d[0] * i
                end_col = col + d[1] * i

                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    # Check if the bishop is pinned
                    if (
                        not piece_pinned
                        or pin_direction == d
                        or pin_direction
                        == (
                            -d[0],
                            -d[1],
                        )
                    ):
                        # Check if the square is empty
                        end_piece = self.board[end_row][end_col]

                        if end_piece == "--":
                            moves.append(
                                Move((row, col), (end_row, end_col), self.board)
                            )

                        # Check if the square contains an enemy piece
                        elif end_piece[0] == enemy_color:
                            moves.append(
                                Move((row, col), (end_row, end_col), self.board)
                            )
                            break

                        # Check if the square contains a friendly piece
                        else:
                            break

                # Off board
                else:
                    break

    def get_queen_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the queen moves for the queen located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # The queen moves are a combination of the rook and bishop moves
        self.get_rook_moves(row, col, moves)
        self.get_bishop_moves(row, col, moves)

    def get_king_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the king moves for the king located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # Check in all directions
        row_moves = (-1, -1, -1, 0, 0, 1, 1, 1)
        col_moves = (-1, 0, 1, -1, 1, -1, 0, 1)

        # Check for valid moves
        ally_colour = "w" if self.whiteToMove else "b"

        for i in range(8):
            end_row = row + row_moves[i]
            end_col = col + col_moves[i]

            # Check if the square is on the board
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]

                # Check if the square is empty or contains an enemy piece
                if end_piece[0] != ally_colour:
                    # Temporarily move the king to the end square
                    if ally_colour == "w":
                        self.whiteKingLocation = (end_row, end_col)
                    else:
                        self.blackKingLocation = (end_row, end_col)

                    # Check if the king is in check
                    in_check, pins, checks = self.check_for_pins_and_checks()

                    # If the king is not in check, add the move to the list
                    if not in_check:
                        moves.append(Move((row, col), (end_row, end_col), self.board))

                    # If the king is in check, place the king back on its original square
                    if ally_colour == "w":
                        self.whiteKingLocation = (row, col)
                    else:
                        self.blackKingLocation = (row, col)

    def get_castle_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the castle moves for the king located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # Check if the king is in check
        if self.square_under_attack(row, col):
            return

        # Check if the squares between the king and rook are empty
        if (self.whiteToMove and self.currentCastlingRights.wks) or (
            not self.whiteToMove and self.currentCastlingRights.bks
        ):
            # Check if the squares between the king and right rook are empty
            self.get_kingside_castle_moves(row, col, moves)
        if (self.whiteToMove and self.currentCastlingRights.wqs) or (
            not self.whiteToMove and self.currentCastlingRights.bqs
        ):
            # Check if the squares between the king and left rook are empty
            self.get_queenside_castle_moves(row, col, moves)

    def get_kingside_castle_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the kingside castle moves for the king located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # Check if the squares between the king and right rook are empty
        if (
            self.board[row][col + 1] == "--"
            and self.board[row][col + 2] == "--"
            and not self.square_under_attack(row, col + 1)
            and not self.square_under_attack(row, col + 2)
        ):
            # Add the castle move to the list
            moves.append(Move((row, col), (row, col + 2), self.board, castle=True))

    def get_queenside_castle_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        Gets all the queenside castle moves for the king located at row, col and adds these moves to the list
        :param row: int
        :param col: int
        :param moves: list[Move]
        :return: None
        """

        # Check if the squares between the king and left rook are empty
        if (
            self.board[row][col - 1] == "--"
            and self.board[row][col - 2] == "--"
            and self.board[row][col - 3] == "--"
            and not self.square_under_attack(row, col - 1)
            and not self.square_under_attack(row, col - 2)
        ):
            # Add the castle move to the list
            moves.append(Move((row, col), (row, col - 2), self.board, castle=True))


class CastleRights:
    """
    Keep track of whether or not castling is possible
    """

    def __init__(self, wks: bool, bks: bool, wqs: bool, bqs: bool) -> None:
        """
        Initialise the castling rights
        :param wks: bool
        :param bks: bool
        :param wqs: bool
        :param bqs: bool
        :return: None
        """

        # Keep track of whether or not castling is possible
        self.wks = wks  # white king side
        self.bks = bks  # black king side
        self.wqs = wqs  # white queen side
        self.bqs = bqs  # black queen side


class Move:
    """
    Defines a move, and the corresponding changes to the board
    """

    def __init__(
        self,
        startSq: tuple,
        endSq: tuple,
        board: list[list[str]],
        enPassant: bool = False,
        castle: bool = False,
    ) -> None:
        """
        Initialise the move
        :param startSq: (row, col)
        :param endSq: (row, col)
        :param board: 2d list of pieces in the form "bR", "wQ", etc.
        :return: None
        """

        # Keep track of the start and end squares
        self.startRow = startSq[0]
        self.startCol = startSq[1]
        self.endRow = endSq[0]
        self.endCol = endSq[1]

        # Keep track of the piece moved and captured
        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]

        # Keep track of whether or not the move is an en passant move
        self.isEnPassantMove = enPassant

        # Keep track of whether or not the move is a castle move
        self.castle = castle

        # Keep track of whether or not the move is a pawn promotion
        self.isPawnPromotion = (self.pieceMoved == "wP" and self.endRow == 0) or (
            self.pieceMoved == "bP" and self.endRow == 7
        )

        # Give each move a unique ID
        self.moveID = (
            self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
        )

    def __eq__(self, other: Any) -> bool:
        """
        Overriding the equals method
        :param other: Any
        :return: bool
        """

        # Check if the other object is a Move object
        if isinstance(other, Move):
            return self.moveID == other.moveID

        # If the other object is not a Move object
        return False

    def get_chess_notation(self) -> str:  # TODO: add more complex notation
        """
        Converts the move into chess notation (e.g., e2e4, e7e5, etc.)
        :return: str
        """

        # Convert the column number to a letter
        cols_to_letters = {
            0: "a",
            1: "b",
            2: "c",
            3: "d",
            4: "e",
            5: "f",
            6: "g",
            7: "h",
        }

        # Convert the start and end squares to chess notation
        start_square = cols_to_letters[self.startCol] + str(8 - self.startRow)
        end_square = cols_to_letters[self.endCol] + str(8 - self.endRow)

        # Return the chess notation
        return start_square + end_square
