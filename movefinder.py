"""The chess AI module."""

# Imports

from engine import Move, GameState


def find_best_move(gamestate: GameState, valid_moves: list, depth=3) -> Move:
    """
    Find the best move using a depth-limited MinMax algorithm.
    :param gamestate: the current game state
    :param valid_moves: list of valid moves
    :param depth: depth of search
    :return: the best move
    """

    # Initialise constants
    PIECE_SCORE = {"K": 100, "Q": 9, "R": 5, "B": 4, "N": 3, "P": 1}
    CHECKMATE_SCORE = 1000
    STALEMATE_SCORE = 0

    # Initialise variables
    turn_multiplier = 1 if gamestate.white_to_move else -1
    max_evaluated_score = float("inf")

    # Find the best move by trying all valid moves and scoring them using MinMax
    for player_move in valid_moves:
        gamestate.make_move(player_move)
        score = minmax(
            gamestate,
            depth - 1,
            not gamestate.white_to_move,
            -float("inf"),
            float("inf"),
            PIECE_SCORE,
            CHECKMATE_SCORE,
            STALEMATE_SCORE,
        )
        gamestate.undo_move()

        # Update the best move
        if score * turn_multiplier < max_evaluated_score:
            max_evaluated_score = score * turn_multiplier
            best_move = player_move

    # Return the best move
    return best_move


def minmax(
    gamestate,
    depth,
    maximizing_player,
    alpha,
    beta,
    PIECE_SCORE,
    CHECKMATE_SCORE,
    STALEMATE_SCORE,
):
    """
    MinMax algorithm with Alpha-Beta pruning.
    :param gamestate: the current game state
    :param depth: depth of search
    :param maximizing_player: whether the player is maximizing or minimizing
    :param alpha: alpha value
    :param beta: beta value
    :param PIECE_SCORE: dictionary of piece scores
    :param CHECKMATE_SCORE: score of checkmate
    :param STALEMATE_SCORE: score of stalemate
    """

    # Base case
    if depth == 0 or gamestate.checkmate or gamestate.stalemate:
        return score_board(gamestate, PIECE_SCORE, CHECKMATE_SCORE, STALEMATE_SCORE)

    # Recursive case
    if maximizing_player:
        # Maximizing player
        max_score = -float("inf")

        # Try all valid moves
        for move in gamestate.get_valid_moves():
            # Make the move
            gamestate.make_move(move)

            # Recursively call minmax
            current_score = minmax(
                gamestate,
                depth - 1,
                False,
                alpha,
                beta,
                PIECE_SCORE,
                CHECKMATE_SCORE,
                STALEMATE_SCORE,
            )

            # Undo the move
            gamestate.undo_move()

            # Update the max score
            max_score = max(max_score, current_score)

            # Update alpha
            alpha = max(alpha, current_score)

            # Prune
            if beta <= alpha:
                break

        # Return the max score
        return max_score
    else:
        # Minimizing player
        min_score = float("inf")

        # Try all valid moves
        for move in gamestate.get_valid_moves():
            # Make the move
            gamestate.make_move(move)

            # Recursively call minmax
            current_score = minmax(
                gamestate,
                depth - 1,
                True,
                alpha,
                beta,
                PIECE_SCORE,
                CHECKMATE_SCORE,
                STALEMATE_SCORE,
            )

            # Undo the move
            gamestate.undo_move()

            # Update the min score
            min_score = min(min_score, current_score)

            # Update beta
            beta = min(beta, current_score)

            # Prune
            if beta <= alpha:
                break

        # Return the min score
        return min_score


def score_board(
    gamestate: GameState, PIECE_SCORE: dict, CHECKMATE_SCORE: int, STALEMATE_SCORE: int
) -> int:  # TODO: add positional scoring
    """
    Score the material on the board and consider checkmate and stalemate.
    :param gamestate: the current game state
    :return: the score
    """

    # Checkmate or stalemate
    if gamestate.checkmate:
        if gamestate.white_to_move:
            return -CHECKMATE_SCORE  # Black wins
        else:
            return CHECKMATE_SCORE  # White wins
    elif gamestate.stalemate:
        return STALEMATE_SCORE

    # Score the material
    score = 0
    for row in gamestate.board:
        for piece in row:
            if piece != "--":
                piece_type = piece[1]
                piece_color = 1 if piece[0] == "w" else -1
                score += piece_color * PIECE_SCORE[piece_type]

    # Return the score
    return score
