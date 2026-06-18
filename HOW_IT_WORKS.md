# How Chess Mini-Me works

A short tour of the program for a reader comfortable with linear algebra and
tensors, and with the basic idea of a neural network as a parameterised
function trained by gradient descent. It is kept deliberately brief.

The program has four parts: a rules engine, a classical search opponent, a
learned "Mini-Me" opponent, and a graphical interface. Only the Mini-Me uses
machine learning; everything else is ordinary algorithms.

## 1. The board as data

A position is an 8x8 grid; each square holds a short code such as `wQ` (white
queen) or `--` (empty). The engine (`engine.py`) stores this grid, generates
the legal moves for the side to move (handling castling, en passant, promotion,
pins and checks), and can make and undo a move. It also detects every way a
game can end: checkmate, stalemate, threefold repetition, the fifty-move rule,
insufficient material and resignation. Nothing here is learned.

## 2. The classical opponent: search

Think of the game as a tree: each node is a position, each edge a move. Define
the value of a position to the side to move recursively as

    v(s) = max over legal moves m of  ( - v(s after m) ),

with the recursion stopped at a fixed depth, where `v` is replaced by a static
score. The single sign flip `-v(...)` encodes "your opponent's best reply is
your worst case"; this is ordinary minimax written in one symmetric form
(negamax). `move_finder.py` implements it with alpha-beta pruning, which skips
branches that cannot change the result and so computes the same answer while
visiting far fewer nodes.

The static score (`evaluate_board`) is a single real number, positive when
White is better. It is the sum over pieces of a material value (a pawn is 100,
a queen 900, and so on) plus a small positional bonus read from a per-piece
table of 64 numbers. That is the whole "understanding" of the classical engine.

## 3. The learned opponent: the Mini-Me

The Mini-Me learns to imitate a particular player. This is the machine-learning
part, and it is plain supervised learning.

### 3.1 Encoding a position and a move as tensors

A position is turned into a binary tensor (`encoding.py`)

    X in {0, 1}^(18 x 8 x 8),

a stack of eighteen 8x8 "planes" (feature maps). Twelve planes are one per
piece type and colour, holding a 1 wherever that piece stands; one plane marks
whose turn it is; four record castling rights; one marks an en passant target.
So "encode" is just a fixed map from a board to a point in R^(18x8x8).

A move is turned into an index in {0, ..., 4095}, by reading its origin square
(0..63) and destination square (0..63) and combining them as
`origin * 64 + destination`. So the move set embeds into a 4096-dimensional
space of one-hot vectors.

### 3.2 The network: a parameterised function

The network (`cloner.py`) is a function

    f_theta : R^(18 x 8 x 8) -> ( R^4096 , R^4 , R ),

with trainable parameters theta. It is a convolutional neural network. Each
layer applies a bank of small 3x3 linear filters that slide across the grid (a
discrete convolution: a local, translation-equivariant linear map), followed by
a pointwise nonlinearity ReLU(x) = max(0, x) and a normalisation step.
Convolutions fit chess because the useful patterns are local and look much the
same wherever they occur on the board.

The layers are grouped into residual blocks, each computing

    x  ->  x + g(x),

so the block only has to learn the deviation g from the identity. Routing the
input straight through via the `+ x` term keeps gradients well-behaved and lets
the network be made deeper without training breaking down.

Three "heads" read the shared features: a policy head producing a vector of
move scores (logits) p in R^4096; a small promotion head in R^4; and a value
head producing a number in (-1, 1) via tanh, a prediction of the eventual
result.

### 3.3 From scores to a move

Only legal moves are allowed, so the illegal entries of p are masked out, and
the rest are passed through a softmax to give a probability distribution:

    P(move = i)  =  m_i * exp(p_i / T)  /  sum_j  m_j * exp(p_j / T),

where m is the 0/1 legality mask and T > 0 is a "temperature". At play time the
Mini-Me samples a move from P, so it varies its choices like a person; T sets
how adventurous it is (small T concentrates on the single most likely move).

### 3.4 Training: behavioural cloning

The training data is a set of pairs (X, i*): a position the player faced and
the move i* they actually played. These come from your own games, or from a
Lichess player's downloaded games (`lichess.py`) replayed to extract their
moves.

We want the network to put high probability on the move the player made, so we
minimise the cross-entropy (equivalently, the negative log-likelihood)

    L(theta)  =  - sum over examples  log P_theta( i* | X )

plus small extra terms: a cross-entropy on the promotion choice, and a
mean-squared error fitting the value head to the game's result.

This is exactly maximum-likelihood fitting of the conditional distribution
p(move | position) that the player embodies. Driving L down pushes the
network's distribution towards the player's own, so it comes to choose the
moves they would, in the positions they would reach. That, made precise, is
what "copying their style" means.

The minimisation is gradient descent on theta (the Adam optimiser, an adaptive
step-size variant), with the gradients computed by backpropagation -- the chain
rule applied layer by layer -- over small random batches of examples
(`training.py`).

### 3.5 Learning over time, and profiles

The dataset is kept on disk and grows: every game you play appends your moves
and the network is refitted, so the fitted distribution tracks you ever more
closely. Each Mini-Me is a "profile" -- a saved theta and dataset. Your own
play trains a profile called `my-style`; importing a Lichess account trains a
separate named profile from that player's games. The mathematics is identical
in both cases; only the data differ.

## 4. Putting it together

The interface (`interface.py`) runs the loop: the engine offers the legal
moves, the chosen opponent (the search of section 2 or the network of section
3) picks one, the move is animated and recorded, and end conditions are
checked. Games can be saved as PGN, and each Mini-Me's parameters and data
persist between sessions.
