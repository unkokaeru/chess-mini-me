# Chess Engine for the Mini-Me Project

## Project Overview

This project is a chess game implemented in Python, using the Pygame library for graphical representation. It consists of two primary files: `engine.py` and `main.py`. The `engine.py` contains the logic of the chess game, including the rules, movements of pieces, and game state. The `main.py` file is responsible for the graphical user interface (GUI), handling user inputs, and rendering the game state on the screen.

## Key Features

- **Game Logic**: Implementations of all chess rules, including en passant, castling, pawn promotion, and checks/checkmates.
- **Move Generation**: Algorithm to generate all valid moves for a given game state, including consideration of pins and checks.
- **Graphical User Interface**: A simple and intuitive interface using Pygame for interactive gameplay.
- **Move Animation**: Smooth animations for piece movements, captures, and special moves like castling and pawn promotion.
- **Undo Functionality**: Ability to undo moves and explore different game strategies.
- **Checkmate and Stalemate Detection**: The game can detect and announce checkmate and stalemate conditions.

## Installation

1. **Prerequisites**: Ensure Python 3.x is installed on your system. Additionally, install the Pygame library using the command:

   ```
   pip install pygame
   ```

2. **Download**: Clone the repository or download the source code.

3. **Run the Game**: Navigate to the project directory and run the `main.py` script:

   ```
   python main.py
   ```

## Usage

- The game starts with a standard chess board.
- Click on a piece to see valid moves highlighted.
- Click on a highlighted square to move the piece.
- Use the 'z' key to undo a move.
- Use the 'r' key to reset the board.
- The game automatically detects and announces checkmate or stalemate.

## Customization

- **Images**: Replace the piece images in the 'images' directory to customize the appearance.
- **Board Colors**: Modify the `colors` array in `main.py` to change the board's color scheme.
- **Animation Speed**: Adjust the `frames_per_square` variable in `main.py` to change the animation speed.

## Contributing

Contributions to the project are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## License

This project is open-source and available under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- Thanks to the Python and Pygame communities for the resources and support.
- Special thanks to Eddie Sharick for creating a wonderful tutorial series that this project is heavily inspirede by.

---