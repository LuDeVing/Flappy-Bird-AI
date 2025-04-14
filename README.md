# Flappy Bird AI 🐦🤖

Welcome to the **Flappy Bird AI** project! This project uses a custom neural network with genetic evolution to teach virtual birds how to navigate through pipes — just like in the classic Flappy Bird game. Enjoy the blend of gameplay and AI learning! 🚀

![image](https://github.com/user-attachments/assets/4349a4f6-806a-4c62-b141-e4f015fc2526)

## Overview 🌟

- **Multiple Birds:** A population of birds (default 100) each controlled by its own AI.
- **Neural Network AI:** Each bird makes decisions (flap or not) using a simple neural network.
- **Genetic Training:** Best performing birds pass their "brains" to the next generation, with random mutations 🌱.
- **Real-Time Training:** The birds improve their strategies on-the-fly based on gameplay data.

## How It Works ⚙️

### Game Mechanics 🎮

- **Bird Movement:**  
  Each bird sprite simulates physics (gravity, velocity, flapping force) and rotates based on its velocity for realism.
  
- **Obstacles (Pipes):**  
  Pipes are generated periodically at a random vertical position. Each set consists of a top and a bottom pipe, creating a gap that the birds must fly through. 🚧
  
- **Scoring:**  
  The score increases when the birds successfully navigate the pipes. The neural network considers the bird's distances from the pipes to make decisions.

### AI and Neural Network 🧠

- **Input Data:**  
  The AI uses four normalized values as input:
  - Distance from the bird to the top of the pipe gap.
  - Distance from the bird to the bottom of the pipe gap.
  - Horizontal distance from the bird to the pipe.
  - Adjusted horizontal distance (considering bird width).

- **Network Structure:**  
  The custom network architecture includes:
  - An input layer with 4 neurons.
  - A hidden layer with 2 neurons using ReLU activation ⚡.
  - An output layer with 2 neurons using softmax for classification (to decide flapping).
  
- **Training Strategy:**  
  The network uses **cross-entropy loss** to adjust weights during training based on the outcomes from gameplay data.
  
- **Evolution:**  
  After each round, the best birds (determined by lifetime performance) have their neural networks copied and slightly mutated to evolve strategies across generations. 🌱➡️🌳

## Project Structure 📂

```
FlappyBirdAI/
│
├── images/                       # Contains game images (background, ground, birds, pipes) 🎨
├── Neural_Network.py             # Neural network and layer implementations 🤖
├── Neural_Functions.py           # Activation and loss functions for the neural network 🧩
├── main.py                       # Main game file that ties everything together 🎮
└── README.md                     # Project documentation (you're reading it now!) 📖
```

## Requirements 🛠️

- **Python:** 3.x (preferably 3.7+)
- **Pygame:** For game graphics and event handling 🎨
- **NumPy:** For mathematical computations and handling neural network operations 🔢

Install the required libraries with:

```
pip install pygame numpy
```

## Installation and Running 🚀

1. **Clone the Repository:**

   ```
   git clone https://github.com/your-username/FlappyBirdAI.git
   cd FlappyBirdAI
   ```

2. **Set Up the Environment:**

   It’s recommended to use a virtual environment:
   
   ```
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install -r requirements.txt 
   ```

3. **Run the Game:**

   Start the game by running:
   
   ```
   python main.py
   ```

## Code Explanation 📝

- **`main.py`:**  
  Manages the game loop, event handling, bird updates, drawing graphics, and pipe generation:
  - A constant frame rate is maintained using a clock ⏰.
  - Each bird makes AI-based decisions to flap.
  - The score increases as birds pass through pipes.
  - Once all birds are eliminated, a new generation is generated and trained.

- **`Neural_Network.py` & `Neural_Functions.py`:**  
  Define the neural network architecture:
  - Layers, activation functions (ReLU, softmax), and the cross-entropy loss function are implemented.
  - Mutations are introduced between generations to promote diversity and learning.

- **AI Decision Process:**  
  The `Bird` class predicts actions every few frames. If the AI decides to flap (or the SPACE key is pressed), the bird’s velocity increases, simulating a flap. The bird’s lifetime performance informs the evolution process.

## License 📄

This project is licensed under the MIT License.
Feel free to use, modify, and distribute it.

---

![image](https://github.com/user-attachments/assets/01c6d9cb-5478-4e88-b80b-f3a42d902717)

Thanks for cheking my project out.

Have fun experimenting with AI and game development in this playful project! Happy coding! 🐦💻🎮
