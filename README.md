# Neuroevolution Pool AI

A Python-based machine learning project that trains an artificial neural network to play 8-ball pool. Instead of using traditional reinforcement learning (like Q-learning), this project uses a Genetic Algorithm (Neuroevolution) to evolve a population of agents over successive generations.

The physics are handled by a pymunk engine, allowing the simulation to evaluate thousands of shots in the background in seconds, while pygame is used to render a visual replay of the best shot from each generation.

## How It Works

# The Brain (Neural Network)
Each agent is a Multi-Layer Perceptron (MLP) built from scratch using numpy

Inputs (32): The normalized X and Y coordinates of all 16 balls on the table. Potted balls are moved to "ghost" coordinates off-screen
Hidden Layer (128): Processes the spatial layout
Outputs (2): Squashed via tanh to output the strike angle (0-360 degrees) and strike force (5-100%)

# The Physics (Pymunk)
To train fast, the game needs to run without drawing graphics.

Physics_engine.py handles the rigid-body physics, friction, and cushion bounces.

It runs a fixed-step simulation (1/60.0) continuously until all balls stop moving.

Real-time distance checks handle pocketing to prevent bounding-box collision bugs (tunneling) against the cushions.

# The Evolution (Genetic Algorithm)
Population: 100 random neural networks are spawned in Generation 1.

Fitness Score: Agents are heavily penalized for scratching (-1000) or potting the 8-ball early (-5000), and rewarded for potting their assigned balls (+1000).

Selection & Mutation: The top 10% of agents pass directly to the next generation. The remaining 90% are cloned from the top with a 5% Gaussian mutation rate applied to their weights.

## Installation
You will need Python 3.x and a few dependencies.

# Clone the repository
git clone [your-repo-link]
cd [your-repo-folder]

# Install required libraries
pip install numpy pygame pymunk
Usage
Run the main script to start the training loop:

Bash
python First_trial.py
What to expect:
When you run the script, the terminal will instantly output the background calculations for the first generation. Once the 100 networks have been evaluated, a pygame window will pop up showing a real-time replay of the Generation Champion's shot (best score and not most balls potted). The window will then instantly reset to train the next generation.

Configuration & Tweaks
Fixed vs. Dynamic Tables
By default, the training generates a completely new random table layout for every generation. This forces the AI to learn generalized pool physics rather than memorizing a single shot.

## If you want to watch the AI perfect a single trick shot on a static layout, you can lock the table generation seed in the main() function:

# In main()
generation_seed = 42

for nn in ga.population:
    random.seed(generation_seed) # Forces the exact same table layout every time
    env = BilliardEnvironment()
    # ...
