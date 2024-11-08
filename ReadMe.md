# AI-Powered Car Simulation

This project is a car simulation environment that uses a Neuroevolution of Augmenting Topologies (NEAT) algorithm to train cars to navigate a race track. The cars are equipped with sensors that provide information about their surroundings, and the NEAT algorithm evolves their neural networks to learn the optimal driving behavior. This code is heavily inspired from the AI-Car-Simulation repo by NeuralNine.

## Features

- Pygame-based simulation environment with a race track and cars
- NEAT algorithm for training the cars' neural networks
- Customizable NEAT configuration options
- Visualization of the cars' sensor inputs and driving behavior
- Performance tracking and fitness evaluation during training

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ai-car-simulation.git
   ```
2. Change to the project directory:
   ```
   cd ai-car-sim
   ```
3. Install the required packages:
   ```
   pip install neat-python pygame
   ```

## Usage

1. Ensure you have the `config.txt` file in the project directory.
2. Run the main script:
   ```
   python main.py
   ```
3. The simulation will start, and you'll see the cars attempting to navigate the track.
4. The NEAT algorithm will evolve the car's neural networks over multiple generations, with the goal of improving their driving performance.
5. You can monitor the progress of the training by observing the cars' movements and the statistics reported in the console.

## Configuration

The simulation's behavior is controlled by the parameters defined in the `config.txt` file. You can modify these parameters to experiment with different NEAT settings and observe their impact on the cars' performance.

Some key configuration options include:
- `pop_size`: The number of cars (genomes) in the population
- `num_hidden`: The number of hidden nodes in the neural networks
- `compatibility_threshold`: The threshold for determining whether two genomes belong to the same species
- `max_stagnation`: The maximum number of generations a species can go without improving before it is considered stagnant

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Implement your changes
4. Test your changes thoroughly
5. Submit a pull request with a description of your changes

