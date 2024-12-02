#  Heat Equation Solver Comparisons
This repository shows how three methods compare in solving the heat equation, an important partial differential equation in physics and engineering. Researchers and practitioners seek understandings into each method's strengths, weaknesses and applicability for solving the heat equation across multiple scenarios. We nondimensionalize the equations for better generality. This approach allows for greater scalability and enables more strong training. We examine several methods:
1. Analytical Solution: The exact solution derived mathematically.
2. Finite Difference Method (FDM): A numerical approach based on discretization.
3. Physics-Informed Neural Networks (PINNs): A machine learning framework that incorporates physical laws into the training process of a neural network.

## Peak
Side-by-side plots of the solutions are provided to illustrate the accuracy and characteristics of each method:

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/JanAlexanderZak/heat_equation_solutions/blob/main/src/plots/analytical_solution.png" alt="Image 1" width="32.9%" />
  <img src="https://github.com/JanAlexanderZak/heat_equation_solutions/blob/main/src/plots/FDM_solution.png" alt="Image 2" width="32.9%" />
  <img src="https://github.com/JanAlexanderZak/heat_equation_solutions/blob/main/src/plots/pinn_solution.png" alt="Image 3" width="32.9%" />
</div>