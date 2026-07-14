We would like develop theory and code to solve constrained derivative-free optimization problems. 

We want theoretically rigorous code that adds onto the theory/methods we have already developed. 

We would like to consider simulation-based constraints which are "relaxable".
That is, the nonlinear simulation-based constraint c(x) <= 0 still returns
meaningful values c(x) is positive/infeasible

As a starting-point for notation, see Conn, Scheinberg, Vicente textbook on derivative-free optimization. 

As a starting-point for software, see the .py code in this repository.

We are on an experimental branch in git, so please make minimal edits in software to add the simulation-based constraint handling feature.

The main idea for the analysis is to build on top of the Conn Scheinberg and Vicente DFO textbook notation. A copy of the PDF of that text book is in this directory. We would like to evaluate the objective and constraints and build fully linear/quadratic models of both. If the method is progressing just fine, then the models don't have to be accurate. 

But let's start with the theory first. Can we prove that if we are always building fully-linear models (of objectives and constraints) on each iteration, that we can prove convergence to (constrained) stationary point? There is some literature on simulation-based constrained DFO research, but we aren't aware of any methods that prove convergence.

Ask questions about anything above that is unclear. 
