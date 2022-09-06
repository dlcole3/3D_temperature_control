# 3-D Temperature Control

This repository contains code for the 3-D temperature control of a cube. This is a linear model predicitve control (MPC) problem constructed in `DynamicNLPModels.jl` and solved with `Ipopt` and `MadNLP.jl`. The results are presented in the conference proceedings, "Interior Point Methods on GPU/SIMD Architecture for Linear MPC".

The file `PDE_boundary_3d_heating.jl` contains the code for building the temperature control model. This model is the discretization of the problem 

$$
\begin{aligned}
    \min_{x, u} &\; \frac{1}{2} \int_0^{T_f} \int_{w \in \Omega} \left( (x(w, t) - d(w, t))^2 + \sum_{i \in \mathcal{U}} r u_i(t)^2 \right) dw dt\\
    \textrm{s.t.} &\; \rho C_p  \frac{\partial x(w, t)}{dt} = k \nabla^2 x(w, t), \quad w \in \Omega, t \in [0, T_f]\\
    & x^l \le x(w, t) \le x^u \quad w \in \Omega, t \in [0, T]\\
    & u^l \le u_i(t) \le u^u \quad i \in \mathcal{U}, w \in \Omega, t \in [0, T]\\
    & x(\bar{w}, t) = u_i(t), \quad \bar{w} \in \partial \Omega_i, i \in \mathcal{U}  \\
    & x(w, 0) = \hat{x}, \quad w \in \Omega
\end{aligned}
$$

where $\Omega = [0, \ell_x]^3 \subseteq \mathbb{R}^3$ is the 3 dimensional domain of a cube of length $\ell_x$, $x: \Omega \times [0, T] \rightarrow \mathbb{R}$ is the temperature, $d: \Omega \times [0, T] \rightarrow \mathbb{R}$  is the set point temperature,  $u_i: [0, T] \rightarrow \mathbb{R}, \forall i \in \mathcal{U}$ are the boundary temperatures (inputs), and $\mathcal{U}$ is the set of all cube faces. We also use $\partial \Omega_i = [0, \ell_x]^2 \subseteq \mathbb{R}^2$ as the boundary of the cube at face $i$. The constants $\rho$, $C_p$, and $k$ correspond to physical properties of the cube, $r$ is a weighting factor on the input, and $s^l$, $s^u$, $u^l$, $u^u$ are variable constraints for the temperatures and inputs.

We solve two instances of this problem, the condensed problem (where the states are eliminated; see [Jerez et al., 2012](https://doi.org/10.1016/j.automatica.2012.03.010)) and the initial sparse problem where the states are not eliminated. The sparse problem is solved with `Ipopt` using Ma27 as the linear solver. Scripts for solving the sparse problem are contained in the files `time_pde_heating_ipopt_fixedT.jl` and `time_pde_heating_ipopt_nz4.jl`. Scripts for solving the condensed problem are contained in the files `time_pde_heating_fixedT.jl` and `time_pde_heating_nz4.jl`. The condensed problem can be solved on both CPU and GPU. 

Output data can be found in the directory `output_files`. .jld files contain dictionaries with dataframes containing the outputs of our runs. Solver outputs are also available, and are identified by the number of states (ns) and the time horizion (T). Since we tested a range of nz values while maintaining a fixed T as well as a range of T values while maintaining a fixed nz, identifiers are also included in the solver output names to correspond to the range of nz or the range of T tests. Lastly, there is a file `write_data.jl` that will output the data from the .jld files to a single .csv. 
