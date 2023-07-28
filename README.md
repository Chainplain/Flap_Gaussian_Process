# Flap_Gaussian_Process
Using the Gaussian process regression to learn the flapping wing dynamics.

The dynamics of flapping wing flight is notoriously challenging to model due to its inherent complexity. 
This paper presents the dynamics learning method based on Gaussian Process Regression (GPR), and the further implementation 
in trajectory tracking controller development. Specifically, through actively querying selected instances for labels, 
the algorithm can learn more efficiently with fewer labeled examples. 
Meanwhile, in order to improve the efficiency of the active learning process, 
a batch-wise data selection approach is employed. By taking advantage of the learned Gaussian Process Regression model, 
an optimization-based control strategy is proposed to refine actuator inputs using acquired knowledge. 
The superiority of the proposed control strategy are validated by high-fidelity numerical simulations.

## Installation
Please refer to https://github.com/Chainplain/FlappingwingSimu#installation for installation.
