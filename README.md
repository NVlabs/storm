# STORM
**Stochastic Tensor Optimization for Robot Motion** - *A GPU Robot Motion Toolkit*

[[Install Instructions](install_instructions.md)]

This package contains code for reactive robot motion leveraging parallel compute on the GPU. The implemented control framework leverages MPPI to optimize over sampled actions and their costs. The costs are computed by rolling out the forward model from the current state with the sampled actions.

<p align="center">
  <img width="500" src="docs/images/coll_demo.gif">
</p>

Most files are documented with sphinx. Once you clone this repo, go into docs folder and run `sh generate_docs.sh` to generate documentation.


## Citation
This code was developed as part of a research publication. The bib for the publication will be added soon.

## Contributing to this code
Refer to CLA before making contributions.
