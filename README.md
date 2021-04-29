# STORM
**Stochastic Tensor Optimization for Robot Motion** - *A GPU Robot Motion Toolkit*

[[Install Instructions](install_instructions.md)] [[Paper](https://arxiv.org/abs/2104.13542)] [[Website](https://sites.google.com/view/manipulation-mppi/home)]

This package contains code for reactive robot motion leveraging parallel compute on the GPU. The implemented control framework leverages MPPI to optimize over sampled actions and their costs. The costs are computed by rolling out the forward model from the current state with the sampled actions.

<p align="center">
  <img width="500" src="docs/images/coll_demo.gif">
</p>

Most files are documented with sphinx. Once you clone this repo, go into docs folder and run `sh generate_docs.sh` to generate documentation.


## Citation
If you use this source code, please cite the below article,

```
@article{storm2021,
  title={Fast Joint Space Model-Predictive Control for Reactive Manipulation},
  author={Mohak Bhardwaj and Balakumar Sundaralingam and Arsalan Mousavian and Nathan Ratliff and Dieter Fox and Fabio Ramos and Byron Boots},
  journal={arXiv preprint},
  year={2021}
}
```

## Contributing to this code
Refer to CLA before making contributions.
