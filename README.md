# CSE3000 Research Project - Bram Pronk - Group 21 | Product of Experts
### Assessing How Variational Auto-Encoders Can Combine Information From Multiple Data Sources in Cancer Cells

[![Python Version](https://img.shields.io/static/v1.svg?label=minimal_python_version&message=3.8.8&color=blue)](https://www.python.org/downloads)

## Description of the Product of Experts in this research
See Research Paper Section 3.2.
For the Product-of-Experts (PoE) MVAE model, proposed by Wu and Goodman, the joint posterior is a product of individual posteriors. This approach is originally introduced by Hinton. The idea is that each unimodal VAE in the model is considered an expert. In PoE, "each expert holds the power of veto—in the sense that the joint distribution will have low density for a given set of observations if just one of the marginal posteriors has low density", as explained by a contrasting section in the original MoE paper. This implies that experts with high precision are weighing more heavily in determining the posterior distribution than lower density experts, since we take a product. The general formula from the paper is given in the Research Paper, and also in the original paper. A more in-depth look is also given in the original paper's supplement.
Forked from https://github.com/kodaim1115/scMM

## Getting Started
<!---

This section should contain installation, testing, and running instructions for people who want to get started with the project. 

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.

--->
It is highly recommended to run this branch in a new Anaconda environment.
The configuration for the Conda environment used in this research is found ```poe_conda_environment.yml```.
This file can be used to create a setup and ready to go environment for this branch.

## Branch Structure
```
|---vision/
    |---results/
    |---datasets.py
    |---model.py
    |---pca.py
    |---predict.py
    |---train.py
    |---umapz.py
|---README.md
|---.gitignore
```


## Authors
This is the personal repository of

    - Bram Pronk            (4613066) i.b.pronk@student.tudelft.nl

My colleagues in Research Group 21:

     - Armin Korkic         (4713052) a.korkic@student.tudelft.nl
     - Boris van Groeningen (4719875) b.vangroeningen@student.tudelft.nl
     - Ivo Kroskinski       (4684958) i.s.kroskinski@student.tudelft.nl
     - Raymond d'Anjou      (4688619) r.danjou@student.tudelft.nl

Research Group 21 is guided by:
    
    - Marcel Reinders (Responsible Professor)
    - Stavros Makrodimitris, Tamim Abdelaal, Mohammed Charrout, Mostafa elTager (Supervisors)
