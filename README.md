# CSE3000 Research Project - Bram Pronk - Group 21 | Mixture-of-Experts
### Assessing How Variational Auto-Encoders Can Combine Information From Multiple Data Sources in Cancer Cells

[![Python Version](https://img.shields.io/static/v1.svg?label=minimal_python_version&message=3.8.8&color=blue)](https://www.python.org/downloads)

## Description of the Mixture of Experts in this research
In the Mixture-of-Experts (MoE) MVAE model as proposed by Shi et al., the joint variational posterior is given as a combination of unimodal posteriors, using a MoE approach. Further details on the mathematics are presented in the original paper.

This repository was originally written for dualomics analysis, transciptome and surface protein data with chromatin accessibility data. This was expanded to the three different modalities in this paper. Furthermore, feature vectors in the original paper are modelled by a negative binomial distribution for transciptome and surface protein data and with a zero-inflated negative binomial for chromatin accessibility data. In this work, each modality is modelled by a normal distribution, akin to the Pytorch-VAE Vanilla-VAE model. With a normal distribution, the encoder can be trained to return the mean and the covariance matrix that describe the posterior distributions.

Forked from https://github.com/kodaim1115/scMM

## Getting Started
<!---

This section should contain installation, testing, and running instructions for people who want to get started with the project. 

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.

--->
It is highly recommended to run this branch in a new Anaconda environment.
The configuration for the Conda environment used in this research is found ```moe_conda_environment.yml```.
This file can be used to create a setup and ready to go environment for this branch.
 
## Branch Structure
TBD

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
