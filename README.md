# Bayesian Data Analysis

This is the GitHub repo for the Thursday seminar materials for the course [Bayesian Data Analysis](https://www.ncl.ac.uk/postgraduate/degrees/module/?code=MAS8405). 

In total there are three Thursday seminars, each covering and extending content covered in the corresponding week, with a particular focus on practical application using ``R``. For each seminar, there is a corresponding folder in this repo with all the materials used in that seminar. This includes a PDF of the slides used, the Jupyter notebook files and corresponding ``R`` scripts. Please feel free to borrow and rework any of the featured code in future work.

In the seminars, I will be presenting using a Jupyter notebook using an ``R`` kernel with Visual Studio Code. Reading and operating a Jupyter notebook offline will require installation of ``Python`` and executing cells with an ``R`` kernel requires more setup. However, you should be able to at least read the notebooks via GitHub on your browser. Just in case, I have also provided corresponding ``R`` files with the raw ``R`` code included. You should be able to execute these ``R`` files on your computer offline, using an installation of ``R`` - perhaps with an IDE like [RStudio](https://www.rstudio.com/products/rstudio/). For ease of use, I would recommend using RStudio with R notebooks over something like Visual Studio Code with Jupyter notebooks. 

Email: matthew.fisher@newcastle.ac.uk

### Links and Further readings:

The following is a list of useful links:

- [Advanced Statistical Computing](https://bookdown.org/rdpeng/advstatcomp/): An online book written with ``R`` focussing on statistical computation. The relevant material here is the section of Markov Chain Monte Carlo. Other sections demonstrate other computational techniques.
- [Statistical Computing for Biologists](https://bio723-class.github.io/Bio723-book/index.html): An online ``R`` course intended for graduate level biologists. The course is really just statistical computing. The relevant material are the introduction to ``ggplot`` and ``dplyr``.
- [JAGS User Manual](https://people.stat.sc.edu/hansont/stat740/jags_user_manual.pdf): If you really want to know all the inner workings of JAGS, the user manual covers pretty much everything. 
- [Resources for "A Students Guide to Bayesian Statistics"](https://study.sagepub.com/lambert): Website of the book "A Student's Guide to Bayesian Statistics". Features videos corresponding to each chapter as well as solutions to problem sets. Unfortunately, all the videos are not complete.
- [R for Data Science](https://r4ds.had.co.nz): Website of the book "R for Data Science". Excellent resource for basics of ``R``. 

## Setup Guide

This guide covers the bare essential requirements for running the ``JAGS``. These requirements should already be setup on university machines. All we require are installations of ``R``, RStudio, JAGS and some ``R`` packages:
1. [Install](https://www.stats.bris.ac.uk/R/) the latest version of  ``R`` for your operating system.
2. [Install](https://www.rstudio.com/products/rstudio/) the free version of RStudio: RStudio Desktop, Open Source Edition.
3. [Install](https://cran.r-project.org/) the latest version of JAGS for your operating system. JAGS should automatically link to ``R`` and you should be ready to go.
4. In an ``R`` terminal or in the terminal of RStudio, install ``rjags`` and ``coda`` with the commands ``install.packages(rjags)`` and ``install.packages(coda)``. To load the library ``rjags``, you may have to restart your ``R`` instance.

Note that ``R`` and ``JAGS`` may also be available on package managers. For instance, if you are using MacOS, they are both available on ``homebrew``.

If you are using Apple Silicon, it is slightly more difficult to install and use ``JAGS``. There are various [options](https://sourceforge.net/p/mcmc-jags/discussion/610037/thread/07e08a3605/), although I recommend the following: 

1. Install the arm64 version of ``R``. This is available [here](https://cran.r-project.org/bin/macosx/).
2. Ensure you have ``homebrew`` installed.
3. In a terminal, install using ``homebrew``, the packages [``JAGS``](https://formulae.brew.sh/formula/jags) and [``pkg-config``](https://formulae.brew.sh/formula/pkg-config) using the commands ``brew install pkg-config`` and ``brew install jags``.
4. In ``R``, install ``rjags`` using ``install.packages("rjags", type = "source")``. If you don't use ``type = "source"``, ``rjags`` will not be loadable. To fix this, just uninstall ``rjags`` using ``remove.packages("rjags")`` and reinstalling. 
