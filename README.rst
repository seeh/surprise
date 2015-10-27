=============================
Surprise Calculator
=============================

This package estimates relative entropy and Surprise between two samples,
assuming they are Gaussian. See http://arxiv.org/abs/1402.3593 for more details.

To be able to estimate the significance of the Surprise, the module relies
on the R package CompQuadForm by P. Lafaye de Micheaux. To interface R with 
Python, the module uses rpy2 (http://rpy.sourceforge.net/). If any of R, 
CompQuadForm, or rpy2 are not available, the module will simply not calculate
the p-value of the Surprise.

For installing R, check out https://www.r-project.org/. Once R is installed, 
CompQuadForm can be simply installed by executing the following line within R

> install.packages("CompQuadForm").

The Python interface for R can then be simply installed via pip:

$ pip install rpy2  

For examples on how to use surprise, see the examples folder.