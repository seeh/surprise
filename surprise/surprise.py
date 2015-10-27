#! /usr/bin/env python

# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

# System imports
from __future__ import print_function, division, absolute_import, unicode_literals

# External modules
from numpy import sum, matrix, diag, sqrt, corrcoef, isclose, trace, identity,\
    atleast_2d, zeros, allclose, log, ndarray
from collections import namedtuple as nt
from numpy.linalg.linalg import eig, det

MomentsSpec = nt('moments', ['mean', 'cov', 'icov'])

class Surprise(object):
    """
    Module for estimating relative entropy and surprise from samples or
    moments of the distributions assuming they are Gaussian.
    
    For a reference see arXiv:1402.3593
    """

    def __init__(self, atol=1e-08, rtol=1e-05):
        """
        Constructor of surprise module.
        
        :param atol: absolute tolerance for covariance matrix inversion
        :param rtol: relative tolerance for covariance matrix inversion
        """
        self.atol = atol
        self.rtol = rtol
        try:
            from rpy2.robjects.packages import importr
            from rpy2.rinterface import RRuntimeError
            try:
                self.cqf = importr(str("CompQuadForm"))
                self.rpy2 = True
            except RRuntimeError:
                self.rpy2 = False
                print("R package CompQuadForm not found -> no p-value for the Surprise")
        except ImportError:
            self.rpy2 = False
            print("No rpy2 module found -> no p-value for the Surprise")

    def __call__(self, dist1, dist2, mode = 'add', dist3 = None,
                 weights1 = None, weights2 = None, weights3 = None,
                 bits = True, getChi2 = False, cqf_method = 'davies',
                 verbose = False):
        """
        Routine for calculating relative entropy and surprise from samples
        or moments. It calculates D(dist2||dist1).
        
        :param dist1, dist2: samples (#samples x # dims array) or moments
        ((mean, covariance) tuple) of distribution 1, 2
        :param mode (optional): relation between dist1 and dist2; either
         * 'add' if dist1 is the prior of dist2
         * 'replace' if dist1 and dist2 are independently derived posteriors
         and the prior is much wider than the constraints
         * 'partial' if dist1 and dist2 are independently derived using the 
         same prior dist3
         Default: add
        :param dist3 (optional): samples (#sample x # dims array) or moments
        (mean, covariance tuple) of joint prior of dist1 and dist2 if mode is
        chosen to be 'partial'; default: None
        :param weights1, weights2, weights3 (optional): #samples array with 
        weights for each sample of dist1, dist2, dist3; default: None
        :param bits: relative entropy results in units of bits; if False, the
        results are in nats; default: True
        :param getChi2: Output weights lambdas for the weighted sum of 
        chi-squared distributed variables and dmu, the value of the
        generalized chi-square; default: False
        :param cqf_method: Method used to estimate the p-value of the
        surprise in CompQuadForm. One out of 'davies', 'farebrother',
        'imhof', 'liu', 'ruben'; default: davies
        :param verbose: print output from CompQuadForm to screen;
        default: False
        
        :returns D, ere, S, sD, p, lambdas, dmu: Relative entropy, expected
        relative entropy, surprise, sigma(D), p-value of the surprise; if 
        getChi2 is True, it also returns lambdas and dmu. 
        """
        mes = "dist 1 has to be tuple or array"
        assert type(dist1) == tuple or type(dist1) == ndarray, mes
        mes = "dist 2 has to be tuple or array"
        assert type(dist2) == tuple or type(dist2) == ndarray, mes 
        if dist3 is not None:
            mes = "dist 3 has to be tuple or array"
            assert type(dist3) == tuple or type(dist3) == ndarray, mes
        m1, m2, m3, d = self.setup(dist1, dist2, dist3, weights1, weights2,
                                   weights3)
        if mode == 'add':
            D, ere, S, sD, lambdas, dmu = self.complementary(m1, m2, d)
        elif mode == 'replace':
            D, ere, S, sD, lambdas, dmu = self.replacement(m1, m2, d)
        elif mode == 'partial':
            D, ere, S, sD, lambdas, dmu = self.jointprior(m1, m2, m3, d)
        else:
            raise Warning('Mode has to be add, replace, or partial.')
        
        # Try to estimate p-value
        if self.rpy2:
            res = self.getPValue(lambdas, dmu, cqf_method)
            p = res[2][0]
            if verbose:
                print('CompQuadForm results:')
                print(res)
        else:
            p = None

        # Turn results from nats into bits
        if bits:
            l2 = log(2)
            D /= l2
            ere /= l2
            S /= l2
            sD /= l2
            
        if getChi2 is True:
            return D, ere, S, sD, p, lambdas, dmu
        else:
            return D, ere, S, sD, p

    def getPValue(self, lambdas, dmu, cqf_method):
        """
        Calculate p-value with CompQuadForm using rpy2.

        :param lambdas: weights for the weighted sum of chi-squared
        distributed variables
        :param dmu: value of the generalized chi-square variable
        corresponding to the surprise
        :param cqf_method: Method used to estimate the p-value of the
        surprise in CompQuadForm. One out of 'davies', 'farebrother',
        'imhof', 'liu', 'ruben'
        :returns res: result from CompQuadForm
        """
        res = self.cqf.__dict__[cqf_method](dmu, lambdas.tolist())
        return res

    def getRelEnt(self, m1, m2, d):
        """
        Estimates relative entropy D(dist2||dist1) from moments of dist1 and
        dist2 in nats.
                
        :param m1: moments instance for dist1  
        :param m2: moments instance for dist2
        :param d: dimensionality
        :return Dpart, deltamu: Covariance-dependent part and mean dependent
        part of the relative entropy
        """
        Dpart = -log(det(m2.cov) / det(m1.cov))
        Dpart += trace(m2.cov * m1.icov) - d
        diff = m1.mean - m2.mean
        deltamu = (diff * m1.icov * diff.T).A1[0]
        return Dpart, deltamu

    def complementary(self, m1, m2, d):
        """
        Return expected relative entropy, surprise and standard deviation
        for dist1 being the prior of dist2.

        :param m1: moments instance for dist1  
        :param m2: moments instance for dist2
        :param d: dimensionality
        :returns D, ere, S, sigmaD, lambdas, deltamu: Relative entropy,
        expected relative entropy, surprise, sigma(D), lambdas and dmu of 
        generalized chi-squared
        """
        Dpart, deltamu = self.getRelEnt(m1, m2, d)
        ASigma = matrix(identity(d) - m1.icov * m2.cov)
        lambdas = eig(ASigma)[0]
        ere = -.5 * log(det(m2.cov) / det(m1.cov))
        D = .5 * (Dpart + deltamu)
        S = D - ere
        sigmaD = trace(ASigma * ASigma)
        sigmaD = sqrt(.5 * sigmaD)
        return D, ere, S, sigmaD, lambdas, deltamu

    def replacement(self, m1, m2, d):
        """
        Return expected relative entropy, surprise and standard deviation
        for dist1 and dist2 being separately analysed posteriors.
        
        :param m1: moments instance for dist1  
        :param m2: moments instance for dist2
        :param d: dimensionality
        :returns D, ere, S, sigmaD, lambdas, deltamu: Relative entropy,
        expected relative entropy, surprise, sigma(D), lambdas and dmu of 
        generalized chi-squared
        """
        Dpart, deltamu = self.getRelEnt(m1, m2, d)
        ASigma = matrix(m1.icov * m2.cov + identity(d))
        lambdas = eig(ASigma)[0]
        ere = .5 * (Dpart + trace(ASigma))
        D = .5 * (Dpart + deltamu)
        S = D - ere
        sigmaD = trace(ASigma * ASigma)
        sigmaD = sqrt(.5 * sigmaD)
        return D, ere, S, sigmaD, lambdas, deltamu

    def jointprior(self, m1, m2, m3, d):
        """
        Return expected relative entropy, surprise and standard deviation
        for dist1 and dist2 having joint prior dist3.
        
        Note that the generalized chi-squared distribution is approximated
        to be central in this case. This is only a good approximation if the
        joint prior is weak compared to m1 and m2.
        
        :param m1: moments instance for dist1  
        :param m2: moments instance for dist2
        :param m3: moments instance for dist3
        :param d: dimensionality
        :returns D, ere, S, sigmaD, lambdas, deltamu: Relative entropy,
        expected relative entropy, surprise, sigma(D), approximate lambdas
        and dmu of generalized chi-squared

        """
        if m3 is None:
            mes='Samples or moments of joint prior has to be specified'
            raise Warning(mes)

        Dpart, deltamu = self.getRelEnt(m1, m2, d)

        Q = m2.icov - m3.icov
        T = (m3.mean - m1.mean) * m3.icov
        W = m2.cov * m1.icov * m2.cov
        ASigma = (Q * W) * (identity(d) + Q * m1.cov)
        lambdas = eig(ASigma)[0]
        twt = (T * W * T.T).A1[0]

        ere = .5 * (Dpart + trace(ASigma) + twt)
        D = .5 * (Dpart + deltamu)
        S = D - ere

        temp = W * (Q + Q * m1.cov * Q) * W
        sigmaD = trace(ASigma * ASigma) + 2 * (T * temp * T.T).A1[0]
        sigmaD = sqrt(.5 * sigmaD)
        
        # That step is a kind of crude correction for the ignored 
        # non-centrality of the chi-squared
        deltamu -= twt
        
        return D, ere, S, sigmaD, lambdas, deltamu

    def setup(self, dist1, dist2, dist3, weights1, weights2, weights3):
        """
        Calculates moments of distributions and sets up the estimator.
        
        :param dist1, dist2, dist3: samples (#samples x # dims array) or
        moments ((mean, covariance) tuple) of distribution 1, 2, 3
        :param weights1, weights2, weights3: #samples array with weights for
        each sample of dist1, dist2, dist3
        :returns m1, m2, m3, d: moments instance for dist1, dist2, and dist3,
        dimensionality of parameter space d
        """

        m1 = self.load(dist1, weights1)
        m2 = self.load(dist2, weights2)
        if dist3 is None:
            m3 = None
        else:
            m3 = self.load(dist3, weights3)
        d = m1.mean.shape[1]
        return m1, m2, m3, d

    def load(self, dist, weights):
        """
        Get mean, covariance and inverse covariance from samples or
        (mean,covariance) tuple in dist.
        
        :param dist: A (#sample x # dims) array or list containing samples
            or a tuple containing mean and covariance.
        :param weights: #samples array with weights for sample dist
        :returns m: moments instance
        """
        if type(dist) is tuple:
            Mu = matrix(dist[0])
            Sigma = matrix(dist[1])
            s = sqrt(diag(Sigma))
            S = matrix(diag(s))
            iS = matrix(diag(1.0 / s))
            Corr = iS * Sigma * iS
        else:
            dist = atleast_2d(dist)
            if dist.shape[0] == 1:
                dist = dist.T
            if weights is None:
                Mu = matrix(dist.mean(axis = 0))
                Corr = matrix(corrcoef(dist.T))
                s = dist.std(axis = 0, ddof = 1)
                S = matrix(diag(s))
                iS = matrix(diag(1.0 / s))
                Sigma = S*Corr*S
            else:
                s=sum(weights)
                if s != 1:
                    w = weights/float(s)
                else:
                    w = weights
                Mu = self.weightedMean(dist, w)
                Corr, s = self.weightedCorr(dist, w, Mu)
                Corr = matrix(Corr)
                S = matrix(diag(s))
                iS = matrix(diag(1.0 / s))
                Mu = matrix(Mu)
                Sigma = S * Corr * S
         
        iCorr = self.secInv(Corr)
        iSigma = iS * iCorr * iS
        return MomentsSpec(mean = Mu, cov = Sigma, icov = iSigma)
        
    def weightedMean(self, sample, weights):
        """
        Get mean of weighted sample.
        
        :param sample: A (#sample x # dims) array of samples
        :param weights: #samples array with weights for sample
        :returns mu: weighted mean
        """
        mu = (weights.reshape(-1,1) * sample).sum(axis = 0)
        return mu
    
    def weightedCorr(self,dist,weights,mu):
        """
        Get correlation matrix of weighted sample.
        
        :param dist: A (#sample x # dims) array of samples
        :param weights: #samples array with weights for sample
        :param mu: weighted mean of sample
        :returns c, s: correlation matrix and standard deviation of weighted
        sample
        """
        D = dist.shape[1]
        d = dist - mu
        s = zeros(D)
        c = identity(D)
        for i in range(D):
            s[i] = sqrt(sum(d[:, i] * d[:, i] * weights))
            for j in range(i):
                temp = sum(d[:, i] * d[:, j] * weights)
                c[i,j] = temp / s[i] / s[j]
                c[j,i] = c[i,j]
        n = (1. - sum(weights * weights))
        return c, s/sqrt(n)
    
    def secInv(self, Mat):
        """
        Matrix inversion. Returns assertion error if the inverted matrix
        times the matrix is not close to identity.

        :param Mat: matrix to invert
        :returns iMat: inverted matrix
        """
        iMat = Mat.I
        mes = 'Inversion failed'
        assert allclose(iMat * Mat, matrix(identity(iMat.shape[0])), 
                        self.rtol, self.atol), mes
        return iMat
