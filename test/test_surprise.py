
# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

"""
Tests for `surprise` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pytest
from surprise import Surprise
from numpy import atleast_2d, allclose, matrix, arange, ones, cov, log, sqrt


class TestSurprise(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        self.td=atleast_2d(arange(0,10,1)).T
        self.tw=ones(10)
        self.surprise=Surprise()
        self.tdt1=([0.],[[1.]])
        self.tdt2=([1.],[[0.5]])
        self.tdt3=([1.],[[2.]])

    def test_weightedLoad(self):
        res = self.surprise.load(self.td, self.tw)
        assert allclose(res.mean, matrix(self.td.mean(0)))
        assert allclose(res.cov, matrix(cov(self.td.T)))
        assert allclose(res.icov, matrix(cov(self.td.T)).I)
        
    def test_load(self):
        res = self.surprise.load(self.td, None)
        assert allclose(res[0], matrix(self.td.mean(0)))
        assert allclose(res[1], matrix(cov(self.td.T)))
        assert allclose(res[2], matrix(cov(self.td.T)).I)

    def test_getExpectedRelEnt(self):
        mode = ['add', 'replace', 'partial']
        D = .5 + log(2)
        ere = [log(2), log(2) + 1, log(2) + .5]
        S = [.5, -.5, 0]
        sD = [.5, 1.5, sqrt(255.)/16.]
        lams = [.5, 1.5, 15./16.]
        dmus = [1., 1., 15./16.]
        # numerical values from R
        p = [0.1573089, 0.4142238, 0.3173187]
        for i, m in enumerate(mode):
            res = self.surprise(self.tdt1, self.tdt2, m, self.tdt3,
                                getChi2 = True, bits = False)
            assert allclose(D, res[0]*2.)
            assert allclose(ere[i], res[1]*2.)
            assert allclose(S[i], res[2]*2.)
            assert allclose(sD[i], res[3]*sqrt(2.))
            assert allclose(lams[i], res[5])
            assert allclose(dmus[i], res[6])
            if self.surprise.rpy2:
                assert allclose(p[i], res[4])
            else:
                assert allclose(None, res[4])
        
    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass

if __name__ == '__main__':
    pytest.main()