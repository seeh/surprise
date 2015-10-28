__author__ = 'Sebastian Seehars'
__email__ = 'seehars@phys.ethz.ch'
__version__ = '0.1.0'
__credits__ = 'ETH Zurich, Institute for Astronomy'

#for py27+py33 compatibility
try:
    from surprise import Surprise
except ImportError:
    from surprise.surprise import Surprise