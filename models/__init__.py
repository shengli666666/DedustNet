import sys,os
dir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
#from DwtFormer import DwtFormer
from PerceptualLoss import LossNetwork as PerLoss
