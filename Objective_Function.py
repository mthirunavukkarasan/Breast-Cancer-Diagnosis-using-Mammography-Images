import numpy as np
from Global_Vars import Global_Vars
from Model_AViT_SNetv2 import Model_AViT_SNetv2
def Objfun_Cls(Soln):
    images = Global_Vars.Images
    Targ = Global_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Eval = Model_AViT_SNetv2(images,Targ, sol.astype('int'))
        Fitn[i] =(1/( Eval[4] + Eval[7])) +Eval[8]+Eval[9]+Eval[11]
    return Fitn



