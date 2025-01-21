import time
import numpy as np
def GOA(X, fitness, lowerbound, upperbound,Max_iterations):
    SearchAgents,dimension = X.shape[0],X.shape[1]
    fit = np.zeros((SearchAgents))
    for i in np.arange(1, SearchAgents + 1).reshape(-1):
        L = X[i,:]
        fit[i] = fitness(L)
    ct = time.time()
    GOA_curve = np.zeros((Max_iterations))
    for t in np.arange(1, Max_iterations + 1).reshape(-1):
        best, blocation = np.amin(fit)
        if t == 1:
            Xbest = X[blocation,:]
            fbest = best
        else:
            if best < fbest:
                fbest = best
                Xbest = X[blocation,:]
        X_P1 = np.zeros((SearchAgents, dimension))
        F_P1 = np.zeros((SearchAgents))
        for i in np.arange(1, SearchAgents + 1).reshape(-1):
            if np.random.rand() < 0.5:
                I = np.round(1 + np.random.rand(1, 1))
                RAND = np.random.rand(1, 1)
            else:
                I = np.round(1 + np.random.rand(1, dimension))
                RAND = np.random.rand(1, dimension)
            X_P1[i, :] = X[i,:] + np.multiply(RAND, (Xbest - np.multiply(I, X[i,:])))
            X_P1[i, :] = np.amax(X_P1[i,:], lowerbound)
            X_P1[i, :] = np.amin(X_P1[i,:], upperbound)
            L = X_P1[i,:]
            F_P1[i] = fitness(L)
            if F_P1[i] < fit[i]:
                X[i, :] = X_P1[i,:]
                fit[i] = F_P1[i]
        X_P2 = np.zeros((SearchAgents,dimension))
        F_P2 = np.zeros((SearchAgents))
        for i in np.arange(1, SearchAgents + 1).reshape(-1):
            X_P2[i, :] = X[i,:] + np.multiply((1 - 2 * np.random.rand(1, 1)), (
                        lowerbound / t + np.multiply(np.random.rand(1, 1), (upperbound / t - lowerbound / t))))
            X_P2[i, :] = np.amax(X_P2[i,:], lowerbound / t)
            X_P2[i, :] = np.amin(X_P2[i,:], upperbound / t)
            X_P2[i, :] = np.amax(X_P2[i,:], lowerbound)
            X_P2[i, :] = np.amin(X_P2[i,:], upperbound)
            L = X_P2[i,:]
            F_P2[i] = fitness(L)
            if F_P2[i] < fit[i]:
                X[i, :] = X_P2[i,:]
                fit[i] = F_P2[i]
        GOA_curve[t] = fbest
    Best_score = fbest
    Best_pos = Xbest
    ct = time.time()-ct
    return Best_score, Best_pos, GOA_curve,ct
