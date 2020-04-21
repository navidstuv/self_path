from utils import stats
import numpy as np

pred = np.load('../pred_mag2.npy')
true = np.load('../true_mag2.npy')
stats(pred, true, 0.36460841)