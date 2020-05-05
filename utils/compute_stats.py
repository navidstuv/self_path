from utils import stats
import numpy as np

pred = np.load('../pred_test_stain1.npy')
true = np.load('../true_test_stain1.npy')
stats(pred, true, 0.5)
