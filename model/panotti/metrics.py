import numpy as np
import pdb
def shomik_tag_score(ground_truth, predict,threshold):
        '''
        This function calculates how many of the tags the model was able to correctly predict
        '''
        #pdb.set_trace()
        ground_truth = np.reshape(ground_truth,(63,))
        #first get the tag locations where ground truth has a '1'
        one_locs = np.where(ground_truth == 1)
        #now extract these locations from the prediction array
        predict_locs = predict[one_locs]
        #pdb.set_trace()
        threshold_locs = np.where(predict_locs > threshold)
        return float(threshold_locs[0].shape[0])/predict_locs.shape[0]
