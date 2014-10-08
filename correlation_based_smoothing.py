import numpy as np

class CorrMatrix():
    """
    creates the correlation matrix for a data set in a
    memory efficient manner.

    additionally, allows the option to enable feature selection, cutting off
    highly correlated features at a given threshold.

    of the two highly correlated features, the feature with the highest average
    correlation is removed.

    returns a list of variables which have been removed from the matrix.
    """
    def __init__(self, correlation_threshold=.9):
        self.correlation_threshold = correlation_threshold

    def select_features(self, feature_data_set):
        corr_mat = create_correlation_matrix(feature_data_set)
        print "Performing correlation based feature selection"

        #setting diagonals to 0
        most_correlated = []
        diags = corr_mat.shape[0]
        corr_mat[range(diags), range(diags)] = 0

        for i in xrange(corr_mat.shape[1]):
            if np.max(np.ma.masked_array(corr_mat[:, i], np.isnan(corr_mat[:, i]))) > self.correlation_threshold:
                j = corr_mat[:, i].argmax(axis=0)
                if np.mean(np.ma.masked_array(corr_mat[:, i], np.isnan(corr_mat[:, i]))) >= \
                        np.mean(np.ma.masked_array(corr_mat[:, j], np.isnan(corr_mat[:, j]))):
                    most_correlated.append(i)
                else:
                    most_correlated.append(j)

        corr_list = list(set(most_correlated))
        corr_list.sort()
        print "Throwing out Correlated Variables:", corr_list
        non_correlated = [idx for idx in sorted(range(feature_data_set.feature_mat.shape[1])) if idx not in corr_list]
        print "Keeping Variables:", non_correlated
        return non_correlated

#helper function
def create_correlation_matrix(feature_data_set):
    """
    creates correlation matrix iteratively as to be memory efficient
    """
    cont_cols = feature_data_set.all_cols_by_type(is_cont=True)
    print "Cont Cols :", cont_cols
    corr_matrix = np.eye(feature_data_set.feature_mat.shape[1])
    for i in xrange(feature_data_set.feature_mat.shape[1]):
        for j in xrange(i, feature_data_set.feature_mat.shape[1]):
            if (i in cont_cols) and (j in cont_cols) and i != j:
                corr_matrix[i, j] = np.corrcoef(
                    feature_data_set.feature_mat[:, i].T,
                    feature_data_set.feature_mat[:, j].T)[0, 1]
                corr_matrix[j, i] = corr_matrix[i, j]
            else:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0
    return corr_matrix