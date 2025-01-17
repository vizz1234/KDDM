import numpy as np
from sklearn.tree import DecisionTreeRegressor


class Node(): 
    """
    Implements an individual node in the Decision Tree. 
    """

    def __init__(self, y):
        self.y = y                 # Values of samples assigned to that node
        self.score = np.inf        # RSS score of the node (measure of impurity)
        self.feature_idx = None    # The feature used for the split (column number)
        self.threshold = None      # Threshold used for the splitt (scalar value)
        self.left_child = None     # Left child of the node (of type Node)
        self.right_child = None    # Right child of the node (of type Node)
        
    def is_leaf(self):
        if self.feature_idx is None:
            return True
        else:
            return False



class MyDecisionTreeRegressor:
    
    def __init__(self, max_depth=None, min_samples_split=2):    
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.node = None
    
    def calc_rss_score_node(self, y):
        return np.sum(np.square(y - np.mean(y)))
    
    def calc_rss_score_split(self, y_left, y_right):      
        return self.calc_rss_score_node(y_left) + self.calc_rss_score_node(y_right)
    
    def calc_thresholds(self, x):               
        ## Get unique values to handle duplicates; return values will already be sorted
        values_sorted = np.unique(x)
        ## The thresholds are all values "between" the unique data values
        return (values_sorted[:-1] + values_sorted[1:]) / 2.0
    
    def create_split(self, x, threshold):
        ## Get all row indices where the value is <= threshold
        indices_left = np.where(x <= threshold)[0]
        ## Get all row indices where the value is > threshold
        indices_right = np.where(x > threshold)[0]
        ## Return split
        return indices_left, indices_right
    
    
    
    def calc_best_split(self, X, y):
        ## Initialize the return values
        best_score, best_threshold, best_feature_idx, best_split = np.inf, None, None, None

        ## Loop through all features (columns of X) to find the best split
        for feature_idx in range(X.shape[1]):

            # Get all values for current feature
            x = X[:, feature_idx]
            
            ################################################################################
            ### Your code starts here ###################################################### 
            thresholds = self.calc_thresholds(x)
            for threshold in thresholds:
                idx_left, idx_right = self.create_split(x, threshold)
                rss_score = self.calc_rss_score_split(y[idx_left], y[idx_right])
                if rss_score < best_score:
                    best_score = rss_score
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_split = (idx_left, idx_right)
    

        
                    
            ### Your code ends here ########################################################
            ################################################################################                      
            
        return best_score, best_threshold, best_feature_idx, best_split
    
    
    def fit(self, X, y):
        
        ## Initialize Decision Tree as a single root node
        self.node = Node(y)

        ## Start recursive building of Decision Tree
        self._fit(X, y, self.node)
        
        ## Return Decision Tree object
        return self
    
    
    def _fit(self, X, y, node, depth=0):    

        ## Calculate and set RSS score of the node itself
        node.score = self.calc_rss_score_node(y) 

        #########################################################################################
        ### Your code starts here ###############################################################
        if len(X) < self.min_samples_split:
            return
        if self.max_depth:
            if depth >= self.max_depth:
                return
        ### Your code ends here #################################################################
        #########################################################################################

        ## Calculate the best split
        score, threshold, feature_idx, split = self.calc_best_split(X, y)

        #########################################################################################
        ### Your code starts here ###############################################################
        if split is None or (len(split[0]) < 1 or len(split[1]) < 1):
            return


        ### Your code ends here #################################################################
        #########################################################################################
        
        ## Split the input and labels using the indices from the split
        X_left, X_right = X[split[0]], X[split[1]]
        y_left, y_right = y[split[0]], y[split[1]]

        ## Update the parent node based on the best split
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left_child = Node(y_left)
        node.right_child = Node(y_right)

        ## Recursively fit both child nodes (left and right)
        self._fit(X_left, y_left, node.left_child, depth=depth+1)
        self._fit(X_right, y_right, node.right_child, depth=depth+1)   

  
    def predict(self, X):
        ## Return list of individually predicted labels
        return np.array([ self.predict_sample(self.node, x) for x in X ])


    def predict_sample(self, node, x):        
        ## If the node is a leaf, return the mean value as the prediction
        if node.is_leaf():
            return np.mean(node.y)

        ## If the node is not a leaf, go down the left or right subtree (depending on the feature value)
        if x[node.feature_idx] <= node.threshold:
            return self.predict_sample(node.left_child, x)
        else:
            return self.predict_sample(node.right_child, x)
        
        
    def get_node_count(self):
        return self._get_node_count(self.node)
        
    def _get_node_count(self, node):
        if node.is_leaf():
            return 1
        else:
            return 1 + self._get_node_count(node.left_child) + self._get_node_count(node.right_child)
        
        
        
        
        
        
        
        
class MyRandomForestRegressor:
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.estimators = []
        
        
    def bootstrap_sampling(self, X, y):
        X_bootstrap, y_bootstrap = None, None

        N, d = X.shape

        #########################################################################################
        ### Your code starts here ###############################################################

        X_bootstrap_idx = np.random.choice(np.arange(N), N, replace=True)
        X_bootstrap = X[X_bootstrap_idx]
        y_bootstrap = y[X_bootstrap_idx]



        ### Your code ends here #################################################################
        #########################################################################################

        return X_bootstrap, y_bootstrap
    
        
    def feature_sampling(self, X):
        N, d = X.shape

        X_feature_sampled, indices_sampled = None, None

        #########################################################################################
        ### Your code starts here ###############################################################
        no_of_features = int(np.ceil(self.max_features * d))
        indices_sampled = np.random.choice(np.arange(d), no_of_features, replace = False)
        X_feature_sampled = X[:, indices_sampled]
        


        ### Your code ends here #################################################################
        #########################################################################################    

        return X_feature_sampled, indices_sampled
    
    
    def fit(self, X, y):
        
        self.estimators = []
        
        for _ in range(self.n_estimators):
            
            regressor, indices_sampled = None, None
            
            #########################################################################################
            ### Your code starts here ###############################################################
            regressor_obj = MyDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_bootstrap, Y_bootstrap = self.bootstrap_sampling(X, y)
            X_feature_sampled, indices_sampled = self.feature_sampling(X_bootstrap)
            regressor = regressor_obj.fit(X_feature_sampled, Y_bootstrap)
            
            ### Your code ends here #################################################################
            #########################################################################################    
        
            self.estimators.append((regressor, indices_sampled))
            
        return self
            
            
    def predict(self, X):
        
        predictions = []
        
        #########################################################################################
        ### Your code starts here ###############################################################
        tree_pred = np.array([regressor.predict(X[:, features_idx]) for regressor, features_idx in self.estimators])
        predictions = np.mean(tree_pred, axis = 0)
        
        ### Your code ends here #################################################################
        #########################################################################################        
        
        return predictions
    
    
    
    
    
    
    
class MyGradientBoostingRegressor:
    
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2):
        self.estimators = []
        self.lr = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.initial_f = None
        
    
    def fit(self, X, y):
        
        self.estimators = []
        
        self.initial_f = y.mean()
        
        # Set initial prediction f_0(x_i) to mean for all data points
        f = np.array([ self.initial_f ]*X.shape[0])

        for m in range(1, self.n_estimators+1):

            #####################################################################################
            ### Your code starts here ###########################################################             
            
            ## Use your implementation of MyDecisionTreeRegressor in here!
            residuals = y - f
            decision_stump = MyDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            decision_stump.fit(X, residuals)
            self.estimators.append(decision_stump)
            decision_stump_predictions = decision_stump.predict(X)
            f += self.lr * decision_stump_predictions
            
            ### Your code ends here #############################################################
            #####################################################################################                 
       
        return self
    
    
    def predict(self, X):
        
        y_pred = np.array([self.initial_f]*X.shape[0])
        
        #########################################################################################
        ### Your code starts here ###############################################################
        for stump in self.estimators:
            y_pred += self.lr * stump.predict(X)        

        
        ### Your code ends here #################################################################
        #########################################################################################        
        
        return y_pred
    
