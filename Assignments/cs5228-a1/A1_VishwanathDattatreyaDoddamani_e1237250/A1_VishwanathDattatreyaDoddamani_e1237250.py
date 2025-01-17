import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances



def clean(df_condos_dirty):
    """
    Handle all "dirty" records in the condos dataframe

    Inputs:
    - df_condos_dirty: Pandas dataframe of dataset containing "dirty" records

    Returns:
    - df_condos_cleaned: Pandas dataframe of dataset without "dirty" records
    """   
    
    # We first create a copy of the dataset and use this one to clean the data.
    df_condos_cleaned = df_condos_dirty.copy()

    #########################################################################################
    ### Your code starts here ###############################################################
    df_condos_dirty['transaction_id'] = df_condos_dirty['transaction_id'].str.replace('X', '')
    df_condos_cleaned['transaction_id'] = df_condos_dirty['transaction_id'].astype(int)

    
    
    zero_pd_indices = df_condos_cleaned['postal_district'] == 0
    area_dic = df_condos_dirty[df_condos_dirty['postal_district'] > 0].set_index('subzone')['postal_district'].to_dict()
    df_condos_cleaned.loc[zero_pd_indices, 'postal_district'] = df_condos_cleaned.loc[zero_pd_indices, 'subzone'].map(area_dic).fillna(0).astype(int)
    df_condos_cleaned.drop(df_condos_cleaned[df_condos_cleaned['postal_district'] == 0].index, inplace=True)

    median_area_by_type = df_condos_cleaned.groupby('type')['area_sqft'].median()

    df_condos_cleaned.loc[df_condos_cleaned['area_sqft'] < 0, 'area_sqft'] = (df_condos_cleaned['type'].map(median_area_by_type))   

    df_condos_cleaned = df_condos_cleaned.drop_duplicates()


    
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_condos_cleaned


def handle_nan(df_condos_nan):
    """
    Handle all nan values in the condos dataframe

    Inputs:
    - df_condos_nan: Pandas dataframe of dataset containing nan values

    Returns:
    - df_condos_no_nan: Pandas dataframe of dataset without nan values
    """       

    # We first create a copy of the dataset and use this one to clean the data.
    df_condos_no_nan = df_condos_nan.copy()

    #########################################################################################
    ### Your code starts here ###############################################################

    df_condos_no_nan = df_condos_no_nan.drop('url', axis = 1)

    subzone_to_planning_area = df_condos_no_nan.dropna(subset=['planning_area']).set_index('subzone')['planning_area'].to_dict()
    df_condos_no_nan['planning_area'] = df_condos_no_nan['planning_area'].fillna(
        df_condos_no_nan['subzone'].map(subzone_to_planning_area)
    )

    df_condos_no_nan['price_per_sqft'] = df_condos_no_nan['price'] / df_condos_no_nan['area_sqft']
    df_condos_pps_median = df_condos_no_nan.groupby('type')['price_per_sqft'].median()
    df_condos_no_nan.loc[df_condos_no_nan['price'].isna(), 'price'] = ((df_condos_no_nan['type'].map(df_condos_pps_median)) * df_condos_no_nan['area_sqft']).astype('int')
    df_condos_no_nan['price'] = df_condos_no_nan['price'].astype('float')
    df_condos_no_nan = df_condos_no_nan.drop('price_per_sqft', axis=1)

    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_condos_no_nan


def extract_facts(df_condos_facts):
    """
    Extract the facts as required from the condos dataset

    Inputs:
    - df_condos_facts: Pandas dataframe of dataset containing the cars dataset

    Returns:
    - Nothing; you can simply us simple print statements that somehow show the result you
      put in the table; the format of the  outputs is not important; see example below.
    """       

    #########################################################################################
    ### Your code starts here ###############################################################
    df_condos_facts['price_per_sqft'] = round(df_condos_facts['price'] / df_condos_facts['area_sqft'], 2)
    df_condos_facts[df_condos_facts['postal_district'] == 11]['price_per_sqft'].max()
    max_ind_q5 = df_condos_facts[df_condos_facts['postal_district'] == 11]['price_per_sqft'].idxmax()

    print('-------------------------------------------------------------------------------')
    print('Q1: ', pd.to_datetime(df_condos_facts['date_of_sale'], format = '%b-%y').min())
    print('-------------------------------------------------------------------------------')
    print('Q2: ', df_condos_facts.groupby('type').count()['transaction_id'])
    print('-------------------------------------------------------------------------------')
    print('Q3: ', len(df_condos_facts[(df_condos_facts['subzone'] == 'redhill') & (df_condos_facts['price'] > 2000000)]))
    print('-------------------------------------------------------------------------------')
    print('Q4: ', df_condos_facts.groupby('planning_area').count()['transaction_id'].idxmax(), ' ', df_condos_facts.groupby('planning_area').count()['transaction_id'].max())
    print('-------------------------------------------------------------------------------')
    print('Q5: ', df_condos_facts[df_condos_facts['postal_district'] == 11][['name', 'price_per_sqft']].loc[max_ind_q5])
    print('-------------------------------------------------------------------------------')
    print('Q6: Correlation Matrix \n', df_condos_facts[['price', 'area_sqft']].corr())
    print('-------------------------------------------------------------------------------')
    print('Q7: ', len(df_condos_facts[(df_condos_facts['floor_level'] == '51 to 55') | (df_condos_facts['floor_level'] == '56 to 60')]))
    print('-------------------------------------------------------------------------------')    

    
    ### Your code ends here #################################################################
    #########################################################################################

    
    
    
    
    
    
    
class MyKMeans:
    
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = 0
        
        
    def initialize_centroids(self, X):
    
        # Pick the first centroid randomly
        c1 = np.random.choice(X.shape[0], 1)

        # Add first centroids to the list of cluster centers
        self.cluster_centers_ = X[c1]

        # Calculate and add c2, c3, ..., ck (we assume that we always have more unique data points than k!)
        while len(self.cluster_centers_) < self.n_clusters:

            # c is a data point representing the next centroid
            c = None

            #########################################################################################
            ### Your code starts here ###############################################################
            distances = euclidean_distances(X, self.cluster_centers_)
            min_distances = np.min(distances, axis = 1)
            min_distances = min_distances ** 2
            min_distances_p = min_distances / np.sum(min_distances)
            c = X[np.random.choice(X.shape[0], p = min_distances_p)]
            ### Your code ends here #################################################################
            #########################################################################################                

            # Add next centroid c to the array of already existing centroids
            self.cluster_centers_ = np.concatenate((self.cluster_centers_, [c]), axis=0)

    
    
    def assign_clusters(self, X):
        # Reset all clusters (i.e., the cluster labels)
        self.labels_ = None

        #########################################################################################
        ### Your code starts here ############################################################### 
        distances = euclidean_distances(X, self.cluster_centers_)
        self.labels_ = np.argmin(distances, axis = 1)     
        ### Your code ends here #################################################################
        #########################################################################################

    

    def update_centroids(self, X):

        # Initialize list of new centroids with all zeros
        new_cluster_centers_ = np.zeros_like(self.cluster_centers_)

        for cluster_id in range(self.n_clusters):

            new_centroid = None

            #########################################################################################
            ### Your code starts here ###############################################################  
            indices = (self.labels_ == cluster_id)
            new_centroid = np.mean(X[indices], axis = 0)

            
            
            ### Your code ends here #################################################################
            #########################################################################################

            new_cluster_centers_[cluster_id] = new_centroid  
            
        # Check if old and new centroids are identical; if so, we are done
        done = (self.cluster_centers_ == new_cluster_centers_).all()    
        
        # Update list of centroids
        self.cluster_centers_ = new_cluster_centers_

        # Return TRUE if the centroids have not changed; return FALSE otherwise
        return done
    
    
    def fit(self, X):
        
        self.initialize_centroids(X)

        self.n_iter_ = 0
        for _ in range(self.max_iter):
            
            # Update iteration counter
            self.n_iter_ += 1
            
            # Assign cluster
            self.assign_clusters(X)

            # Update centroids
            done = self.update_centroids(X)

            if done:
                break

        return self    