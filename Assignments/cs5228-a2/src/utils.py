import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import chain, combinations
from efficient_apriori import apriori



##
## Clustering Auxiliary Methods
##

def plot_kmeans_clustering(X, labels, cluster_centers):
    plt.figure()

    for cluster_id in np.unique(labels):
        cluster_sample_indices = np.where(labels == cluster_id)[0]
        X_cluster = X[cluster_sample_indices]
        if X_cluster.shape[0] > 0:
            plt.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color='C{}'.format(cluster_id), s=150)

            for x in X_cluster:
                plt.plot([x[0],cluster_centers[cluster_id][0]], [x[1],cluster_centers[cluster_id][1]], '--', linewidth=0.5, color='k'.format(cluster_id))
            
        plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='+', color='k', s=250, lw=5)

    plt.tight_layout()
    
    plt.show()




##
## ARM Auxiliary Methods
##

def powerset(iterable, min_len=None, max_len=None):
    s = list(iterable)
    if min_len is None:
        min_len = 0
    if max_len is None:
        max_len = len(s)
    return chain.from_iterable(combinations(sorted(s), r) for r in range(min_len, max_len+1))


def merge_itemsets(a, b):
    itemset = set(a).union(set(b))
    itemset = tuple(sorted(itemset))
    return itemset


def unique_items(transactions):
    unique_items = set()
    
    for t in transactions:
        unique_items.update(set(t))
    
    return unique_items


def support_count(transactions, itemset):
    
    # Set the initial support count to 0
    support_count = 0
    
    # Check for each transaction if it contains the itemset
    # If so, increment support count
    for t in transactions:
        if set(itemset).issubset(set(t)):
            support_count += 1
            
    # Return support count
    return support_count


def support(transactions, itemset):
    
    if len(transactions) == 0:
        return 0.0
    
    # Return support count
    return support_count(transactions, itemset) / len(transactions)


def confidence(transactions, association_rule):
    # Split association rule into itemsets X and Y (reflecting X=>Y)
    X, Y = association_rule
    
    # Calculate the support count for X
    support_count_X = support_count(transactions, X)
    
    # If the support count of X is 0, return 0 to avoid division by zero
    if support_count_X == 0:
        return 0.0
    
    # Calculate X*union*Y
    itemset = tuple(sorted(set(X).union(set(Y))))
    
    # Caluculate and return the confidence
    return support_count(transactions, itemset) / support_count_X


def generate_association_rules(itemset):
    assoc_rules = []
    
    for X in powerset(itemset, min_len=1, max_len=len(itemset)-1):
        Y = tuple(sorted(set(itemset).difference(set(X))))
        assoc_rules.append((X, Y))
    
    return assoc_rules



def show_top_rules(transactions, min_support=0.0, min_confidence=0.0, k=5, sort='lift', reverse=True, id_map=None, rhs=None):

    _, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)

    rule_count = len(rules)

    # Do the filtering
    if rhs is not None:
        rules = [ r for r in rules if len(r.rhs) == 1 and r.rhs[0] == rhs ]

    rule_count_filtered = len(rules)    
    
    # Do the sorting
    if sort == 'support':
        rules = sorted(rules, key=lambda rule: rule.support, reverse=reverse)
    elif sort == 'confidence':
        rules = sorted(rules, key=lambda rule: rule.confidence, reverse=reverse)
    else:
        rules = sorted(rules, key=lambda rule: rule.lift, reverse=reverse)
    
    
    # Do the capping; we only want to see the top-k rules
    rules = rules[0:(k+1)]       
    
    # Do the showing
    #print('=== Total Number of Rules: {} | umber of rules with matching RHS: {} ==='.format(rule_count, rule_count_filtered))
    print('=== Total Number of Rules: {} ==='.format(rule_count))
    for r in rules:
        if id_map is not None:
            lhs, rhs = [ id_map[i] for i in r.lhs ], [ id_map[i] for i in r.rhs ]
        else:
            lhs, rhs = [ str(i) for i in r.lhs ], [ str(i) for i in r.rhs ]
        lhs, rhs = '; '.join(lhs), '; '.join(rhs)
        
        print('({}) => ({})  [s: {:.2f}, c: {:.2f}, l: {:.2f}]'.format(lhs, rhs, r.support, r.confidence, r.lift))
    print()
