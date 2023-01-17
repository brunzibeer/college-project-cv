import numpy as np
import sys, getopt

from scipy import spatial
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import minkowski
from os import system

DESCPATH = "placeholder/path"
LABPATH = "placeholder/path"

def retrieve(idx, descriptors, distance_type):
    '''
    Args:
        - idx: index of the query
        - descriptors: descriptors array
        - distance_type: distance measure to use for computation
    
    Returns:
        - distances: array of distances with respect to the query
    '''

    query = descriptors[idx] # Getting the query from the descriptor list
    distances = list()
    
    for desc in descriptors:
        # Skipping self-difference
        if np.array_equal(query, desc):
            continue

        # Computing the various distances
        if distance_type == "cosine":
            dist = spatial.distance.cosine(desc, query)

        elif distance_type == "euclidian":
            dist = np.linalg.norm(desc - query)
        
        elif distance_type == "earth mover":
            dist = wasserstein_distance(desc, query)
        
        elif distance_type == "minkowski":
            dist = minkowski(desc, query, 1)

        distances.append(dist)
    
    return distances

# Computing Precision at each rank
def compute_p(distances, labels, idx, rank=6):
    '''
    Args:
        - distances: distances array from the retrieve method
        - labels: labels array to check performance
        - idx: query index
        - rank: max rank for precision computation
    
    Returns:
        - precision_array: precision for each query
    '''

    sorted = np.argsort(distances)
    query_gt = labels[idx]
    tp = 0
    precision_array = list()

    for i, desc in enumerate(sorted):
        if tp == rank:
            break
        
        if labels[desc] == query_gt:
            tp += 1
            precision_array.append(tp/(i+1))
    
    precision_array = np.array(precision_array)
    
    return precision_array
    
# Computing Average Precision
def compute_ap(distances, labels, idx, scope=11):
    '''
    Args:
        - distances: distances array from the retrieve method
        - labels: labels array to check performance
        - idx: query index
        - scope: we look only at the first s scores

    Returns:
        - score: score for every query
    '''

    sorted = np.argsort(distances)[:scope]
    query_gt = labels[idx]
    numerator = 0
    counter = 1
    tp = 0
    score = 0

    for desc in sorted:
        if labels[desc] == query_gt:
            tp += 1
            numerator += tp/counter
        counter += 1
    
    if tp != 0:
        score = numerator / tp
    
    return score

# Computing R-Precision
def compute_rp(distances, labels, idx, relevant_obj=8):
    '''
    Args:
        - distances: distances array from the retrieve method
        - labels: labels array to check performance
        - idx: query index
        - relevant_obj: number of relevant objects we consider

    Returns:
        - rp: R-Precision score for each query
    '''

    sorted = np.argsort(distances)[:relevant_obj]
    query_gt = labels[idx]
    tp = 0

    for desc in sorted:
        if labels[desc] == query_gt:
            tp += 1
    
    rp = tp/relevant_obj

    return rp


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:r:o:")
    except getopt.GetoptError:
        print("Usage: retrieval.py [-s scope(int) | -r rank(int) | -o relevant_obj(int)]")
        sys.exit(2)
    
    for opt, val in opts:
        if opt == "-s":
            try:
                val = int(val)
                scope = val
            except ValueError:
                print("Error!! Scope must be int")
                sys.exit(1)
        
        if opt == "-r":
            try:
                val = int(val)
                rank = val
            except ValueError:
                print("Error!! Rank must be int")
                sys.exit(1)
        
        if opt == "-o":
            try:
                val = int(val)
                relevant_obj = val
            except ValueError:
                print("Error!! Relevant Obj must be int")
                sys.exit(1)

    # Loading labels and descriptors
    labels = np.load(LABPATH)
    descriptors = np.load(DESCPATH)

    # Types of distances
    test_distances = ["euclidian", "cosine", "earth mover", "minkowski"]
    np.random.seed(22)
    test_queries = np.unique(np.random.randint(labels.shape[0], size=labels.shape[0]//3))

    system('cls')
    print("===================================================================================")
    print("-------------------------------------RETRIEVAL-------------------------------------")
    print("===================================================================================\n")

    for d in test_distances:
        print(f"Testing with {d.capitalize()} Distance for {test_queries.shape[0]} Queries")
        average_precisions = list()
        average_p = list()
        average_rp = list()
        for idx in test_queries:
            computed_distances = retrieve(idx, descriptors, d)
            ap = compute_ap(computed_distances, labels, idx, scope=scope)
            p = compute_p(computed_distances, labels, idx, rank=rank)
            rp = compute_rp(computed_distances, labels, idx, relevant_obj=relevant_obj)

            average_precisions.append(ap)
            average_p.append(p)
            average_rp.append(rp)
        
        average_rp = np.stack(average_rp)
        
        # Computing the mean over all the tests
        mAP = np.mean(average_precisions)
        mP = np.mean(average_p, axis=0)
        mRP = np.mean(average_rp)
        print("\n-----------------------------------------------------------------------------------")
        print(f"\nMean Average Precision @ Scope {scope}: {round(mAP,3)}")
        print("\n-----------------------------------------------------------------------------------\n")
        for i in range(6):
            print(f"Average Precision @ Rank {(i+1)}/6: {round(mP[i],3)} ")
        print("\n-----------------------------------------------------------------------------------")
        print(f"\nAverage R-Precision with {relevant_obj} relevant objects: {round(mRP,3)}")
        print("\n===================================================================================\n")
    


    


