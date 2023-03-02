from utils import LoadOnlyLandmarks
import numpy as np
from TestAccuracy import search
import os
from icecream import ic


def compute_distances(landmark_dic):
    """Compute all distances between all landmarks for a single file"""
    distances = {}

    landmarks = landmark_dic.keys()

    for lm in landmarks:
        distances[lm] = {}
        for lm2 in landmarks:
            if lm != lm2:
                distances[lm][lm2] = np.linalg.norm(landmark_dic[lm] - landmark_dic[lm2])

    return distances

def compute_difference_distances(gold_dic, test_dic):
    """Compare the distances between all landmarks for two files in a dictionary"""
    gold = compute_distances(gold_dic)
    test = compute_distances(test_dic)

    differences = {}

    for lm in test.keys():
        differences[lm] = {}
        for lm2 in test[lm].keys():
            differences[lm][lm2] = abs(gold[lm][lm2] - test[lm][lm2])

    return differences

def compute_directions(landmark_dic):
    """Compute all directions between all landmarks for a single file"""
    directions = {}

    landmarks = landmark_dic.keys()

    for lm in landmarks:
        directions[lm] = {}
        for lm2 in landmarks:
            if lm != lm2:
                directions[lm][lm2] = landmark_dic[lm] - landmark_dic[lm2]

    return directions

def compute_difference_directions(gold_dic, test_dic):
    """Compare the angular differences between all landmarks for two files in a dictionary"""
    gold = compute_directions(gold_dic)
    test = compute_directions(test_dic)

    angular_diff = {}

    for lm in test.keys():
        angular_diff[lm] = {}
        for lm2 in test[lm].keys():
            angular_diff[lm][lm2] = np.arccos(np.dot(gold[lm][lm2], test[lm][lm2]) / (np.linalg.norm(gold[lm][lm2])* np.linalg.norm(test[lm][lm2])))

    return angular_diff

def get_count(differences, max_diff=1, min_diff=0):
    landmark_count = {}

    for lm in differences.keys():
        count = 0
        for lm2 in differences[lm].keys():
            if differences[lm][lm2] > max_diff:
                count += 1
            if differences[lm][lm2] < min_diff:
                count -= 1
        landmark_count[lm] = count

    return landmark_count

def sum_count(distance_count, direction_count):
    """Sum the number count of distance and direction"""
    sum_count = {}

    for lm in distance_count.keys():
        sum_count[lm] = distance_count[lm] + direction_count[lm]

    return sum_count

def get_removed_landmark(test_path,gold_path):
    """Get the list of landmark that should be removed from the landmark dictionary based on the difference with the gold standard"""

    test_ldmk = LoadOnlyLandmarks(test_path)
    gold_ldmk = LoadOnlyLandmarks(gold_path)

    dist_diff = compute_difference_distances(gold_ldmk ,test_ldmk)
    dir_diff = compute_difference_directions(gold_ldmk ,test_ldmk)

    dist_count = get_count(dist_diff,max_diff=15)
    dir_count = get_count(dir_diff,max_diff=0.4,min_diff=0.1)

    tot_count = sum_count(dist_count, dir_count)

    removed_landmarks = []
    for lm in tot_count.keys():
        if tot_count[lm] > len(test_ldmk):
            removed_landmarks.append(lm)

    return removed_landmarks
    
if __name__ == "__main__":

    gold_path = "/home/lucia/Desktop/Luc/DATA/ASO/TESTFILES/GOLD.mrk.json"
    test_folder_path = "/home/lucia/Desktop/Luc/DATA/ASO/ACCURACY/Maxillary/ALI_OUTPUT/"

    # main(gold_path, test_folder_path)

    gold_ldmk = LoadOnlyLandmarks("/home/lucia/Desktop/Luc/DATA/ASO/TESTFILES/GOLD.mrk.json")
    # gold_ldmk = LoadOnlyLandmarks('/home/lucia/Desktop/Luc/DATA/ASO/GOLD/Felicia/MAMP_0002_T1.mrk.json')
    test_json_files = search("/home/lucia/Desktop/Luc/DATA/ASO/ACCURACY/Maxillary/ALI_OUTPUT/","json")["json"]
    # test_json_files = search("/home/lucia/Desktop/Luc/DATA/ASO/ACCURACY/Head/Felicia_BAMP_ASO_OUTPUT/","json")["json"]
    
    for i in range(len(test_json_files)):
        # print(i)
        # i = 49
        name = os.path.basename(test_json_files[i]).split('_lm')[0] 
        # print(i,name)
        removed_landmark = get_removed_landmark(test_json_files[i],gold_path)

        if len(removed_landmark) > 0:
            print("{} | {} --> {} removed {} (len={})".format(i,name, len(removed_landmark), removed_landmark,len(LoadOnlyLandmarks(test_json_files[i]))))
        # break