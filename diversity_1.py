import numpy as np
import sklearn
import os
import pdb
import matplotlib.pyplot as plt
from scipy.stats import moment
import json
import math
CLASS_DICT = {
    0: 'cow',
    1: 'sheep',
    2: 'bird',
    3: 'person',
    4: 'cat',
    5: 'dog',
    6: 'horse',
    7: 'aeroplane',
    8: 'motorbike',
    9: 'bicycle',
    10: 'pottedplant',
}
colors = ['red', 'blue', 'green', 'orange']



class_dict_reverse = {v: k for k, v in CLASS_DICT.items()}

def to_one_hot(num, num_classes):
    """Convert a number to one-hot encoding"""
    one_hot = np.zeros(num_classes)
    one_hot[num] = 1
    return one_hot


def calculate_diversity(embeddings, metric='cosine'):
    """
    Calculate the diversity of a set of embeddings.

    Args:
        embeddings (np.ndarray): A 2D array where each row is an embedding.
        metric (str): The distance metric to use ('cosine' or 'euclidean').

    Returns:
        float: The average pairwise distance between embeddings.
    """
    if metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(embeddings)
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings)
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

    # Calculate the average distance, excluding self-distances
    np.fill_diagonal(distances, 0)
    avg_distance = np.mean(distances)

    return avg_distance


def get_x_y_overlap(head_bbx, torso_bbx):
    x1_min, x1_max, y1_min, y1_max = head_bbx
    x2_min, x2_max, y2_min, y2_max = torso_bbx

    # Compute overlap in x and y directions - only magnitudes
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    # Only return overlap if both axes have overlap
    if x_overlap > 0 and y_overlap > 0:
        return (x_overlap, y_overlap)
    else:
        print("No overlap")
        return 0, 0


import numpy as np


def get_row_theta(head_bbx, torso_bbx):
    """
    Calculate the magnitude and angle between head and torso bounding box centers.

    Parameters:
    - head_bbx: tuple (xmin, xmax, ymin, ymax) in pixel coordinates
    - torso_bbx: tuple (xmin, xmax, ymin, ymax) in pixel coordinates

    Returns:
    - magnitude: Euclidean distance between centers
    - angle_deg: angle (in degrees) from head to torso, wrt X-axis (top-left origin)
    """
    print("Calculating row theta for head and torso bounding boxes")

    # Unpack bounding boxes
    x1_min, x1_max, y1_min, y1_max = head_bbx
    x2_min, x2_max, y2_min, y2_max = torso_bbx

    # Midpoints
    head_x = (x1_min + x1_max) / 2
    head_y = (y1_min + y1_max) / 2
    torso_x = (x2_min + x2_max) / 2
    torso_y = (y2_min + y2_max) / 2

    # Vector from head to torso
    dx = torso_x - head_x
    dy = torso_y - head_y


    # Vector from head to torso
    # Magnitude of the vector
    magnitude = math.hypot(dx, dy)

    # Angle with respect to the horizontal (top-left origin, y increases downward)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return magnitude, angle_deg



    # canvas size is 660
    
def get_clusters(head_bbx, torso_bbx):
    x1_min, x1_max, y1_min, y1_max = head_bbx
    x2_min, x2_max, y2_min, y2_max = torso_bbx

    # Compute midpoints
    midpoint_head_x = (x1_min + x1_max) / 2
    midpoint_head_y = (y1_min + y1_max) / 2
    midpoint_torso_x = (x2_min + x2_max) / 2
    midpoint_torso_y = (y2_min + y2_max) / 2
    return midpoint_head_x, midpoint_head_y, midpoint_torso_x, midpoint_torso_y
    
    

# hardcoded for head and torso only

def coefficient_of_variation(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    return std / abs(mean) if mean != 0 else np.inf


def cluster(bounding_box):
    x1_min, x1_max, y1_min, y1_max = bounding_box


    # Compute midpoints
    midpoint_1_x = (x1_min + x1_max) / 2
    midpoint_1_y = (y1_min + y1_max) / 2
    return midpoint_1_x, midpoint_1_y

    
def compute_stats(points, n=3):
    """
    Computes standard deviation and n-th central moment for x and y coordinates.

    Args:
        points (list of tuples): List of (x, y) coordinates.
        n (int): Order of the central moment to compute (default is 3).

    Returns:
        dict: {
            'std_x': ..., 'std_y': ...,
            'moment_x': ..., 'moment_y': ...
        }
    """
    x_vals, y_vals = zip(*points)  # Unpack coordinates 
    x = np.array(x_vals)
    y = np.array(y_vals)
    

    return {
        'std_x': np.std(x),
        'std_y': np.std(y),
        'moment_x': moment(x, moment=n),
        'moment_y': moment(y, moment=n),
        'dispersion_x': coefficient_of_variation(x),
        'dispersion_y': coefficient_of_variation(y)
    }
    
    
cow_part_labels = {0: 'head', 1: 'left horn', 2: 'right horn', 3: 'torso', 4: 'neck', 5: 'left front upper leg', 6: 'left front lower leg', 7: 'right front upper leg', 8: 'right front lower leg', 9: 'left back upper leg', 10: 'left back lower leg', 11: 'right back upper leg', 12: 'right back lower leg', 13: 'tail'}

bird_part_labels = {0: 'head', 1: 'torso', 2: 'neck', 3: 'left wing', 4: 'right wing', 5: 'left leg', 6: 'left foot', 7: 'right leg', 8: 'right foot', 9: 'tail'}

cat_part_labels = {0: 'head', 1: 'torso', 2: 'neck', 3: 'left front leg', 4: 'left front paw', 5: 'right front leg', 6: 'right front paw', 7: 'left back leg', 8: 'left back paw', 9: 'right back leg', 10: 'right back paw', 11: 'tail'}

dog_part_labels = {0: 'head', 1: 'torso', 2: 'neck', 3: 'left front leg', 4: 'left front paw', 5: 'right front leg', 6: 'right front paw', 7: 'left back leg', 8: 'left back paw', 9: 'right back leg', 10: 'right back paw', 11: 'tail', 12: 'muzzle'} 


person_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'left lower arm': 3, 'left upper arm': 4,
    'left hand': 5, 'right lower arm': 6, 'right upper arm': 7,
    'right hand': 8, 'left lower leg': 9, 'left upper leg': 10,
    'left foot': 11, 'right lower leg': 12, 'right upper leg': 13,
    'right foot': 14
}

horse_part_labels = {
    'head': 0, 'left front hoof': 1, 'right front hoof': 2, 'torso': 3,
    'neck': 4, 'left front upper leg': 5, 'left front lower leg': 6,
    'right front upper leg': 7, 'right front lower leg': 8,
    'left back upper leg': 9, 'left back lower leg': 10,
    'right back upper leg': 11, 'right back lower leg': 12, 'tail': 13,
    'left back hoof': 14, 'right back hoof': 15}

aeroplane_part_labels = {0:'body', 1:'left wing', 2:'right wing', 3:'stern', 4:'tail'}
for i in range(0, 6):
    aeroplane_part_labels[5+i] = f'engine'
for i in range(0,8):
    aeroplane_part_labels[11+i] = f"wheel"


bicycle_part_labels = {0:'body', 1:'front wheel', 2:'back wheel', 3:'chainwheel', 4:'handlebar', 5:'saddle', 6:'headlight'}

motorbike_part_labels = {0:'body', 1:'front wheel', 2:'back wheel', 3:'saddle', 4:'handlebar', 5:'headlight', 6:'headlight', 7:'headlight'}

pottedplant_part_labels = {0:'body', 1:'pot', 2:'plant'}


import sys
sys.path.append("/home/saksham.gupta/inference/Palgo_pipeline/Palgo-main/src/ImagegenDataProcessing/utils/")
data_root ="/ssd_scratch/saksham.gupta/data/"
#import data set matrices
import constants
CANVAS_SIZE = 660


path = "/ssd_scratch/saksham.gupta/data_playgen/class_v_test_combined_mask_data.np"
path_1 = "/ssd_scratch/saksham.gupta/data_playgen/layout_pred.npy"

import yaml
 # either tak frm yaml or termianl input (argparse)
with open("/home/saksham.gupta/inference/diversity/config_diversity.yml", "r") as f:
    config = yaml.safe_load(f)
    
SPLIT_ARRAY= []
bruh_1 = ['train', 'test', 'val', 'playgen']
#bruh_1 = ['train']

#  I want a per plit, per class list of dictionaries (2)
for variable_1 in bruh_1:
    plt.clf() # top is from writing the points ion teh same canvas
    
    if variable_1 == 'train' or variable_1 == 'test' or variable_1 == 'val':
        
        x_train_path = f"X_{variable_1}_combined_mask_data.np"
        obj_class_train_path = f"class_v_{variable_1}_combined_mask_data.np"
        images_train_path = f"img_{variable_1}_combined_mask_data.np"
        x_train = np.load(os.path.join(data_root, x_train_path), allow_pickle=True)
        obj_class_train = np.load(os.path.join(data_root, obj_class_train_path),allow_pickle=True)
        images_train = np.load(os.path.join(data_root, images_train_path), allow_pickle=True)

    # Maybe consider test and train combined too
    elif variable_1 == 'playgen':
        x_train = np.load(path_1, allow_pickle=True)
        obj_class_train = np.load(path, allow_pickle=True)



    # print(x_train.shape)
    # print(x_train[0, :, 1:].shape)
    train_bbxs = [x_train]

    x_train[:, :, 1:] *=CANVAS_SIZE # except where the part checker

    # print(x_train[0, :, 1:])

    # print(to_one_hot(1, 10))
    # only cow

    #input ='bird'
    bruh=['cow', 'bird', 'cat', 'dog','sheep'] # for playgen
    # some problem with horse and person
    # aeroplane, motorbike, bicycle 
    # no human and horse for now
    overlap_distribution_ht_class=[]
    geometry_distribution_ht_class = []
    cluster_list_new=[]
    # for class
    CLASS_ARRAY = []
    
    for input in bruh: #only iterating thorugh cow for now
        print(input)
        
        label = class_dict_reverse.get(input)
        # For playgen

        if variable_1 == 'playgen':
            one_hot_vector = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 7))]
        # for Test,Train,Val
        else:
            one_hot_vector = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 11)) ]

        cow_part_labels_reverse = {v: k for k, v in cow_part_labels.items()}
        bird_part_labels_reverse = {v: k for k, v in bird_part_labels.items()}
        cat_part_labels_reverse = {v: k for k, v in cat_part_labels.items()}    
        dog_part_labels_reverse = {v: k for k, v in dog_part_labels.items()}
        horse_part_labels_reverse = {v: k for k, v in horse_part_labels.items()}
        sheep_part_labels_reverse = {v: k for k, v in cow_part_labels.items()}
        person_part_labels_reverse = {v: k for k, v in person_part_labels.items()}

        if input == 'cow':
            part_labels_reverse = cow_part_labels_reverse
        elif input == 'bird':
            part_labels_reverse = bird_part_labels_reverse
        elif input == 'cat':
            part_labels_reverse = cat_part_labels_reverse
        elif input == 'dog':
            part_labels_reverse = dog_part_labels_reverse
        elif input == 'horse':
            part_labels_reverse = horse_part_labels_reverse
        elif input == 'sheep':
            part_labels_reverse = cow_part_labels_reverse
        elif input == 'person':
            part_labels_reverse = person_part_labels_reverse
        elif input == 'aeroplane':
            part_labels_reverse = aeroplane_part_labels
        elif input == 'motorbike':
            part_labels_reverse = motorbike_part_labels
        elif input == 'bicycle':
            part_labels_reverse = bicycle_part_labels


        #parts =['head','torso']
        # threw an error here, when I changed from torso to body
        
        #Parts Hyperparameter
        
        # Combinations for cow - upper leg
        # parts = ['torso', 'right front upper leg'] # for cow
        # parts= ['torso', 'left front upper leg'] # for cow
        # parts= ['torso', 'right back upper leg'] # for cow
        # parts= ['torso', 'left back upper leg'] # for cow
        # parts= ['head', 'torso'] # for cow
        
        # For some classes,
        # cow , dog, horse, sheep, bird, cat, person
        parts= ['head', 'torso'] # for cow
        
        
        # Only works for cow, bird, cat, dog,  sheep,
        # doesn't work for  person, horse
        
        row=[]

        for part in parts:
            row.append(part_labels_reverse.get(part))
            
        print("Row:", row)

        if variable_1 == 'playgen':
            all_rows_except= np.array([i for i in range(16) if i!= row[0] and i!= row[1]])
        else:
            all_rows_except = np.array([i for i in range(18) if i!= row[0] and i!= row[1]])

        print("All rows except:", all_rows_except)


      # this is for playgen inanimate data

        x_train_1 = x_train[one_hot_vector]
        #isolating cases where there are only head and torso
        
        # images/ layouts where only these parts are present
        #  Playgen
        
        # You don't pass x_train in this? 
        # if config['parts'] == 'limited':
        #     L = []
        #     for i in range(len(x_train)):
        #         shape_like = x_train[i][0]  # Pick any part to get the correct shape
        #         Zero = np.zeros_like(shape_like)
        #         true_counter = True  

        #         for j in all_rows_except:
        #             if not np.all(x_train[i][j] == Zero):
        #                 true_counter = False
        #                 break  # No need to check further if one is non-zero

        #         if true_counter:
        #             L.append(i)
        #         else:
        #             print(f"Example {i}: No cases found with only head and torso bounding boxes.")


        #     x_train_cow = x_train[L]  # Filter to only those with head and torso

                
        # Bounding boxes for head and torso
        
        bounding_box_1 = {'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}
        bounding_box_2 = {'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}

        actually_present =[] # list of indices present
        for j,i in enumerate(x_train_1):
            # print("This the bounding box for head and right front upper leg",i[row[0]],i[row[1]])
            part_present_1,_,_,_,_=i[row[0]]
            part_present_2,_,_,_,_=i[row[1]]

            if part_present_1 == 0 or part_present_2 == 0:
             #   print("one of the is not present, skipping")
                continue
            
            part_present_1,xmin,ymin,xmax,ymax=i[row[0]]
            # Based on the parts - you choose which dictionary to append it to
            bounding_box_1['xmin'].append(xmin)
            bounding_box_1['xmax'].append(xmax)
            bounding_box_1['ymin'].append(ymin)
            bounding_box_1['ymax'].append(ymax)

            part_present_2,xmin,ymin,xmax,ymax=i[row[1]]
            bounding_box_2['xmin'].append(xmin)
            bounding_box_2['xmax'].append(xmax)
            bounding_box_2['ymin'].append(ymin)
            bounding_box_2['ymax'].append(ymax)
            
        # train split - cow - normal
            actually_present.append(j)
            
        interm_tuple = (bounding_box_1, bounding_box_2) 
        CLASS_ARRAY.append(interm_tuple)
        print("Class array:", len(CLASS_ARRAY))

    SPLIT_ARRAY.append(CLASS_ARRAY)
    print("Split array:", len(SPLIT_ARRAY))

      
# The below fucntion uses 
def combination_function(split,object_class, parts_array, configuration):
    # Making all individual graphs first , then assessing permutations and combinations
    # WILL add parts array eventually in the fucntion
    # Across all instance in the split, for a given object class, get the bounding boxes of the head and torso
    # Access all is the above froma global array
    # # configurations : overlap geometry cluster - corresponding graphs and dispersion metrics
    # if split == 'train':
    #     print("Train")
    #     # consider SPLIT_ARRAY[0]
    # if object_class=='cow':
    #     print("Cow")
    #     # consider SPLIT_ARRAY[0][0] # just get teh index from the bruh_1 array
    #     # If configuration is overlap
    Array = SPLIT_ARRAY[0][0]  # Assuming you want the first class in the first split
    if configuration == 'overlap':
        # One graph
        # instead of a tuple make a dictionary
        overlap=[]
        #  Just two dicts at a time - LATER
        dict_1 =Array[0] # - PART -1
        dict_2 =Array[1] # - PART -2
        for i in range(len(dict_1['xmin'])):
            overlap.append(get_x_y_overlap((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i]),(dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
        #plot graph:
        overlap = [pt for pt in overlap if pt != (0, 0)]  # Filter out zero overlaps
        colors = ['red', 'blue', 'green', 'orange']
            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in overlap]
        y_1_vals = [pt[1] for pt in overlap]
        plt.scatter(x_1_vals, y_1_vals, color=colors[0], label=bruh[0])
        plt.xlabel("X_overlap")
        plt.ylabel("Y_overlap")
        plt.title("Overlap Plot of Four Classes")
        plt.legend()
        plt.grid(True)
        # for individual plots -add p
        plt.savefig(f"/home/saksham.gupta/inference/diversity/overlap_1.png")
    

    if configuration == 'geometric':
        geometry=[]
        # one graph
        #  Just two dicts at a time - LATER
        dict_1 =Array[0] # - PART -1
        dict_2 =Array[1] # - PART -2
        for i in range(len(dict_1['xmin'])):
            geometry.append(get_row_theta((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i]),(dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
        #plot graph:
        colors = ['red', 'blue', 'green', 'orange']
            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in geometry]
        y_1_vals = [pt[1] for pt in geometry]
        plt.scatter(x_1_vals, y_1_vals, color=colors[0], label=bruh[0])
        plt.xlabel("Angle")
        plt.ylabel("Magnitude")
        plt.title("Geometry Plot of Four Classes")
        plt.legend()
        plt.grid(True)
        # for individual plots -add p
        plt.savefig(f"/home/saksham.gupta/inference/diversity/geometry_1.png")
        
    if configuration == 'midpoints':
        # Two graphs
        cluster_list_1 = []
        cluster_list_2 = []
        # one graph
        #  Just two dicts at a time - LATER
        dict_1 =Array[0] # - PART -1
        dict_2 =Array[1] # - PART -2
        plt.clf()
        
        for i in range(len(dict_1['xmin'])):

            # Only takes one seth of bounding box coordinates
            cluster_list_1.append(cluster((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i])))
        #plot graph:
            cluster_list_2.append(cluster((dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
            
        plt.clf()
          
        colors = ['red', 'blue', 'green', 'orange']
            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in cluster_list_1]
        y_1_vals = [pt[1] for pt in cluster_list_1]
        pdb.set_trace()
        
        plt.scatter(y_1_vals, x_1_vals, color=colors[0], label=bruh[0])
        plt.xlabel("Angle")
        plt.ylabel("Magnitude")
        plt.title("Geometry Plot of Four Classes")
        plt.legend()
        plt.grid(True)
        # for individual plots -add p
        plt.savefig(f"/home/saksham.gupta/inference/diversity/cluster_1.png")
        plt.clf()
            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in cluster_list_2]
        y_1_vals = [pt[1] for pt in cluster_list_2]
        plt.scatter(x_1_vals, y_1_vals, color=colors[0], label=bruh[0])
        plt.xlabel("Angle")
        plt.ylabel("Magnitude")
        plt.title("Geometry Plot of Four Classes")
        plt.legend()
        plt.grid(True)
        # for individual plots -add p
        plt.savefig(f"/home/saksham.gupta/inference/diversity/cluster_2.png")
            
    
    
    
    
        

        
        

combination_function('train', 'cow', ['head', 'torso'], 'midpoints')
        