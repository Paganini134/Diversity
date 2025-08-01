# /home/saksham.gupta/inference/Olaf_dataset/matrices/splits/class_v_train_combined_mask_data.np


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


parts_array= [
    ['head', 'torso'],  
    ['torso', 'right front upper leg'],  # cow
    ['torso', 'left front upper leg'],   # cow
    ['torso', 'right back upper leg'],   # cow
    ['torso', 'left back upper leg'],    # cow
]
class_dict_reverse = {v: k for k, v in CLASS_DICT.items()}

def to_one_hot(num, num_classes):
    """Convert a number to one-hot encoding"""
    one_hot = np.zeros(num_classes)
    one_hot[num] = 1
    return one_hot


def get_x_y_overlap(head_bbx, torso_bbx):
    x1_min, x1_max, y1_min, y1_max = head_bbx
    x2_min, x2_max, y2_min, y2_max = torso_bbx

    # Compute overlap in x and y directions - only magnitudes
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    
    
    head_bbx = (x1_min, x1_max, y1_min, y1_max)
    torso_bbx = (x2_min, x2_max, y2_min, y2_max)
    

    # Only return overlap if both axes have overlap
    if x_overlap > 0 and y_overlap > 0:
        return (x_overlap, y_overlap)
    elif is_fully_inside(head_bbx, torso_bbx) or is_fully_inside(torso_bbx, head_bbx):
        return 0,0
    else:
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

def is_fully_inside(inner, outer):
    x1_min, x1_max, y1_min, y1_max = inner
    x2_min, x2_max, y2_min, y2_max = outer

    return (x1_min >= x2_min and x1_max <= x2_max and
            y1_min >= y2_min and y1_max <= y2_max)

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
# data_root ="/ssd_scratch/saksham.gupta/data/"
data_root = "/archive/projects/palgo/syn_data_animate/"
#import data set matrices
import constants
CANVAS_SIZE = 660


path = "/ssd_scratch/saksham.gupta/data_playgen/class_v_test_combined_mask_data.np"
path_1 = "/ssd_scratch/saksham.gupta/data_playgen/layout_pred.npy"

import yaml
 # either tak frm yaml or termianl input (argparse)
with open("/home/saksham.gupta/inference/diversity/config_diversity.yml", "r") as f:
    config = yaml.safe_load(f)
    
bruh_1 = ['train', 'test', 'val'] # 18,5 format
#bruh_1 = ['train']
NET_ARRAY=[]
for variable_2 in range(0,5):
    SPLIT_ARRAY= []
    
#  I want a per split, per class list of dictionaries (2)
    for variable_1 in bruh_1:
        plt.clf() # top is from writing the points ion teh same canvas
        
        if variable_1 == 'train' or variable_1 == 'test' or variable_1 == 'val':
            # /home/saksham.gupta/inference/Olaf_dataset/matrices/splits/class_v_train_combined_mask_data.np

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
        print(x_train.shape)

        #input ='bird'
        # horse?
        bruh=['cow', 'bird', 'cat', 'dog','sheep','horse','person'] # for playgen
        # bruh =['cow']
        # some problem with horse and person
        # aeroplane, motorbike, bicycle 
        # no human and horse for now

        # for class
        CLASS_ARRAY = []
        
        for input in bruh: #only iterating thorugh cow for now
            print(input)
            
            label = class_dict_reverse.get(input)
            # For playgen
            print(obj_class_train.shape)

            if variable_1 == 'playgen':
                one_hot_vector = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 7))]
            # for Test,Train,Val
            else:
                one_hot_vector = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 10)) ]

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
                part_labels_reverse = horse_part_labels
            elif input == 'sheep':
                part_labels_reverse = cow_part_labels_reverse
            elif input == 'person':
                part_labels_reverse = person_part_labels
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
            if input == 'cow' or input == 'sheep':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'right front upper leg'],
                    ['torso', 'left front upper leg'],
                    ['torso', 'right back upper leg'],
                    ['torso', 'left back upper leg']
                ]
                parts = part_array[variable_2]

            elif input == 'bird':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'right wing'],
                    ['torso', 'left wing'],
                    ['torso', 'right leg'],
                    ['torso', 'left leg']
                ]
                parts = part_array[variable_2]

            elif input == 'cat' or input == 'dog':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'left front leg'],
                    ['torso', 'right front leg'],
                    ['torso', 'left back leg'],
                    ['torso', 'right back leg']
                ]
                parts = part_array[variable_2]

            elif input == 'horse':

                part_array = [
                    ['head', 'torso'],
                    ['torso', 'left front upper leg'],
                    ['torso', 'right front upper leg'],
                    ['torso', 'left back upper leg'],
                    ['torso', 'right back upper leg']
                ]
                parts = part_array[variable_2]

            elif input == 'person':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'left upper arm'],
                    ['torso', 'right upper arm'],
                    ['torso', 'left lower leg'],
                    ['torso', 'right lower leg']
                ]
                parts = part_array[variable_2]

            
            
            # Only works for cow, bird, cat, dog,  sheep,
            # doesn't work for  person, horse
            
            row=[]

            for part in parts:
                row.append(part_labels_reverse.get(part))
                

            # Only for the limimted parts generation
            if variable_1 == 'playgen':
                all_rows_except= np.array([i for i in range(16) if i!= row[0] and i!= row[1]])
            else:
                all_rows_except = np.array([i for i in range(18) if i!= row[0] and i!= row[1]])


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
            print(f"parts_actually present {variable_2} {variable_1} {input} {len(actually_present)}")
            print("Class array:", len(CLASS_ARRAY))


        SPLIT_ARRAY.append(CLASS_ARRAY)
        print("Split array:", len(SPLIT_ARRAY))
    NET_ARRAY.append(SPLIT_ARRAY)
 

  
# The below function uses
# 
# s 
def combination_function(split,object_class, parts_array_number, configuration):
    
    arr = NET_ARRAY[parts_array_number]
    
    if split =='train':
        Arr_1 = arr[0]
    if split == 'playgen':
        Arr_1 = arr[1]
    
    arr_1 = Arr_1[bruh.index(object_class)]
    
    if configuration == 'overlap':
        # One graph
        # instead of a tuple make a dictionary
        overlap=[]
        plt.clf()
        #  Just two dicts at a time - LATER
        dict_1 =arr_1[0] # - PART -1
        dict_2 =arr_1[1] # - PART -2
        for i in range(len(dict_1['xmin'])):
            overlap.append(get_x_y_overlap((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i]),(dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
            
            head_bbx = (dict_1['xmin'][i], dict_1['xmax'][i], dict_1['ymin'][i], dict_1['ymax'][i])
            torso_bbx = (dict_2['xmin'][i], dict_2['xmax'][i], dict_2['ymin'][i], dict_2['ymax'][i])
                    

                
        #plot graph:
        overlap = [pt for pt in overlap if pt != (0, 0)]  # Filter out zero overlaps
            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in overlap]
        y_1_vals = [pt[1] for pt in overlap]
        # checking complete overap cases
        

        plt.figure(figsize=(6, 6))
        plt.scatter(y_1_vals, x_1_vals, color='red', label='Cluster Points')  # Note: x and y swapped

        plt.xlabel("X (increasing →)")
        plt.ylabel("Y (increasing ↓)")

        # Add title with point count
        num_points = len(dict_1['xmin'])
        plt.title(f"Overlap plot - {num_points} points")

        os.makedirs('/home/saksham.gupta/inference/diversity_olaf/overlap', exist_ok=True)
        filename = f"/home/saksham.gupta/inference/diversity_olaf/overlap/overlap_{split}_{object_class}_{parts_array[parts_array_number]}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plt.clf()

    if configuration == 'geometric':
        geometry=[]
        # one graph
        #  Just two dicts at a time - LATER
        dict_1 =arr_1[0] # - PART -1
        dict_2 =arr_1[1] # - PART -2
        for i in range(len(dict_1['xmin'])):
            geometry.append(get_row_theta((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i]),(dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
        #plot graph:

            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in geometry]
        y_1_vals = [pt[1] for pt in geometry]
        
        
        plt.figure(figsize=(6, 6))
        plt.scatter(x_1_vals, y_1_vals, color='red', label='Geometric Points')  # Note: x and y swapped

        
        plt.xlabel("X ")
        plt.ylabel("Y ")

        
        
        
        num_points = len(dict_1['xmin'])
 
        plt.title(f"Geometric Plot - {num_points} points")

        # Save figure
        os.makedirs('/home/saksham.gupta/inference/diversity_olaf/geometric', exist_ok=True)
        
        filename = f"/home/saksham.gupta/inference/diversity_olaf/geometric/geometric_{split}_{object_class}_{parts_array[parts_array_number]}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        
    if configuration == 'cluster':
        # Two graphs
        cluster_list_1 = []
        cluster_list_2 = []
        # one graph
        #  Just two dicts at a time - LATER
        print(type(arr_1))
        print(len(arr_1))
        dict_1 =arr_1[0] # - PART -1
        dict_2 =arr_1[1] # - PART -2
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
        plt.figure(figsize=(6, 6))
        plt.scatter(x_1_vals, y_1_vals, color='red', label='Cluster Points')  # Note: x and y swapped

        # Invert axes to match desired orientation
        plt.gca().invert_yaxis()  # Makes x increase down
        plt.gca().xaxis.tick_top()  # Moves x-axis to top
        plt.gca().xaxis.set_label_position('top')
        plt.xlabel("Y (increasing →)")
        plt.ylabel("X (increasing ↓)")

        # Add title with point count
        num_points = len(dict_1['xmin'])
        plt.title(f"Cluster plot - {num_points} points")

        # Save figure
        filename = f"/home/saksham.gupta/inference/diversity_olaf/cluster_{split}_{object_class}_{parts_array[parts_array_number]}_0.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plt.clf()
        
        
            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in cluster_list_2]
        y_1_vals = [pt[1] for pt in cluster_list_2]
        plt.figure(figsize=(6, 6))
        plt.scatter(x_1_vals, y_1_vals, color='red', label='Cluster Points')  # Note: x and y swapped
        # Invert axes to match desired orientation
        plt.gca().invert_yaxis()  # Makes x increase down
        plt.gca().xaxis.tick_top()  # Moves x-axis to top
        plt.gca().xaxis.set_label_position('top')
        plt.xlabel("Y (increasing →)")
        plt.ylabel("X (increasing ↓)")

        # Add title with point count
        num_points = len(dict_1['xmin'])
        plt.title(f"Cluster plot - {num_points} points")

        # Save figure
        os.makedirs('/home/saksham.gupta/inference/diversity_olaf/cluster', exist_ok=True)
        
        filename = f"/home/saksham.gupta/inference/diversity_olaf/cluster/cluster_{split}_{object_class}_{parts_array[parts_array_number]}_1.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plt.clf()
                    
    
    
    
    
        

        # Cluster
# for i in ['train','playgen']:
#     for j in ['cow', 'bird', 'cat', 'dog','sheep', 'person', 'horse',]:
#         for k in range(0,5):
#             combination_function(i, j, k, 'cluster')

        #  Overlap - pull up the correpsonfing bounding box images too
        # check for the containement cases for the torso and upper legs
print("="*90)
for i in ['train','playgen']:
    for j in ['cow', 'bird', 'cat', 'dog','sheep', 'person', 'horse',]:
        for k in range(0,5):
            print(f"Running overlap for {i}, {j}, {k}")
            combination_function(i, j, k, 'overlap')


# for i in ['train','playgen']:
#     for j in ['cow', 'bird', 'cat', 'dog','sheep', 'person', 'horse',]:
#         for k in range(0,5):
#             combination_function(i, j, k, 'geometric')

        
        
        


# Code for clustering

# what kind of input does it expect







# tranining an autoencoder to capture diversity
# So far we have been VaeS for inference- but now the data set is all the bounding boxes and the grounf trith is the divesity.

# can a combination of autoencoders be used
# some latent space representation of the bounding boxes
# graphs and latent space representations are weird

import pandas as pd

rows = []


