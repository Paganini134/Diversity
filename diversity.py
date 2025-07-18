import numpy as np
import sklearn
import os
import pdb
import matplotlib.pyplot as plt
from scipy.stats import moment
import json

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

    # Compute overlap in x and y directions
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    # Only return overlap if both axes have overlap
    if x_overlap > 0 and y_overlap > 0:
        return (x_overlap, y_overlap)
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

    # Compute midpoints
    midpoint_head_x = (x1_min + x1_max) / 2
    midpoint_head_y = (y1_min + y1_max) / 2
    midpoint_torso_x = (x2_min + x2_max) / 2
    midpoint_torso_y = (y2_min + y2_max) / 2

    # Vector from head to torso
    dx = midpoint_torso_x - midpoint_head_x
    dy = midpoint_torso_y - midpoint_head_y

    # Magnitude of the vector
    magnitude = np.sqrt(dx**2 + dy**2)

    # Angle relative to X-axis (OpenCV uses top-left origin)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360  # Normalize to [0, 360)

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

import sys
sys.path.append("/home/saksham.gupta/inference/Palgo_pipeline/Palgo-main/src/ImagegenDataProcessing/utils/")
data_root ="/ssd_scratch/saksham.gupta/data/"
#import data set matrices
import constants
CANVAS_SIZE = 660


dispersion_dict ={
    'cow': {
        'disp_x': None,
        'disp_y': None,
    },
    'cat': {
        'disp_x': None,
        'disp_y': None,
    },
    'dog': {
        'disp_x': None,
        'disp_y': None,
    },
    'bird': {
        'disp_x': None,
        'disp_y': None,
    }
}


# playgen paths
path = "/ssd_scratch/saksham.gupta/data_playgen/class_v_test_combined_mask_data.np"
path_1 = "/ssd_scratch/saksham.gupta/data_playgen/layout_pred.npy"


    
import yaml
 # either tak frm yaml or termianl input (argparse)
with open("/home/saksham.gupta/inference/diversity/config_diversity.yml", "r") as f:
    config = yaml.safe_load(f)
    
list_1=[] # list of list of tuples for cow
list_2 = [] # list of list of tuples for cow
bruh_1 = ['train', 'test', 'val', 'playgen']
#bruh_1 = ['train']
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
    bruh=['cow', 'bird', 'cat', 'dog','horse','sheep','person','aeroplane', 'motorbike', 'bicycle'] # for playgen
    # aeroplane, motorbike, bicycle 
    # no human and horse for now
    overlap_distribution_ht_class=[]
    geometry_distribution_ht_class = []
    cluster_list_new=[]

    for input in bruh: #only iterating thorugh cow for now
        label = class_dict_reverse.get(input)
        # For playgen
        if variable_1 == 'playgen':
            one_hot_cow = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 7))]
        # for Test,Train,Val
        else:
            one_hot_cow = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 11)) ]

        cow_part_labels_reverse = {v: k for k, v in cow_part_labels.items()}
        bird_part_labels_reverse = {v: k for k, v in bird_part_labels.items()}
        cat_part_labels_reverse = {v: k for k, v in cat_part_labels.items()}    
        dog_part_labels_reverse = {v: k for k, v in dog_part_labels.items()}

        if input == 'cow':
            part_labels_reverse = cow_part_labels_reverse
        elif input == 'bird':
            part_labels_reverse = bird_part_labels_reverse
        elif input == 'cat':
            part_labels_reverse = cat_part_labels_reverse
        elif input == 'dog':
            part_labels_reverse = dog_part_labels_reverse


        parts =['head','torso']
        #parts= ['left front upper leg', 'left front lower leg', 'right front upper leg','right back upper leg', 'right back lower leg','torso']

        row=[]

        for part in parts:
            row.append(part_labels_reverse.get(part))


        #isolating cases where there are only head and torso
        if variable_1 == 'playgen':
            all_rows_except= np.array([i for i in range(16) if i!= row[0] and i!= row[1]])
        else:
            all_rows_except = np.array([i for i in range(18) if i!= row[0] and i!= row[1]])
        
      # this is for playgen inanimate data

        x_train_cow = x_train[one_hot_cow]
        
        # images/ layouts where only these parts are present
        #  Playgen
        
        # You don't pass x_train in this? 
        if config['parts'] == 'limited':
            L = []
            for i in range(len(x_train)):
                shape_like = x_train[i][0]  # Pick any part to get the correct shape
                Zero = np.zeros_like(shape_like)
                true_counter = True  

                for j in all_rows_except:
                    if not np.all(x_train[i][j] == Zero):
                        true_counter = False
                        break  # No need to check further if one is non-zero

                if true_counter:
                    L.append(i)
                else:
                    print(f"Example {i}: No cases found with only head and torso bounding boxes.")


            x_train_cow = x_train[L]  # Filter to only those with head and torso

                
        #parse through x_train_cow 
        # hardcoded
        # dictionary of lists for bounding boxes
        bounding_box_head = {
            'xmin': [],
            'xmax': [],
            'ymin': [],
            'ymax': []
        }

        bounding_box_torso = {
            'xmin': [],
            'xmax': [],
            'ymin': [],
            'ymax': []
        }

        bounding_box_upper_leg = {
            'xmin': [],
            'xmax': [],
            'ymin': [],
            'ymax': []
        }

        bounding_box_lower_leg = {
            'xmin': [],
            'xmax': [],
            'ymin': [],
            'ymax': []
        }


        for i in x_train_cow:

            _,xmin,ymin,xmax,ymax=i[row[0]]
            bounding_box_head['xmin'].append(xmin)
            bounding_box_head['xmax'].append(xmax)
            bounding_box_head['ymin'].append(ymin)
            bounding_box_head['ymax'].append(ymax)
            
            _,xmin,ymin,xmax,ymax=i[row[1]]
            bounding_box_torso['xmin'].append(xmin)
            bounding_box_torso['xmax'].append(xmax)
            bounding_box_torso['ymin'].append(ymin)
            bounding_box_torso['ymax'].append(ymax)
            
            
            
        # print(f"Bounding box for head: {bounding_box_head}")    
        # print(f"Bounding box for torso: {bounding_box_torso}")    

        #parse throught the dictionary
        overlap_distribution_ht = []
        for i in range(len(bounding_box_head['xmin'])):
            # (x,y) tuple
            overlap_distribution_ht.append(get_x_y_overlap((bounding_box_head['xmin'][i],bounding_box_head['xmax'][i],bounding_box_head['ymin'][i],bounding_box_head['ymax'][i]),(bounding_box_torso['xmin'][i],bounding_box_torso['xmax'][i],bounding_box_torso['ymin'][i],bounding_box_torso['ymax'][i])))

            
            # print(((bounding_box_head['xmin'][i],bounding_box_head['xmax'][i],bounding_box_head['ymin'][i],bounding_box_head['ymax'][i]),(bounding_box_torso['xmin'][i],bounding_box_torso['xmax'][i],bounding_box_torso['ymin'][i],bounding_box_torso['ymax'][i])))
            
        print("X,y Overlap distirbution of the head and torso bounding boxes")

        
        
        geometry_list=[]
        cluster_list = []
        
        for i in range(len(bounding_box_head['xmin'])):
            # (x,y) tuple
            geometry_list.append(get_row_theta((bounding_box_head['xmin'][i],bounding_box_head['xmax'][i],bounding_box_head['ymin'][i],bounding_box_head['ymax'][i]),(bounding_box_torso['xmin'][i],bounding_box_torso['xmax'][i],bounding_box_torso['ymin'][i],bounding_box_torso['ymax'][i])))
            
            
        for i in range(len(bounding_box_head['xmin'])):
    # (x,y) tuple
            cluster_list.append(get_clusters((bounding_box_head['xmin'][i],bounding_box_head['xmax'][i],bounding_box_head['ymin'][i],bounding_box_head['ymax'][i]),(bounding_box_torso['xmin'][i],bounding_box_torso['xmax'][i],bounding_box_torso['ymin'][i],bounding_box_torso['ymax'][i])))


        # x,y overlap tupples between both bounding boxes of each instance
        # plot for head
        #bounding_box_head
        computed_states = compute_stats(overlap_distribution_ht, n=3)
        # Overlap of the head and torso bounding boxes out of all the ma
        # ts
        print("Computed statistics for overlap distribution:")
        # print(f"Standard Deviation X: {computed_states['std_x']}")
        # print(f"Standard Deviation Y: {computed_states['std_y']}")
        # print(f"3rd Moment X: {computed_states['moment_x']}")   
        # print(f"3rd Moment Y: {computed_states['moment_y']}")
        print(f"Dispersion X: {computed_states['dispersion_x']}")
        print(f"Dispersion Y: {computed_states['dispersion_y']}")
        # append to dict
        dispersion_dict[input]['disp_x'] = computed_states['dispersion_x']
        dispersion_dict[input]['disp_y'] = computed_states['dispersion_y']

        
        # save to a json
        # further looking at clusters from distributions
        overlap_distribution_ht_class.append(overlap_distribution_ht)
        geometry_distribution_ht_class.append(geometry_list)# Append the overlap distribution for this
        cluster_list_new.append(cluster_list) # Append the cluster list for this class
        list_1.append(overlap_distribution_ht)  # Append the overlap distribution for this class 
        list_2.append(geometry_list)

    # class
    # if you run for all 4 this will be a list of four elements
    # print("this is the dispersion dict")
    # print(dispersion_dict)
    # print("Overlap distribution of head and torso bounding boxes:")

    # print(overlap_distribution_ht_class)
    # # plotting the overlaps for different classes:
    colors = ['red', 'blue', 'green', 'orange']
        # for some reason it makes individual graphs for each class as well - got it
    for i in range(len(bruh)):
        # this loop runs per class
        x_1_vals = [pt[0] for pt in overlap_distribution_ht_class[i]]
        y_1_vals = [pt[1] for pt in overlap_distribution_ht_class[i]]
        plt.scatter(x_1_vals, y_1_vals, color=colors[i], label=bruh[i])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot of Four Classes")
    plt.legend()
    plt.grid(True)
    # for individual plots -add I
    plt.savefig(f"/home/saksham.gupta/inference/diversity/overlap_distribution_{variable_1}_{config['parts']}.png")
    
    plt.clf() # top is from writing the points ion teh same canvas
    

    for i in range(len(bruh)):
        x_vals = [pt[0] for pt in geometry_distribution_ht_class[i]]
        y_vals = [pt[1] for pt in geometry_distribution_ht_class[i]]
        plt.scatter(x_vals, y_vals, color=colors[i], label=bruh[i])

    plt.xlabel("Magnitude")
    plt.ylabel("Angle ")
    plt.title("Scatter Plot of Four Classes")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/saksham.gupta/inference/diversity/geometric_distribution_{variable_1}_{config['parts']}.png") 
    

    plt.clf() # top is from writing the points ion teh same canvas
    
    # how does cluster_list_new for all classes?

    for i in range(len(bruh)):
        x_vals = [pt[0] for pt in cluster_list_new[i]]
        y_vals = [pt[1] for pt in cluster_list_new[i]]
        plt.scatter(x_vals, y_vals, color=colors[i], label=bruh[i])

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot of Four Classes")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert Y-axis to move origin to top-left
    plt.savefig(f"/home/saksham.gupta/inference/diversity/cluster_head_{variable_1}.png")

    plt.clf()  # Clear the canvas before next plot

    for i in range(len(bruh)):
        x_vals_1 = [pt[2] for pt in cluster_list_new[i]]
        y_vals_1 = [pt[3] for pt in cluster_list_new[i]]
        plt.scatter(x_vals_1, y_vals_1, color=colors[i], label=bruh[i])

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot of Four Classes")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Again invert Y-axis
    plt.savefig(f"/home/saksham.gupta/inference/diversity/cluster_torso_{variable_1}.png")

    plt.clf()

    
    file_path = "/home/saksham.gupta/inference/diversity/dispersion.json"

    # Load existing list
    if os.path.getsize(file_path) != 0:
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Append the new dictionary
    existing_data.append(dispersion_dict)

    # Write back to file
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)
        
        
plt.clf() # top is from writing the points ion teh same canvas

print("DONE-------------------------")
pdb.set_trace()

# need to fix this dection of the code


for i in range(len(bruh_1)):
    array = [i for i in range()]







for i in range(len(bruh_1)):
    x_vals = [pt[0] for pt in list_1[0][i]]
    y_vals = [pt[1] for pt in list_1[1][i]]
    plt.scatter(x_vals, y_vals, color=colors[i], label=bruh_1[i])

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of Four Classes")
plt.legend()
plt.grid(True)
plt.savefig(f"/home/saksham.gupta/inference/diversity/per_split/overlap_distribution_{variable_1}_{config['parts']}_1.png")

plt.clf() # top is from writing the points in the same canvas


for i in range(len(bruh_1)):
    x_vals = [pt[0] for pt in list_2[i]]
    y_vals = [pt[1] for pt in list_2[i]]
    plt.scatter(x_vals, y_vals, color=colors[i], label=bruh_1[i])

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of Four Classes")
plt.legend()
plt.grid(True)
plt.savefig(f"/home/saksham.gupta/inference/diversity/per_split/geometric_distribution_{variable_1}_{config['parts']}_1.png")
        
        
plt.clf() # top is from writing the points in the same canvas
   
 
# Visualizing the no. of points where head lies- this has to form a cluster?




# comparing list_1 and overlay_distribution_ht_class
pdb.set_trace()


















