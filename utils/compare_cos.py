import numpy as np
from scipy.spatial.distance import cosine
def extract_by_index_sum(image,rate):

    depth, height, width = image.shape
    total_elements = depth * height * width


    threshold = (depth + height + width-3) * rate


    extracted_data = []
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                if d + h + w < threshold: 
                    extracted_data.append(image[d, h, w])

    return np.array(extracted_data)

def adjusted_cosine_similarity(W_a, W_b):
    W_a = np.array(W_a)
    W_b = np.array(W_b)
    
    mean_a = np.mean(W_a)
    mean_b = np.mean(W_b)
    
    W_a_centered = W_a - mean_a
    W_b_centered = W_b - mean_b
    
    numerator = np.dot(W_a_centered, W_b_centered)
    denominator = np.sqrt(np.sum(W_a_centered**2) * np.sum(W_b_centered**2))
    
    if denominator == 0:
        return 0  
    else:
        return numerator / denominator


def compare_cossim_dct_filters(dct_filters1, dct_filters2,rate=2/3):
    num_filters = min(len(dct_filters1), len(dct_filters2))
    dct_distances = []
    dct_distances_modcos=[]

    for i in range(num_filters):
        filter1 = dct_filters1[i]
        filter2 = dct_filters2[i]

       
        min_shape = np.minimum(filter1.shape, filter2.shape)
        filter1 = filter1[:min_shape[0], :min_shape[1], :min_shape[2]]
        filter2 = filter2[:min_shape[0], :min_shape[1], :min_shape[2]]

        
        low_freq1 = extract_by_index_sum(filter1,rate)
        low_freq2 = extract_by_index_sum(filter2,rate)

        
        distance = cosine(low_freq1, low_freq2)
        dct_distances.append(distance)

        distance_modcos = adjusted_cosine_similarity(low_freq1.flatten(), low_freq2.flatten())
        dct_distances_modcos.append(distance_modcos)
    return dct_distances,dct_distances_modcos

def compare_cossim_dct_filters_long(dct_filters1, dct_filters2,rate=2/3):
    num_filters = min(len(dct_filters1), len(dct_filters2))
    dct_distances = []
    dct_distances_modcos=[]
    all_low_freq1 = []
    all_low_freq2 = []

    for i in range(num_filters):
        filter1 = dct_filters1[i]
        filter2 = dct_filters2[i]

       
        min_shape = np.minimum(filter1.shape, filter2.shape)
        filter1 = filter1[:min_shape[0], :min_shape[1], :min_shape[2]]
        filter2 = filter2[:min_shape[0], :min_shape[1], :min_shape[2]]

        low_freq1 = extract_by_index_sum(filter1,rate)
        low_freq2 = extract_by_index_sum(filter2,rate)
        all_low_freq1.append(low_freq1.flatten())
        all_low_freq2.append(low_freq2.flatten())


    concatenated_low_freq1 = np.concatenate(all_low_freq1)
    concatenated_low_freq2 = np.concatenate(all_low_freq2)

   
    distance = cosine(concatenated_low_freq1, concatenated_low_freq2)
    dct_distances.append(distance)

    distance_modcos = adjusted_cosine_similarity(concatenated_low_freq1, concatenated_low_freq2)
    dct_distances_modcos.append(distance_modcos)
    return dct_distances

