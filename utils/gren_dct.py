from scipy.fftpack import dctn
import numpy as np
import os
def extract_and_dct_conv_filters(layer, output_file):
    filters = layer.weight.data.cpu().numpy()
    dct_filters = []
    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)
    with open(output_file, 'w') as f:
        for filter_idx in range(filters.shape[0]):
            filter_data = filters[filter_idx]
            f.write(f'Convolutional Filter {filter_idx + 1} (Before DCT):\n')
            f.write(np.array2string(filter_data, separator=',') + '\n')

            filter_dct = dctn(filter_data, type=2, norm='ortho')  
            f.write(f'Convolutional Filter {filter_idx + 1} (After DCT):\n')
            f.write(np.array2string(filter_dct, separator=',') + '\n\n')
            dct_filters.append(filter_dct)

    return dct_filters

