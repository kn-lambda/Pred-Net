import numpy as np


def load_npy(file_name, num_frames=10, limit=None):
    """load video npy-data, and divide the sequence of frames into batches
    Args:
        file_name (str)  : path of npy file
        num_frames (int) : number of frames in each batch
        limit (int or None) : the number of sequences returned
    Returns:
        numpy array : normalized video data -- shape (batch, time, height, width, channel)
    """
    datalist = []
    img_seq = np.load(file_name)
    num_batches = img_seq.shape[0] // num_frames
    
    for i in range(num_batches):
        
        if limit is not None and i >= limit:
            break
        
        batch = img_seq[i * num_frames : (i+1)*num_frames]
        datalist.append(batch)
        
    data = np.array(datalist).astype(np.float32)
    data = data / 255
    
    return data


def iterate_arr(arr, batch_size, shuffle=True, repeat=True):
    """make generator of data
    Args:
        arr (numpy array) : input data
        batch_size (int) : the number of batches returned
        shuffle (bool) : if true, data is shuffled when the next epoch begins
        repeat (bool)  : if false, only one epoch
    Returns:
        numpy array : shape (batch, time, height, width, channel)
    """
    length = arr.shape[0]
    start_index = 0
    end_index = batch_size
    
    while True:
        
        if end_index <= length:
            yield arr[start_index : end_index]
            start_index += batch_size
            end_index = start_index + batch_size
            
        else:
            if repeat:
                start_index = 0
                end_index = batch_size
                
                if shuffle:
                    np.random.shuffle(arr)
                yield arr[start_index : end_index]
                start_index += batch_size
                end_index = start_index + batch_size
                
            else:
                yield None
