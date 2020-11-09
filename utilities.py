def standardize(x):
    t1 = tf.math.divide_no_nan(
    tf.subtract(
        x['t1'], 
        tf.reduce_min(x['t1'])
    ), 
    tf.subtract(
        tf.reduce_max(x['t1']), 
        tf.reduce_min(x['t1'])
    )
  )
    t2 = tf.math.divide_no_nan(
    tf.subtract(
        x['t2'], 
        tf.reduce_min(x['t2'])
    ), 
    tf.subtract(
        tf.reduce_max(x['t2']), 
        tf.reduce_min(x['t2'])
    )
  )
    t1c = tf.math.divide_no_nan(
    tf.subtract(
        x['t1c'], 
        tf.reduce_min(x['t1c'])
    ), 
    tf.subtract(
        tf.reduce_max(x['t1c']), 
        tf.reduce_min(x['t1c'])
    )
  )
    flair = tf.math.divide_no_nan(
    tf.subtract(
        x['flair'], 
        tf.reduce_min(x['flair'])
    ), 
    tf.subtract(
        tf.reduce_max(x['flair']), 
        tf.reduce_min(x['flair'])
    )
  )
    mask = x['mask']
    return {'flair':flair, 'mask':mask, 't1':t1, 't2':t2, 't1c':t1c}

def reshaping(x):
    """
    Takes a tensor and reshapes it.
    Inputs:
          x (tuple of tensors with shape (240,240))
    Outputs:
          image, mask (tuples of tensors with shape (240,240,1))
    """
    batchsize = -1
    dims = (240,240,1)
    image = tf.stack([x['t1'],x['flair'],x['t2'],x['t1c']], axis=0)
    mask = tf.stack([x['mask'], x['mask'], x['mask'],x['mask']], axis=0)
    return image, mask

def get_folders():
    """
    Returns a list of all the folders in the HGG and LGG directories.
    """
    HGG_Folders= os.listdir('/content/drive/My Drive/MICCAI_BraTS_2019_Data_Training/HGG')
    LGG_Folders = os.listdir('/content/drive/My Drive/MICCAI_BraTS_2019_Data_Training/LGG')
    return(HGG_Folders, LGG_Folders)
  
def load_nii_data(filename):
    """
    Load in an nii file and return an array containing the pixel data. 
    inputs:
        filename (string): The name of the nii file you want to load.
    outputs:
        array (numpy array): A numpy array of the given image.
    """
    image = nib.load(filename)
    array = image.get_fdata()
    array = array.astype(np.float16)
    return(array)


def binarize(x):
    """
    Convert a given mask array from having multiple categories to 1 and 0.
    inputs:
          array (a numpy array): An array containing multiple integer codings for categories. [0,1,2,3]
    outputs:
          array (a numpy array): An array containing only 1s and 0s. 
    """
    mask = tf.where(x['mask']>0, 1, 0)
    flair = x['flair']
    t1 = x['t1']
    t2 = x['t2']
    t1c= x['t1c']
    return{'flair':flair, 'mask':mask, 't1':t1, 't2':t2, 't1c':t1c}



def cast(x):
    mask = tf.cast(x['mask'], tf.float32)
    flair = tf.cast(x['flair'], tf.float32)
    t1 = tf.cast(x['t1'], tf.float32)
    t2 = tf.cast(x['t1'], tf.float32)
    t1c = tf.cast(x['t1c'], tf.float32)
    return {'flair':flair, 'mask':mask, 't1':t1, 't2':t2, 't1c':t1c}

def binary_prediction(x):
    binary = tf.where(x >.50, 1 , 0)
    binary = tf.cast(binary, tf.float32)
    return(binary)


def dice_coef(y_true, y_pred):    #original img size is 240*240,
    smooth = 1
    """
    Computes the dice coefficient for a ground truth and predicted mask.

    inputs:
          y_true (numpy array): An array of 1s and 0s corresponding to binary class assignments.
          y_pred (numpy array): Predicted class assignments
    outputs:
          dice_coef (float): The dice coefficient of the two arrays
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * binary_prediction(y_pred_f))
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    Uses predicted and actual values to calculate the dice coefficent loss. 
    """
    return 1-dice_coef(y_true, y_pred)