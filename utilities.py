def standardize(x):
    """
    Scales the image to have a value between 0 and 1. 
    Inputs:
        x: An image tensor.
    Outputs:
        image, mask: The standardized image and mask. 
    
    """
    image = tf.truediv(
    tf.subtract(
        x['image'], 
        tf.reduce_min(x['image'])
    ), 
    tf.subtract(
        tf.reduce_max(x['image']), 
        tf.reduce_min(x['image'])
    )
  )
    mask = x['mask']
    return (image,mask)

def reshaping(x):
    """
    Takes a tensor and reshapes it by adding an extra dimension.
    Inputs:
          x (tuple of tensors with shape (240,240))
    Outputs:
          image, mask (tuples of tensors with shape (240,240,1))
    """
    batchsize = -1
    dims = (240,240,1)
    image = tf.reshape(x['image'], [batchsize,dims[0], dims[1], dims[2]])
    mask = tf.reshape(x['mask'], [batchsize,dims[0], dims[1], dims[2]])
    return {'image':image, 'mask': mask}

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


def voxel_clip(x):
    """
    Clips the image to the 2nd and 98th percentile values.
    inputs:
        img (a numpy array): The image you want to clip
    outputs:
        img (a numpy array): The image with its values clipped
    """
    upper = np.percentile(x['image'], 98)
    lower = np.percentile(x['image'], 2)
    x['image'][x['image'] > upper] = upper
    x['image'][x['image'] < lower] = lower
    return {'image':image, 'mask': mask}

def binarize(x):
    """
    Convert a given mask array from having multiple categories to 1 and 0.
    inputs:
          array (a numpy array): An array containing multiple integer codings for categories. [0,1,2,3]
    outputs:
          array (a numpy array): An array containing only 1s and 0s. 
    """
    mask = tf.where(x['mask']>0, 1, 0)
    image = x['image']
    return {'image':image, 'mask':mask}



def cast(x):
    """
    Converts the image and mask tensors to float32.
    Inputs:
        image, mask: The image and mask tensors.
    Outputs:
        image, mask: The converted tensors.
    """
    image = tf.cast(x['image'], tf.float32)
    mask= tf.cast(x['mask'], tf.float32)
    return {'image':image, 'mask': mask}

def binary_prediction(x):
    """
    Convert a prediction into a binary outcome.
    
    Inputs: 
        x: Predicted probability.
    Outputs:
        binary: Binary outcomes.
        
    """
    binary = tf.where(x >.50, 1 , 0)
    binary = tf.cast(binary, tf.float32)
    return(binary)

def crop(x):
    tf.image.central_crop
    image = tf.cast(x['image'], tf.float32)
    mask= tf.cast(x['mask'], tf.float32)

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

    inputs:
          y_true (numpy array): An array of 1s and 0s corresponding to binary class assignments.
          y_pred (numpy array): Predicted class assignments.
    outputs:
          dice_coef (float): The dice loss of the two arrays.
    """
    return 1-dice_coef(y_true, y_pred)
