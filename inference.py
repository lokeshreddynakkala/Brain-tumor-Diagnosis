import io
import numpy as np
from PIL import Image

def load_image_bytes(file_bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    return arr

def preprocess_for_keras(img_array, target_size=(224,224)):
    from PIL import Image as PILImage
    arr = PILImage.fromarray((img_array*255).astype('uint8')).resize(target_size)
    arr = np.array(arr).astype('float32')/255.0
    return np.expand_dims(arr, 0)

def preprocess_for_torch(img_array, target_size=(224,224)):
    import torch
    from PIL import Image as PILImage
    arr = PILImage.fromarray((img_array*255).astype('uint8')).resize(target_size)
    arr = np.array(arr).astype('float32')/255.0
    arr = np.transpose(arr, (2,0,1))
    tensor = torch.tensor(arr).unsqueeze(0)
    return tensor

def postprocess_probs(probs):
    return {'class': int(np.argmax(probs)), 'scores': probs.tolist()}
