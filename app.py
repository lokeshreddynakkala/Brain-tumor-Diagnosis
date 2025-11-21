import glob
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)
MODEL = None
MODEL_TYPE = None

def load_model():
    global MODEL, MODEL_TYPE
    h5 = glob.glob("models/*.h5")
    pts = glob.glob("models/*.pt")
    if h5:
        MODEL_TYPE = "keras"
        from tensorflow import keras
        MODEL = keras.models.load_model(h5[0])
        print("Loaded Keras model:", h5[0])
    elif pts:
        MODEL_TYPE = "torch"
        import torch
        MODEL = torch.load(pts[0], map_location='cpu')
        MODEL.eval()
        print("Loaded Torch model:", pts[0])
    else:
        print("No model found in models/")

def prepare_image(file_storage, target_size=(224,224)):
    img = Image.open(file_storage).convert('RGB').resize(target_size)
    arr = np.array(img).astype('float32')/255.0
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'No model loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    f = request.files['image']
    img = prepare_image(f, target_size=(224,224))
    if MODEL_TYPE == 'keras':
        x = np.expand_dims(img, 0)
        prob = MODEL.predict(x).tolist()[0]
        return jsonify({'class': int(np.argmax(prob)), 'scores': prob})
    elif MODEL_TYPE == 'torch':
        import torch
        t = torch.tensor(np.transpose(img, (2,0,1))).unsqueeze(0)
        with torch.no_grad():
            out = MODEL(t)
            if isinstance(out, (list, tuple)):
                out = out[0]
            prob = torch.nn.functional.softmax(out, dim=1).cpu().numpy().tolist()[0]
            import numpy as _np
            return jsonify({'class': int(_np.argmax(prob)), 'scores': prob})
    else:
        return jsonify({'error': 'Unsupported model type'}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8000)
