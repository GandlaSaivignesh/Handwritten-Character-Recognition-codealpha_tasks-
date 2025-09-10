import argparse, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict(model_path, image_path):
    model = load_model(model_path)
    img = image.load_img(image_path, color_mode='grayscale', target_size=(28,28))
    img_arr = image.img_to_array(img)/255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_arr)
    return int(np.argmax(pred))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/best_model.h5")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print("Predicted class:", predict(args.model_path, args.image))
