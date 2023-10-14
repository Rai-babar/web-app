from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from PIL import Image, ImageDraw
import numpy as np
from functions import tversky,focal_tversky,tversky_loss
import os
import io
import cv2
import keras
from keras.losses import binary_crossentropy
#import tensorflow as tf
# Register custom loss function
from keras.utils import get_custom_objects
import keras.backend as K
from skimage.io import imread
from skimage import img_as_ubyte
import uvicorn

# Custom loss functions
epsilon = 1e-5
smooth = 1


# Register the custom loss function
get_custom_objects().update({"focal_tversky": focal_tversky})

app = FastAPI()


# Load your classification, detection, and segmentation models here
detection_model = keras.models.load_model('BrainTumorDetection.h5')
segmentation_model = keras.models.load_model('ResUNet-segModel-weights.hdf5', custom_objects={"focal_tversky": focal_tversky, "tversky": tversky})
#segmentation_model = tf.keras.models.load_model('ResUNet-segModel-weights.h5')

templates = Jinja2Templates(directory="templates")

# Define the base URL for static files
base_url = "static/"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})



def segment_tumor(image_path, model_seg):
    '''
    Segments and analyzes a tumor in an image using a segmentation model.

    Args:
        image_path (str): Path to the image file.
        model_seg (tf.keras.Model): The segmentation model.

    Returns:
        np.ndarray: Segmented mask image.
    '''
    X = np.empty((1, 256, 256, 3))
    img = imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)
    
    img -= img.mean()
    img /= img.std()
    X[0,] = img
    
    predict = model_seg.predict(X)
    
    return predict.squeeze().round()

# Function to predict and save the segmentation
def predict_and_save_segmentation(image_path, seg_model, output_path):
    # Predict segmentation for the image
    segmented_mask = segment_tumor(image_path, seg_model)

    # Check if the image has a tumor (non-empty mask)
    has_tumor = np.sum(segmented_mask) > 0

    if has_tumor:
        # Draw a red bounding box around the segmented area
        contours, _ = cv2.findContours(segmented_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = imread(image_path)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the image with the red bounding box
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        # Save the segmented mask as an output
        segmented_mask_image = img_as_ubyte(segmented_mask)  # Convert to 8-bit image
        cv2.imwrite(output_path, segmented_mask_image)


# ...
@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...), model_type: str = Form(...)):
    try:
        print("Starting upload_file function...")
        # Read and preprocess the uploaded image
        contents = await file.read()

        # Use PIL to open and process the image
        image = Image.open(io.BytesIO(contents))

        # Initialize output_path
        output_path = None

        # Resize the image based on the selected model_type
        if model_type == "detection":
            # Resize for the detection model (64x64)
            image = image.resize((64, 64))
        elif model_type == "segmentation":
            # Resize for the segmentation model (256x256)
            image = image.resize((256, 256))
            # Save the uploaded image to a temporary file for segmentation
            temp_image_path = "static/temp_image.png"
            image.save(temp_image_path)
            # Call the segmentation function with the image path
            output_path = "static/segmented_image.png"
            predict_and_save_segmentation(temp_image_path, segmentation_model, output_path)
        else:
            return JSONResponse(content={"error": "Invalid model type"}, status_code=400)

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Normalize the image to a range between 0 and 1
        image_np = image_np.astype(np.float32) / 255.0

        # Convert RGBA image to RGB if it has 4 channels (alpha channel)
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # Perform inference based on the selected model_type
        if model_type == "detection":
            model = detection_model
        elif model_type == "segmentation":
            model = segmentation_model
        else:
            return JSONResponse(content={"error": "Invalid model type"}, status_code=400)

        # Make a prediction
        print("Before model prediction...")
        if model_type == "detection":
            prediction = model.predict(np.expand_dims(image_np, axis=0))
        elif model_type == "segmentation":
            # Load the segmented image for response
            segmented_image = cv2.imread(output_path)
            segmented_image_rgba = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGBA)

        print("After model prediction...")

        # Create a response with the prediction result
        response_data = {"model_type": model_type}
        if model_type == "detection":
            result_message = "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer Detected"
            response_data["result"] = result_message

            # Convert the image array to PIL Image
            img_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # Convert to uint8 and scale

            # Draw a border if cancer is detected
            if prediction[0][0] > 0.5:
                draw = ImageDraw.Draw(img_pil)
                draw.rectangle([(0, 0), (63, 63)], outline="red", width=1)

            # Resize the image to the desired size (e.g., 300x300 pixels)
            img_pil = img_pil.resize((300, 300))

            # Save the image to a temporary file in the "static" directory
            temp_image_path = os.path.join("static", "temp_result_image.png")
            img_pil.save(temp_image_path)

            # Return the response data with the image URL
            response_data["image_url"] = temp_image_path
        elif model_type == "segmentation":
            # For segmentation, return the segmented image URL
            response_data["segmented_image_url"] = output_path

        # Return the response data
        return JSONResponse(content=response_data)

    except Exception as e:
        print("Error:", str(e))  # Print the error to the console for debugging
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Add an endpoint to serve static files
@app.get("/{file_path:path}", response_class=FileResponse)
async def serve_static(file_path: str):
    return FileResponse(file_path)  # Return the FileResponse instead of a dictionary



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)