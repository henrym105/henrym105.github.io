# To run, use command: "streamlit run app.py"

import torch
import streamlit as st
from PIL import Image
from torchvision.transforms import ToTensor
from streamlit_drawable_canvas import st_canvas
from torchnn import ImageClassifier  # my custom neural network class
import os

HOME = os.getcwd()
if 'Project1' in HOME:
    model_state_path = f'{HOME}/model_state.pt'
else:
    model_state_path = f'{HOME}/Project1/model_state.pt'


# Load the trained model
model = ImageClassifier()
model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
model.eval()

# create a title
st.title('Number Predictor')
st.write('Use your cursor to draw a digit (0-9), then click predict button.')
# st.write('\nnote, for best resutls draw the number as large as possible.')

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color='black',  # Fixed fill color with black
    stroke_width=10,
    stroke_color='white',
    background_color='black',
    width=280,
    height=280,
    drawing_mode='freedraw',
    key='canvas'
)

# add help caption below canvas
st.caption('NOTE, for best results draw the number as large as possible.')

def process_image(image_data):
    """
    Process the image data from the canvas to a format suitable for the model.
    """
    # Convert the canvas drawing to an Image
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    # Convert to grayscale
    img = img.convert('L')
    # Resize to 28x28
    img = img.resize((28, 28))
    return img

def predict_number(image):
    """
    Predict the number from the processed image.
    """
    # Convert image to tensor
    img_tensor = ToTensor()(image).unsqueeze(0)
    # Predict
    with torch.no_grad():
        prediction = model(img_tensor)
    return torch.argmax(prediction, dim=1).item()

# When the user clicks the 'Predict number drawn' button
if st.button('Predict number drawn'):
    if canvas_result.image_data is not None:
        # Process the image and prepare it for the model
        img = process_image(canvas_result.image_data)
        # Predict the number
        predicted_number = predict_number(img)
        # st.write(f'Predicted number: {predicted_number}')
        st.write(f"## **Prediction:** {predicted_number}")
        # st.write(f"## {predicted_number}")

    else:
        st.write('Please draw a digit to predict.')
