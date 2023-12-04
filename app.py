# To run, use command: "streamlit run app.py"

import subprocess
import sys
import time
import streamlit as st
from PIL import Image
from torchnn import ImageClassifier  # Import your neural network class

# from torchvision.transforms import ToTensor
# import torch
# from streamlit_drawable_canvas import st_canvas

try: 
    from torchvision.transforms import ToTensor
except ModuleNotFoundError as e:
    subprocess.Popen([f'{sys.executable} -m pip install torchvision'], shell=True)
    from torchvision.transforms import ToTensor

try:
    import torch
except ModuleNotFoundError as e:
    subprocess.Popen([f'{sys.executable} -m pip install torch'], shell=True)
    import torch
    time.sleep(90)

try:
    from streamlit_drawable_canvas import st_canvas
except ModuleNotFoundError as e:
    subprocess.Popen([f'{sys.executable} -m pip install streamlit_drawable_canvas'], shell=True)
    from streamlit_drawable_canvas import st_canvas
    time.sleep(90)


# Load the trained model
model = ImageClassifier()
model.load_state_dict(torch.load('model_state.pt', map_location=torch.device('cpu')))
model.eval()

st.title('Number Predictor')

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
