import streamlit as st
from PIL import Image

# Import img segmentation python script
import img_segmentation 

st.sidebar.title("About")

st.sidebar.info("The application takes in a 2D CT pre-processed scan and returns the predicted segmentations, if any. The prediction is done using a Pre-Trained U-Net with MobileNetV2 Encoder. It achieves a Train Dice Score of 88.2 & Test Dice score of 87.9.")

st.title('Breast Tumour Segmentation Web App')

uploaded_file = st.file_uploader("Upload a pre-processed 2D CT Scan...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded CT Scan.', use_column_width=True)
    st.write("")
    st.write("")
    st.write("Classifying...")

    # Run Prediction
    pred_mask, pred_mask_com = img_segmentation.prediction(image)
    pred_mask_com_str = ', '.join(map(str, pred_mask_com))
    
    pred_image = Image.open('predicted_mask.png')
    st.image(pred_image, caption='Predicted Mask', use_column_width=True)
    st.write('Predicted Tumour Centroid is: ' + pred_mask_com_str)
    st.write("")
    st.write("End of Prediction")
