import cv2
import numpy as np
import streamlit as st

uploaded_file = st.file_uploader("Choose an image file", type="jpg")

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
   
    st.image(opencv_image, channels="BGR")

    
    original_image = opencv_image
    
   
    resized_image = cv2.resize(original_image, (800, 600))  

    
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    
    edges = cv2.Canny(grayscale_image, 50, 150)

    
    sharpened_image = cv2.filter2D(original_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


    alpha = 1.5  
    beta = 50    
    contrast_enhanced_image = cv2.convertScaleAbs(sharpened_image, alpha=alpha, beta=beta)

    
    st.image(contrast_enhanced_image, use_column_width=True, channels='BGR', caption='Final Enhanced Image')
