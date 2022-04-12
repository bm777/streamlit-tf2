import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from pathlib import Path
st.set_option('deprecation.showfileUploaderEncoding', False)


SIDEBAR_XRAY = 'X-Ray Image'
SIDEBAR_CT_SCAN = 'CT-Scan Image'
SIDEBAR_OPTIONS = [SIDEBAR_XRAY, SIDEBAR_CT_SCAN]


@st.cache(allow_output_mutation=True)
def loading_model():
    model_path = "model_lung_disease_detection.h5"
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache(allow_output_mutation=True)
def load_image(img):
    img = Image.open(img)
    return img

if __name__== '__main__':
    model = loading_model()

    st.title("Lung Disease Detection")

    st.sidebar.title("Which images you have?")

    app_mode = st.sidebar.selectbox("Please Select from the following", SIDEBAR_OPTIONS)

    if app_mode == SIDEBAR_XRAY:
        flag = 1
    elif app_mode == SIDEBAR_CT_SCAN:
        flag = 0
    else:
        raise("None of them")

    uploaded_file = st.file_uploader("Upload your X-Ray/ CT Scan Image", type=['png', 'jpg', 'jpeg'])

    if  uploaded_file:

        file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type,
                              "filesize":uploaded_file.size}

        st.write(file_details)
        st.image(load_image(uploaded_file), width = 250)
        file = os.path.join("temp", uploaded_file.name)
        with open(file, "wb") as f:
            f.write((uploaded_file).getbuffer())

        # st.success("File Saved!")

        img = tf.keras.preprocessing.image.load_img(file, target_size = (256,256),)
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(input_arr, axis=0) 

        x = model.predict(img)

        x=x.reshape(1,-1)[0]
        print(x)
        print(x[0]>0.5)
        print(x[1]>0.5)
        print(x[2]>0.5)
        print(x[3]>0.5)
        #['diseased_ctscan', 'diseased_xray', 'not_diseased_ctscan', 'not_diseased_xray'] class
        ct = [x[0], x[3]]
        print(ct)

        if flag ==1:
            if x[1]> 0.5:
                st.success("Lung Disease")
            else:
                st.success("No Disease")
        if flag==0:
            if x[0]>0.5:
                st.success("Lung Disease")
            else:
                st.success("No Disease")
        # st.success("Prediction done")
