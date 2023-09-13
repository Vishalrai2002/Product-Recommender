import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit.components.v1 as components


# python -m streamlit run main.py


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st. set_page_config(page_title="BrainCafe",page_icon="",layout="wide")
import numpy as np

    
# st.image("", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
import streamlit as st

# Define your title text
title_text = "BrainCafe"

# Define the desired color (e.g., "red", "#00ff00", "rgb(255, 0, 0)")
color = "red"

# Create a custom HTML element with CSS styling
styled_title = f'<h1 style="color: {color};">{title_text}</h1>'

# Display the styled title using st.markdown
st.markdown(styled_title, unsafe_allow_html=True)

page = """ <nav>
            <ul id="MenuItems">
              <li><a href="index.html">Home</a></li>
              <li><a href="http://localhost:8501">Upload Image</a></li>
              <li><a href="products.html">Products</a></li>
              <li><a href="">About</a></li>
              <li><a href="">Contact</a></li>
              <li><a href="account.html">Account</a></li>
              <!-- TODo: 22:20 -->
            </ul>
          </nav>  
          """

st.title('Find Product from Image "Personalized Product Recommendations system"')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
# uploaded_file = st.file_uploader("Choose an image")
with st.chat_message("user"):
    st.write("Hello 👋")
    st.write("I am your Personal Assiatant...")
    st.write("Please Upload the image for you want Recommendation")
    uploaded_file = st.file_uploader("Choose an image")
    # st.line_chart(np.random.randn(30, 3))
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        with st.chat_message("user"):
            st.write("Showing recommendations for this Image : ")
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        message = st.chat_message("user")
        message.write("Hello👋 here are some recommendations :")
        # message.bar_chart(np.random.randn(30, 3))
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
            

        
        if len(indices)>=10 :
            col6,col7,col8,col9,col10 = st.columns(5)
            with col6:
                st.image(filenames[indices[0][5]])
            with col7:
                st.image(filenames[indices[0][6]])
            with col8:
                st.image(filenames[indices[0][7]])
            with col9:
                st.image(filenames[indices[0][8]])
            with col10:
                st.image(filenames[indices[0][9]])
                
        if len(indices) >= 15 :
            col11,col12,col13,col14,col15 = st.columns(5)
            with col11:
                st.image(filenames[indices[0][10]])
            with col12:
                st.image(filenames[indices[0][11]])
            with col13:
                st.image(filenames[indices[0][12]])
            with col14:
                st.image(filenames[indices[0][13]])
            with col15:
                st.image(filenames[indices[0][14]])
                
        if len(indices) >= 20 :
            col16,col17,col18,col19,col20 = st.columns(5)
            with col16:
                st.image(filenames[indices[0][15]])
            with col17:
                st.image(filenames[indices[0][16]])
            with col18:
                st.image(filenames[indices[0][17]])
            with col19:
                st.image(filenames[indices[0][18]])
            with col20:
                st.image(filenames[indices[0][19]])
            
    else:
        st.header("Some error occured in file upload")

