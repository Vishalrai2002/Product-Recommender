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

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st. set_page_config(page_title="Fipkart",page_icon="",layout="wide")
import numpy as np

    
st.image("data:image/png;base64,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", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
page = """ <div class="container-fluid bg-primary Header fixed-top">
        <div class="row py-2 d-flex">
            <!-- Logo -->
            <div class="col logo offset-lg-1">
                <a href="#">
                    <img width="75" src="//img1a.flixcart.com/www/linchpin/fk-cp-zion/img/flipkart-plus_8d85f4.png"
                        alt="Flipkart" title="Flipkart">
                </a>
                <a href="#">Explore <span>Plus</span>
                    <img width="10" src="//img1a.flixcart.com/www/linchpin/fk-cp-zion/img/plus_aef861.png">
                </a>
            </div>

            <!-- Search -->
            <div class="col col-md-4  search d-flex dropdown bg-white">
                <input class="form-control dropdown-toggle" type="search"
                    placeholder="Search for products, brands and more" aria-label="Search" id="navbarDropdown"
                    data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                <i class="fa fa-search mt-2 ml-sm-1 text-primary"></i>
                <div class="dropdown-menu col-12 search-item" aria-labelledby="navbarDropdown">
                    <h6 class="ml-4">Discover More</h6>
                    <div class="dropdown-divider"></div>
                    <a class="dropdown-item py-2" href=""><i class="fa fa-search text-secondary mr-3"></i>mobiles</a>
                    <a class="dropdown-item py-2" href=""><i class="fa fa-search text-secondary mr-3"></i>t-shirts</a>
                    <a class="dropdown-item py-2" href=""><i class="fa fa-search text-secondary mr-3"></i>shoes</a>
                    <a class="dropdown-item py-2" href=""><i class="fa fa-search text-secondary mr-3"></i>laptop</a>
                    <a class="dropdown-item py-2" href=""><i class="fa fa-search text-secondary mr-3"></i>tv</a>
                    <a class="dropdown-item py-2" href=""><i class="fa fa-search text-secondary mr-3"></i>sarees</a>
                </div>
            </div>

            <div class="col upload">
                <button class="form-control"><a href="http://localhost:8501">upload image</a></button>
            </div>

            <!-- Login -->
            <div class="col dropdown login">
                <button class="btn bg-white text-primary" type="button" id="loginMenuButton" data-toggle="dropdown"
                    aria-haspopup="true" aria-expanded="true">
                    Login
                </button>
                <div class="dropdown-menu login-list col-12 aria-labelledby=" loginMenuButton">
                    <div class="d-flex">
                        <h6 class="ml-md-1">New Customer?</h6>
                        <a href="#" class="ml-auto mr-2" id="signUp">Sign Up</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-user-circle text-primary mt-2 ml-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="">My Profile</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-plus text-primary mt-2 ml-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="">Flipkart Plus Zone</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-book text-primary mt-2 ml-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="">Orders</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-heart text-primary mt-2 ml-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="">Wishlist</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-chess-bishop text-primary mt-2 ml-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="">Rewards</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-gift text-primary mt-2 ml-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="">Gift Cards</a>
                    </div>
                </div>
            </div>

            <!-- More -->
            <div class="col dropdown more">
                <a class="btn dropdown-toggle text-white ml-lg-2 ml-sm-0" href="#" role="button" id="moreMenuLink"
                    data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                    More
                </a>

                <div class="dropdown-menu more-list" aria-labelledby="moreMenuLink">
                    <div class="d-flex">
                        <i class="fa fa-bell text-primary ml-md-3 mt-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="#">Notification Preferences</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-archive text-primary ml-md-3 mt-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="#">Sell On Flipkart</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-question-circle text-primary ml-md-3 mt-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="#">24x7 Customer Care</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-chart-line text-primary ml-md-3 mt-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="#">Advertise</a>
                    </div>
                    <div class="dropdown-divider"></div>
                    <div class="d-flex">
                        <i class="fa fa-download text-primary ml-md-3 mt-2" aria-hidden="true"></i>
                        <a class="dropdown-item" href="#">Download App</a>
                    </div>
                </div>
            </div>

            <!-- Cart -->
            <div class="col col-md-1 d-flex justify-content-center">
                <i class="fa fa-shopping-cart text-white mt-2" aria-hidden="true"></i>
                <a href="" class="btn text-white">Cart</a>
            </div>
        </div>
    </div> """

st.title('Find Product from Image "Recommender System"')

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
    st.write("I am your flipkart assiatant...")
    st.write("Please Upload the image for you want Recommendation")
    uploaded_file = st.file_uploader("Choose an image")
    # st.line_chart(np.random.randn(30, 3))
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        with st.chat_message("user"):
            st.write("Showing recommendations for this Inmage : ")
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        message = st.chat_message("user")
        message.write("Hello human here are some recommendations :")
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
