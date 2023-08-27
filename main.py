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
from PIL import Image


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#python -m streamlit run main.py


st. set_page_config(page_title="BrainCafe",page_icon="✌",layout="wide")
   

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/styleForm.css")



image = Image.open('logo.png')

st.image(image, caption=None, width=200, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
 

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
    st.write("I am your Personal Assiatant...")
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
        
        
        
# contact

with st.container():
        st.write("---")
        st.header("Get In Touch With Us!")
        st.write("##")
        
        contact_form="""
        <form action="https://formsubmit.co/raivishal261252@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your Name" required>
            <input type="email" name="email" placeholder="Your Email" required>
            <textarea name="message" placeholder="Your Message" required></textarea>
            <button type="submit">Send!</button>
        </form>
        """
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()
        
