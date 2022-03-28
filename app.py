import streamlit as st

from data import *
from input import  from_image, set_style

st.title("My Famous Portrait")

style_name = st.sidebar.selectbox("Choose Style", style_images_names)

set_style(style_name)

# st.sidebar.header('Choose Content')

method = st.sidebar.radio('Choose Content', options=['Upload', 'Celebs'])

from_image(style_name, method)


