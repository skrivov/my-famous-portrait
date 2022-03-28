import threading
import numpy as np
import functools
import streamlit as st

from PIL import Image
import cv2
import imutils

from data import *
import io

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt


def show_image(image):
    n = len(image)
    image_size =  image.shape[1]
    w = (image_size * 6) // 320
    fig = plt.figure(figsize=(w * n, w))
    plt.imshow(image[0] , aspect='equal')
    plt.axis('off')
    st.pyplot(fig)


def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image


@functools.lru_cache(maxsize=None)
def load_local_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

@functools.lru_cache(maxsize=None)
def load_uploaded_image(content_file, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image = Image.open(content_file)
  img_byte_arr =  io.BytesIO()
  image.save(img_byte_arr, format='PNG')
  img_byte_arr = img_byte_arr.getvalue()
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      img_byte_arr,
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img


def set_style(style_name):
    style_image_path = style_images_dict[style_name]
    style = Image.open(style_image_path)
    style = np.array(style)  # pil to cv
    style = cv2.cvtColor(style, cv2.COLOR_RGB2BGR)
    st.sidebar.image(style, width=300, channels='BGR')


def from_image(style_name, method):
    style_image_path = style_images_dict[style_name]
    # load image to tf
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.
    style_image = load_local_image(style_image_path, style_img_size)

    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

    if method == 'Upload':
        content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    else:
        content_name = st.sidebar.selectbox("Choose the content images:", celeb_images_names)
        content_file = celebs_images_dict[content_name]

    if content_file is not None:
        content = Image.open(content_file)
        content = np.array(content) #pil to cv
        content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()

    # WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=200)
    # content = imutils.resize(content, width=WIDTH)
    st.sidebar.image(content, width=300, channels='BGR')

    output_image_size = 384  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)

    if method == 'Upload':
        content_image = load_uploaded_image(content_file, content_img_size)
    else:
        content_image = load_local_image(content_file, content_img_size)

    hub_handle = 'tfhub'
    hub_module = hub.load(hub_handle)

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    show_image(stylized_image)


