import streamlit as st
import os
import imageio as iio

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

st.title('LipNet Full Stack App')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip Reading')
    st.write('This application is originally build on the LipNet deep learning model')


# Generating a list of options or videos
options = os.listdir(os.path.join('data','s1'))
selected_video = st.selectbox('Choose Video',options)

# Generate two columns
col1,col2 = st.columns(2)

if options:

    with col1:
        st.info('This video displays the converted video in mp4 format')
        file_path = os.path.join('data','s1',selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        #st.info('This is all the machine learning model sees when making a prediction')
        video,annotations = load_data(tf.convert_to_tensor(file_path))
        # iio.mimsave('animation.gif', video_frames, fps=10)
        
        st.info('This is the outputs of the machine learning model as tokens')
        model = load_model()
        ywhat = model.predict(tf.expand_dims(video,axis=0))
        decoder = tf.keras.backend.ctc_decode(ywhat, [75], greedy=True)[0][0]
        st.text(decoder)
        decoder_numpy = tf.keras.backend.ctc_decode(ywhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder_numpy)

        st.info('Decode the raw tokens into words')
        st.text (num_to_char(decoder_numpy))

        st.info('Converting bunch of letters to words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder_numpy)).numpy().decode('utf-8')
        st.text(converted_prediction)