import tensorflow as tf
from typing import List
import cv2
import os

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale and crop
        gray = tf.image.rgb_to_grayscale(frame)
        cropped = gray[190:236, 80:220, :]
        frames.append(cropped)
    cap.release()

    # Stack frames into a tensor
    video = tf.stack(frames, axis=0)
    mean = tf.math.reduce_mean(video)
    std = tf.math.reduce_std(tf.cast(video, tf.float32))
    return tf.cast((video - mean), tf.float32) / std


def load_alignments(path: str) -> tf.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Alignment file not found: {path}")
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] != 'sil':
            tokens.append(parts[2])
    # Prepend a space token to separate words
    spaced = [' '] + tokens
    splits = tf.strings.unicode_split(spaced, input_encoding='UTF-8')
    ids = char_to_num(tf.reshape(splits, (-1,)))
    # Drop the initial space token
    return ids[1:]


def load_data(path: tf.Tensor):
    # Decode filename and get base name
    p = path.numpy().decode('utf-8')
    file_name = os.path.splitext(os.path.basename(p))[0]

    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments
