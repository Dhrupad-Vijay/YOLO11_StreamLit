import cv2 # type: ignore
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO # type: ignore
from PIL import Image

# Get absolute path of the current file
FILE = Path(__file__).resolve()

# Get parent directory of current file
ROOT = FILE.parent

# Add root path to sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Get relative path of the root directory wrt the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'

SOURCE_LIST = [IMAGE, VIDEO]

# Image Config
IMAGE_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGE_DIR/'Sample_test_cyclist.jpg'
DEFAULT_DETECT_IMAGE = IMAGE_DIR/'Sample_detection_cyclist.jpg'

# Video Config
VIDEO_DIR = ROOT/'videos'
VIDEOS_DICT = {
    'video 1' : VIDEO_DIR/'Test_Detection_1.mp4',
    'video 2' : VIDEO_DIR/'Test_Detection_2.mp4'
}

# Model Configurations
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'yolo11n.pt'
CLASSIFICATION_MODEL = MODEL_DIR/'yolo11n-cls.pt'
SEGMENTATION_MODEL = MODEL_DIR/'yolo11n-seg.pt'

# Page layout
st.set_page_config(
    page_title="YOLO11",
    page_icon="🤖"
)

# Page header
st.header("Object detection using YOLO11")

# Side bar
st.sidebar.header(" Model Configuration")

# Choose Model: Detection, Segmentation or Pose Estimation
model_type = st.sidebar.radio("Select an option", ["Detection", "Classification", "Segmentation"])

# Select confidence value
confidence_value = float(st.sidebar.slider("Select model confidence value", 25, 100, 70))/100

# Selecting model type
if model_type == "Detection":
    model_path = Path(DETECTION_MODEL)
elif model_type == "Classification":
    model_path = Path(CLASSIFICATION_MODEL)
elif model_type == "Segmentation":
    model_path = Path(SEGMENTATION_MODEL)

# Load YOLO11 model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f'Unable to load model, check specified path - {model_path}')
    st.error(e)

# Image and video config
st.sidebar.header("Image/Video config")
source_radio = st.sidebar.radio(
    "Select source", SOURCE_LIST
)

source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        "Choose an image...", type = ('jpg', 'png', 'jpeg', 'bmp', 'heic')
    )
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = Image.open(source_image)
                st.image(source_image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f'Error occured while opening image')
            st.error(e)
    with col2:
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption="Detection image output", use_column_width=True)
            else:
                if st.sidebar.button("Detect objects"):
                    result = model.predict(uploaded_image, conf = confidence_value)
                    boxes = result[0].boxes
                    result_plotted = result[0].plot()[:,:,::-1]
                    st.image(result_plotted, caption="Detected image", use_column_width=True)
                    
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error(e)
        except Exception as e:
            st.error("Error occured while opening image")
            st.error(e)
            
elif source_radio == VIDEO:
    source_video = st.sidebar.selectbox(
        "Choose a video...", VIDEOS_DICT.keys()
    )
    with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        if st.sidebar.button("Detect video objects"):
            try:
                video_cap = cv2.VideoCapture(
                    str(VIDEOS_DICT.get(source_video))
                    
                )
                st_frame = st.empty()
                while (video_cap.isOpened()):
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9/16))))
                        # Predict objects in the image with YOLO11
                        result = model.predict(image, conf=confidence_value)
                        # Plot detected objects on video frame
                        result_plotted = result[0].plot()
                        st_frame.image(result_plotted, caption="Detected Video",
                            channels="BGR",
                            use_column_width=True)
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading viedo - "+str(e))