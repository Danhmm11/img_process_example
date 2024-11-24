import cv2 
import yaml
import numpy as np
from motpy import Detection,ModelPreset, MultiObjectTracker
import argparse
import logging

#deefine maping 
#__________________________________________________________________________________________________________________________
# Mapping of threshold types
THRESHOLD_TYPES = {
    "THRESH_BINARY": cv2.THRESH_BINARY,
    "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
    "THRESH_TRUNC": cv2.THRESH_TRUNC,
    "THRESH_TOZERO": cv2.THRESH_TOZERO,
    "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV,
    "ADAPTIVE_THRESH_MEAN_C": cv2.ADAPTIVE_THRESH_MEAN_C,
    "ADAPTIVE_THRESH_GAUSSIAN_C": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    "THRESH_OTSU": cv2.THRESH_OTSU
}
BLUR_TYPES = {
    "GAUSSIAN": cv2.GaussianBlur,
    "MEAN": cv2.blur

}
MODEL_SPEC = {
    "constant_acceleration_and_static_box_size_2d": ModelPreset.constant_acceleration_and_static_box_size_2d.value,
    
    }

LOGGING_LEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL

}

list_name_id_square=[]
list_name_id_circle=[]
#_________________________________________________________________________________________________



parser = argparse.ArgumentParser(description="Chương trình demo argparse")

# Thêm các tham số
parser.add_argument("-name","--filename",action="store", help="file name of input",default=None,type = str)
parser.add_argument("-o", "--output",  action="store", help="file name of output", default=None ,type= str)
parser.add_argument("-c", "--config", action="store", help="Config file", default="config.yaml",type= str)
parser.add_argument("-s", "--sigma", action="store", help="sigma use for blur", default= None, type = int )
parser.add_argument("-iter_erode","--iterations_erode", action = "store", help = "iterations_erode", default = None, type = int )
parser.add_argument("-iter_dilate","--iterations_dilate", action = "store", help = "iterations_dilate", default = None, type = int )
parser.add_argument("-th", "--threshold", action="store", help="theshold use for tracking", default= None, type = int )
parser.add_argument("-m", "--mode", action="store", help="not using config", default= 0, type = bool )
parser.add_argument("--model_log", action="store", help="mode log", default= 'a', type = str )
parser.add_argument("--speed",action = "store", help = "speed of video", default = None, type = int )
parser.add_argument("--level_log",action = "store", help = "level of logging", default = "DEBUG", type = str )
# Phân tích tham số
args = parser.parse_args()


#logging config

def config_logger(): 
    """
    Configures the logging settings for the application.

    Sets the logging level, format, and handlers (file and console) based on 
    command-line arguments.

    Logging levels are defined in the LOGGING_LEVEL dictionary and can be 
    specified by the user through the --level_log argument, defaulting to 
    "DEBUG".

    The log messages are saved to "params.log" with the mode specified by 
    the --model_log argument and are also printed to the console.
    """
    logging.basicConfig(
    level=LOGGING_LEVEL.get(args.level_log, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("params.log", mode = args.model_log),
        logging.StreamHandler()  # Ghi log ra console
    ]
   
)
    logging.info("................Start.............")
config_logger()    




# argparse




def log_dict(data, parent_key=""):
    """
    Ghi log nội dung dictionary (hoặc nested dictionary) theo từng dòng.

    Args:
        data (dict): Dữ liệu cần log.
        parent_key (str): Key cha (dùng để log các key con rõ ràng hơn).
    """  
    for key, value in data.items():
        if isinstance(value, dict):  # Nếu giá trị là dictionary, gọi đệ quy
            logging.info(f"{parent_key}{key}:")
            log_dict(value, parent_key=f"{parent_key}{key}.")
        else:
            logging.info(f"{parent_key}{key}: {value}")
    
def load_config(config_path = "config.yaml"):
    
    """
    Load the configuration file specified by config_path and apply the command line
    arguments to the configuration. If the --mode argument is True, the command line
    arguments will overwrite the values in the configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration.
    """

    with open(config_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)  # Nạp dữ liệu vào từ điển
    logging.debug(args)
    mapping_dict = {
        "filename": "load_video.path",         
        "output": "save_frame.out_path",       
        "sigma": "blur.sigma",                
        "threshold": "threshold.threshold",   
        "iterations_erode": "threshold.iterations_erode",  
        "iterations_dilate": "threshold.iterations_dilate",
        "speed": "show_frame.speed",   
    }
    
    if args.mode:
        print(mapping_dict.items())
        for arg_name, config_key in mapping_dict.items():
            arg_value = getattr(args, arg_name)
            if arg_value is not None: 
                keys = config_key.split(".")  # Phân tách đường dẫn trong mapping_dict
                current = data
                for key in keys[:-1]:  # Tới phần tử cuối cùng của đường dẫn                    
                    current = current[key]
                current[keys[-1]] = arg_value 
        logging.debug(f"input args: {args}") 
    return data

def put_text(frame, config):
    """
    Function to put text on the frame
    arg:
        frame nparray = the frame to be shown
        text string = the text to be put on the frame
    return:
      
    """
    cv2.putText(frame, config['mssv'], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, config['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    

 
def show_this(frame, config, name="frame2"):
    """
    Displays a frame in a resizable window.

    Args:
        frame (numpy.ndarray): The frame to be displayed.
        config (dict): Configuration dictionary containing width and height for the window.
        name (str): Name of the window. Defaults to "frame2".
    """
    # Extract width and height from the configuration
    size_frame = tuple(config['size'])

    # Resize the frame to the specified dimensions
    resized_frame = cv2.resize(frame,size_frame)

    # Create and configure a resizable window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size_frame[0], size_frame[1])

    # Display the frame in the window
    cv2.imshow(name, resized_frame)

def threshold(frame, config):
    """
    Applies thresholding, erosion, and dilation on the input frame.

    Args:
        frame (numpy.ndarray): The input grayscale frame to be thresholded.
        config (dict): Configuration dictionary for thresholding and morphological operations.

    Returns:
        tuple: 
            - ret (float): Thresholding return value.
            - cleaned_image (numpy.ndarray): Processed image after thresholding and morphological operations.
            - frame_threshold (numpy.ndarray): The raw thresholded frame.
    """
    # Combine all specified threshold types
    type_threshold = sum(THRESHOLD_TYPES.get(item, 0) for item in config['type'])

    # Extract thresholding parameters
    low= config['threshold']
    kernel_size = tuple(config['kerner_size'])
    iterations_erode = config['iterations_erode']
    iterations_dilate = config['iterations_dilate']

    # Create the kernel for morphological operations
    kernel = np.ones(kernel_size, np.uint8)

    # Apply thresholding
    ret, frame_threshold = cv2.threshold(frame, low, 255, type=type_threshold)

    # Perform morphological opening (erosion followed by dilation)
    opened_image = cv2.erode(frame_threshold, kernel, iterations=iterations_erode)
    closed_image = cv2.dilate(opened_image, kernel, iterations=iterations_dilate)

    # Ensure the output is grayscale-compatible
    result = cv2.bitwise_and(closed_image, closed_image)

    # Convert to RGB for consistency in further processing
    cleaned_image = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return ret, cleaned_image, frame_threshold




def count_square_circle(frame, original_image, config, tracker, tracker2):
    """
    Function to process the frame and classify shapes into squares or circles.
    Args:
        frame (np.ndarray): The frame after thresholding.
        original_image (np.ndarray): Original frame for visualization.
        config (dict): Configuration dictionary.
        tracker (MultiObjectTracker): Tracker for square objects.
        tracker2 (MultiObjectTracker): Tracker for circle objects.
    Returns:
        np.ndarray: Frame with annotated shapes and IDs.
    """
    contours, _ = cv2.findContours(
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Parameters from config
    circle_ratio = config['circle']
    square_ratio = config['square']
    min_width = config['min_width']
    min_height = config['min_height']
    bounding_box_area_max = config['bounding_box_area_max']
    
    square_detections = []
    circle_detections = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small contours (noise)
        if w > min_width and h > min_height:
            roi = frame[y:y + h, x:x + w]
            contour_area = cv2.contourArea(contour)
            bounding_box_area = w * h
            if (bounding_box_area > bounding_box_area_max):
                continue
            fill_ratio = contour_area / bounding_box_area
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == 4 and fill_ratio > square_ratio:  # Square
                square_detections.append(Detection(box=[x, y, x + w, y + h]))
            elif len(approx) > 4 and circle_ratio < fill_ratio < square_ratio:  # Circle
                circle_detections.append(Detection(box=[x, y, x + w, y + h]))
           
    # Tracking detections
    tracked_squares = tracker.step(square_detections)
    tracked_circles = tracker2.step(circle_detections)

    # Annotate squares
    for obj in tracked_squares:
        x1, y1, x2, y2 = map(int, obj.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Square ID: {obj.id[:5]}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        list_name_id_square.append(obj.id[:5])
       
    # Annotate circles
    for obj in tracked_circles:
        x1, y1, x2, y2 = map(int, obj.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"Circle ID: {obj.id[:5]}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        list_name_id_circle.append(obj.id[:5])
    # Display counts
    cv2.putText(frame, f"Squares: {len(set(list_name_id_square))}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Circles: {len(set(list_name_id_circle))}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
  
    return frame


# Replace this function in `main` and call `count_square_circle` instead of `count_square_crile`.


def denoise(frame,config):
    """
    the funtion make the frame from the video and process this
    return
        frame nparray: result of the frame

    """
    frame_denoised = cv2.GaussianBlur(frame,tuple(config['kernel_size']), config['sigma'])
    return frame_denoised


def make_frame_show(frames=[], frame_size=(640, 480)):
    """
    Processes a list of frames into a single combined frame for display.

    Args:
        frames (list): List of frames to process.
        frame_size (tuple): Target size for resizing each frame.

    Returns:
        numpy.ndarray: Combined frame with all input frames arranged.
    """
    if len(frames) < 4:
        logging.error(f"error: {ValueError}")
        raise ValueError("At least 4 frames are required to combine.")
        
    processed_frames = []
    # Convert grayscale frames to RGB and resize all frames
    for frame in frames:
        if len(frame.shape) == 2:  # Grayscale frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = cv2.resize(frame, frame_size)
        processed_frames.append(frame)
    
    # Combine frames into a grid (2x2 layout)
    top_row = np.hstack((processed_frames[0], processed_frames[1]))  # Top row
    bottom_row = np.hstack((processed_frames[2], processed_frames[3]))  # Bottom row
    combined_frame = np.vstack((top_row, bottom_row))
    
    return combined_frame

def create_continus(config):
    """
    Creates a multi-object tracker based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing:
            - 'dt': Time delta for each frame (in seconds).
            - 'fps': Frames per second of the video.
            - 'model_spec': Tracker model specification key.
            - 'tracker_kwargs': Parameters for the tracker, including:
                * 'min_steps_alive': Minimum steps required to consider a track active.
                * 'max_staleness': Maximum frames of inactivity before a track is removed.

    Returns:
        MultiObjectTracker: A tracker instance configured with the provided settings.
    """
    # Extract parameters from the configuration
    dt_per_frame = config['dt'] / config['fps']
    model_spec = MODEL_SPEC.get(config['model_spec'])
    tracker_params = config.get("tracker_kwargs", {})
    
    # Initialize tracker with configuration
    tracker = MultiObjectTracker(
        dt=dt_per_frame,
        model_spec=model_spec,
        active_tracks_kwargs={
            'min_steps_alive': tracker_params.get('min_steps_alive', 2),
            'max_staleness': tracker_params.get('max_staleness', 6)
        },
        tracker_kwargs={
            'max_staleness': tracker_params.get('max_staleness', 12)
        }
    )
    return tracker

def add_frame(frames, config):
    """
    Selects and returns specific frames based on configuration.

    Args:
        frames (list): List of available frames.
        config (dict): Configuration dictionary containing the 'frame_choose' key, 
                       which specifies the indices of frames to select.

    Returns:
        list: List of selected frames based on the configuration.
    """
    # Get the list of indices to select frames
    frame_indices = config['frame_choose']

    # Select frames based on the indices
    selected_frames = [frames[i] for i in frame_indices]

    return selected_frames


def main():
    """
    Main process flow for object detection and tracking.
    Loads configuration, processes video frames, and saves results.
    """
    # Load configuration
    config = load_config(args.config)
    
    logging.info("<<<<<<<<<<<Config file>>>>>>>>>>>>>")
    log_dict(config)
    logging.info("<<<<<<<<<<<End Config file>>>>>>>>>>>>>")
    # Initialize video capture and output writer
    video_path = config["load_video"]["path"]
    frame_size = tuple(config["show_frame"]["size"])
    fps = config["show_frame"]["fps"]
    output_path = config["save_frame"]["out_path"]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    speed = config["show_frame"]["speed"]
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    
    # Initialize trackers
    tracker = create_continus(config.get("counter"))
    tracker2 = create_continus(config.get("counter"))
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        blurred_frame = denoise(gray, config.get("blur"))

        # Thresholding
        _, thresholded_frame, pre_threshold_frame = threshold(blurred_frame, config.get("threshold"))

        # Detect and count objects (squares and circles)
        annotated_frame = count_square_circle(thresholded_frame, frame, config.get("counter"), tracker, tracker2)

        # Combine frames for display
        frames_to_show = [frame, gray, blurred_frame, pre_threshold_frame, annotated_frame]
        selected_frames = add_frame(frames_to_show, config.get("display"))
        concatenated_frame = make_frame_show(selected_frames, frame_size)

        # Add text annotations
        put_text(concatenated_frame, config.get("text_video"))

        # Resize and write to output video
        resized_frame = cv2.resize(concatenated_frame, frame_size)
        out.write(resized_frame)

        # Display the result
       
        show_this(concatenated_frame, config.get("show_frame"))

        # Exit if 'q' is pressed
        # if cv2.waitKey(0) & 0xFF == ord('n'):
        #     continue
        if cv2.waitKey(speed) & 0xFF == ord('q'):
            break
        
    # REMOVE ID CIRCLE AND SQUARE
    

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    result_circle = len(set(list_name_id_circle))
    result_square = len(set(list_name_id_square))
    print("circle:"+ str(len(set(list_name_id_circle))))
    print("square:"+ str(len(set(list_name_id_square))))
    logging.debug("AAAA")
if __name__ == "__main__":
    main()
    logging.info("................END.............")
    
    