import os
import numpy as np
import cv2
import argparse
import pyclipper
import bpu_infer_lib
import matplotlib.pyplot as plt
import collections


class DetectionModel:
    def __init__(self, model_path, threshold=0.5, ratio_prime=2.7, input_size=(640, 640)):
        """
        Initialize the detection model.

        Args:
            model_path: Path to the model file.
            threshold: Threshold for binary classification.
            ratio_prime: Ratio used for dilation of contours.
            input_size: Desired input size for the model.
        """
        self.model = bpu_infer_lib.Infer(False)
        if not self.model.load_model(model_path):
            raise RuntimeError(f"Failed to load model from '{model_path}'.")
        
        self.threshold = threshold
        self.ratio_prime = ratio_prime
        self.input_size = input_size

    def predict(self, img, img_path, min_area=100):
        """
        Predict bounding boxes for the given image.

        Args:
            img: The input image.
            img_path: Path to the image file.
            min_area: Minimum area of the detected contour to be considered.

        Returns:
            dilated_polys: List of dilated polygons.
            boxes_list: List of bounding boxes.
        """
        img_shape = img.shape[:2]
        
        self.model.read_img_to_nv12(img_path, 0)
        self.model.forward()
        
        # preds = self.model.get_infer_res_np_float32(0, self.input_size[0] * self.input_size[1]).reshape(1, *self.input_size)
        preds = self.model.get_infer_res_np_float32(0).reshape(1, *self.input_size)
        preds = np.where(preds > self.threshold, 255, 0).astype(np.uint8).squeeze()
        preds = cv2.resize(preds, (img_shape[1], img_shape[0]))

        contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dilated_polys = self._dilate_contours(contours)
        boxes_list = self._get_bounding_boxes(dilated_polys, min_area)

        return dilated_polys, boxes_list

    def _dilate_contours(self, contours):
        """Dilate contours using the ratio_prime."""
        dilated_polys = []
        for poly in contours:
            poly = poly[:, 0, :]
            arc_length = cv2.arcLength(poly, True)
            if arc_length == 0:
                continue
            D_prime = (cv2.contourArea(poly) * self.ratio_prime / arc_length)

            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            dilated_poly = np.array(pco.Execute(D_prime))

            if dilated_poly.size == 0 or dilated_poly.dtype != np.int_ or len(dilated_poly) != 1:
                continue
            dilated_polys.append(dilated_poly)
        return dilated_polys

    def _get_bounding_boxes(self, dilated_polys, min_area):
        """Get bounding boxes from dilated polygons."""
        boxes_list = []
        for cnt in dilated_polys:
            if cv2.contourArea(cnt) < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.int_)
            boxes_list.append(box)
        return boxes_list

class strLabelConverter:
    """Convert between string and label for OCR tasks.

    Args:
        alphabet (str): Set of possible characters.
        ignore_case (bool, default=True): Whether to ignore case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # Add a special '-' character for padding

        # Create a dictionary mapping characters to indices
        self.dict = {}
        for i, char in enumerate(alphabet):
            # Note: 0 is reserved for 'blank' required by CTC loss
            self.dict[char] = i + 1

    def encode(self, text):
        """Encode a string or a list of strings into a sequence of indices.

        Args:
            text (str or list of str): The text(s) to convert.

        Returns:
            np.array: Encoded text as an array of indices.
            np.array: Array of lengths for each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return np.array(text, dtype=np.int32), np.array(length, dtype=np.int32)

    def decode(self, t, length, raw=False):
        """Decode a sequence of indices back into a string.

        Args:
            t (np.array): Encoded text as an array of indices.
            length (np.array): Array of lengths for each text.

        Raises:
            AssertionError: If the length of the text and the provided length do not match.

        Returns:
            str or list of str: Decoded text.
        """
        if len(length) == 1:
            length = length[0]
            assert len(t) == length, f"text with length: {len(t)} does not match declared length: {length}"
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # Batch decoding mode
            assert len(t) == length.sum(), f"texts with length: {len(t)} do not match declared length: {length.sum()}"
            texts = []
            index = 0
            for i in range(length.size):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], np.array([l]), raw=raw))
                index += l
            return texts

class rec_model:
    def __init__(self, model_path, converter, input_size=(48, 320), output_size=(40, 97)):
        """
        Initialize the recognition model.

        Args:
            model_path (str): Path to the model file.
            converter (strLabelConverter): Object to handle string and label conversion.
            output_size (tuple): Output size of the model.
        """
        self.model = bpu_infer_lib.Infer(False)
        if not self.model.load_model(model_path):
            raise RuntimeError(f"Failed to load model from '{model_path}'.")

        self.converter = converter
        self.output_size = output_size
        self.input_size = input_size

    def predict(self, img_path):
        """
        Perform prediction on an image.

        Args:
            img (np.array): Input image.
            img_path (str): Path to the image file.

        Returns:
            str: Raw prediction result.
            str: Simplified prediction result.
        """
        # Read the image and convert to NV12 format for inference
        self.model.read_img_to_nv12(img_path, 0)
        self.model.forward()
        
        # Get the model inference result and reshape it
        # preds = self.model.get_infer_res_np_float32(0, self.output_size[0] * self.output_size[1]).reshape(1, *self.output_size)
        preds = self.model.get_infer_res_np_float32(0).reshape(1, *self.output_size)
        print(preds.shape)
        
        # Transpose and get the argmax to obtain final prediction
        preds = np.transpose(preds, (1, 0, 2))
        preds = np.argmax(preds, axis=2)
        preds = preds.transpose(1, 0).reshape(-1)
        preds_size = np.array([preds.size], dtype=np.int32)
        raw_pred = self.converter.decode(np.array(preds), np.array(preds_size), raw=True)
        sim_pred = self.converter.decode(np.array(preds), np.array(preds_size), raw=False)
        return raw_pred,sim_pred
    
    def predict_float(self, img):
        """
        Perform prediction on an image.

        Args:
            img (np.array): Input image.
            img_path (str): Path to the image file.

        Returns:
            str: Raw prediction result.
            str: Simplified prediction result.
        """
        # Read the image and convert to float3212 format for inference
        image_resized = cv2.resize(img, dsize=(self.input_size[1], self.input_size[0]))
        image_resized = (image_resized / 255.0).astype(np.float32)
        input_image = np.zeros((image_resized.shape[0], image_resized.shape[1], 3), dtype=np.float32)
        input_image[:image_resized.shape[0], :image_resized.shape[1], :] = image_resized
        input_image = image_resized[:, :, [2, 1, 0]]  # bgr->rgb
        input_image = input_image[None].transpose(0, 3, 1, 2) # NHWC -> HCHW
        
        
        self.model.read_numpy_arr_float32(input_image, 0)
        self.model.forward(more=True)
        
        # Get the model inference result and reshape it
        # preds = self.model.get_infer_res_np_float32(0, self.output_size[0] * self.output_size[1]).reshape(1, *self.output_size)
        preds = self.model.get_infer_res_np_float32(0).reshape(1, *self.output_size)
        print("shape:", preds.shape)
        
        # Transpose and get the argmax to obtain final prediction
        preds = np.transpose(preds, (1, 0, 2))
        preds = np.argmax(preds, axis=2)
        preds = preds.transpose(1, 0).reshape(-1)
        preds_size = np.array([preds.size], dtype=np.int32)
        raw_pred = self.converter.decode(np.array(preds), np.array(preds_size), raw=True)
        sim_pred = self.converter.decode(np.array(preds), np.array(preds_size), raw=False)
        return raw_pred,sim_pred

def load_image(img_path):
    """Load an image from a file path."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    return img

def draw_bbox(img, bboxes, color=(128, 240, 128), thickness=3):
    """
    Draw bounding boxes on an image.

    Args:
        img: The input image.
        bboxes: A list of bounding boxes to draw.
        color: The color of the bounding boxes.
        thickness: The thickness of the bounding box lines.

    Returns:
        The image with bounding boxes drawn on it.
    """
    img_copy = img.copy()
    for bbox in bboxes:
        bbox = bbox.astype(int)
        cv2.polylines(img_copy, [bbox], isClosed=True, color=color, thickness=thickness)
    return img_copy

def crop_and_rotate_image(img, box):
    """Crop the image using the bounding box coordinates."""
    rect = cv2.minAreaRect(box) 
    box = cv2.boxPoints(rect).astype(np.intp)
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2]

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]],
                        dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    if angle >= 45:
        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    else: 
        rotated = warped
    
    print("width:", rotated.shape[1], "height:", rotated.shape[0])
    
    return rotated

def display_image(img, title="Image", output_path='output/predict.jpg'):
    """Display an image using Matplotlib and save it to a file.

    Args:
        img: The image to display.
        title: The title for the displayed image.
        output_path: The path where the image will be saved.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the image to a file
    cv2.imwrite(output_path, img)
    
    # Display the image using Matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    plt.title(title)
    plt.axis('off')
    plt.show()

def draw_text_on_image(img, texts, boxes, font_scale=0.7, color=(0, 0, 0), font_thickness=3):
    """Draw recognized texts on a white image based on bounding boxes."""
    for text, box in zip(texts, boxes):
        # Get the center of the bounding box
        center_x = int((box[0][0] + box[2][0]) / 2)
        center_y = int((box[0][1] + box[2][1]) / 2)
        
        # Calculate the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Calculate the position for the text to center it in the bounding box
        text_x = 0
        text_y = center_y + text_height // 2  # Use bottom of text

        # Draw the text on the image
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

def init_args():
    parser = argparse.ArgumentParser(description='paddleocr')
    parser.add_argument('--det_model_path', default='model/en_PP-OCRv3_det_640x640_nv12.bin', type=str)
    parser.add_argument('--rec_model_path', default='model/en_PP-OCRv3_rec_48x320_rgb.bin', type=str)
    parser.add_argument('--image_path', default='data/paddleocr_test.jpg', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='output/predict.jpg', type=str, help='img path for output')
    args = parser.parse_args()
    return args

def main():
    # Alphabet for OCR
    alphabet = """0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./  """
    converter = strLabelConverter(alphabet)

    # Load image
    img = load_image(args.image_path)

    # Initialize detection and recognition models
    detection_model = DetectionModel(args.det_model_path)
    recognition_model = rec_model(args.rec_model_path, converter)

    # Detect bounding boxes
    _, boxes_list = detection_model.predict(img, args.image_path)

    # Draw bounding boxes on the original image
    img_boxes = draw_bbox(img, boxes_list)
    display_image(img_boxes, title="Bounding Boxes", output_path=args.output_folder)

    # Recognize text in each detected box
    recognized_texts = []
    
    for i, box in enumerate(boxes_list):
        print(f"Box {i + 1}:")
        cropped_img = crop_and_rotate_image(img, box)  # Crop the image using the bounding box
        raw_pred, sim_pred = recognition_model.predict_float(cropped_img)
        recognized_texts.append(sim_pred)
        print(f"Raw Prediction: {raw_pred}")
        print(f"Simplified Prediction: {sim_pred} \n")
    
    # Create a white image of the same size as the original
    white_image = np.ones(img.shape, dtype=np.uint8) * 255

    # Draw recognized texts on the white image
    draw_text_on_image(white_image, recognized_texts, boxes_list, font_scale=3, color=(0, 0, 255), font_thickness=5)  # Red text

    # Combine original and white images for display
    combined_image = np.hstack((img_boxes, white_image))

    # Display the combined image
    display_image(combined_image, title="Original and Recognized Texts", output_path=args.output_folder)

if __name__ == '__main__':
    args = init_args()
    main()
