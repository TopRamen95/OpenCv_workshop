import cv2 as cv
import numpy as np
import argparse
import os

def visualize(image, faces, thickness=2):
    for face in faces:
        coords = face[:-1].astype(np.int32)
        cv.rectangle(image, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
        for i in range(4, 14, 2):
            cv.circle(image, (coords[i], coords[i+1]), 2, (255, 0, 0), thickness)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def compute_accuracy(detected_faces, total_images):
    return (detected_faces / total_images) * 100

# Set up argument parser
parser = argparse.ArgumentParser(description='Face recognition script.')
parser.add_argument('-r', '--reference_image', required=True, help='Path to Aadhaar reference image (long or short version)')
parser.add_argument('-s', '--sample_images_folder', required=True, help='Folder containing sample images')

args = parser.parse_args()

# Read reference image (Aadhaar card)
reference_image = cv.imread(args.reference_image)
if reference_image is None:
    print(f"Error: Unable to open reference image at {args.reference_image}")
    exit(1)

# Load sample images
sample_images = load_images_from_folder(args.sample_images_folder)
if not sample_images:
    print(f"Error: No images found in folder {args.sample_images_folder}")
    exit(1)

# Initialize face detector and recognizer
score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000
faceDetector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (reference_image.shape[1], reference_image.shape[0]), score_threshold, nms_threshold, top_k)
recognizer = cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

# Detect face in reference image (Aadhaar card)
faceInAadhaar = faceDetector.detect(reference_image)
if faceInAadhaar[1] is None:
    print("No face detected in reference image.")
    exit(1)
visualize(reference_image, faceInAadhaar[1])
cv.imshow("Reference Face", reference_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Extract feature from Aadhaar card
face1_align = recognizer.alignCrop(reference_image, faceInAadhaar[1][0])
face1_feature = recognizer.feature(face1_align)

# Parameters for face matching
cosine_similarity_threshold = 0.363
l2_similarity_threshold = 1.128

detected_faces = 0
matched_faces = 0

# Process each sample image
for idx, query_image in enumerate(sample_images):
    faceDetector.setInputSize((query_image.shape[1], query_image.shape[0]))
    faceInQuery = faceDetector.detect(query_image)
    
    if faceInQuery[1] is not None:
        detected_faces += 1
        visualize(query_image, faceInQuery[1])
        cv.imshow(f"Query Face {idx + 1}", query_image)
        cv.waitKey(0)
        
        face2_align = recognizer.alignCrop(query_image, faceInQuery[1][0])
        face2_feature = recognizer.feature(face2_align)

        cosine_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
        l2_score = recognizer.match(face1_feature, face2_feature, cv.FACE_RECOGNIZER_SF_FR_NORM_L2)

        if cosine_score > cosine_similarity_threshold or l2_score < l2_similarity_threshold:
            matched_faces += 1
        
    cv.destroyAllWindows()

# Compute accuracy
detection_accuracy = compute_accuracy(detected_faces, len(sample_images))
matching_accuracy = compute_accuracy(matched_faces, detected_faces)

print(f"Detection Accuracy: {detection_accuracy}%")
print(f"Matching Accuracy: {matching_accuracy}%")
