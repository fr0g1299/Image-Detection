import cv2
import easyocr
import matplotlib.pyplot as plt

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], verbose=False)

# Load the Haar cascades for face, eyes, and license plate detection
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
plate_cascade = cv2.CascadeClassifier('./cascades/haarcascade_russian_plate_number.xml')

# Function to detect faces and license plates in a photo
def detect_faces_in_photo(img, edit, faces, gray):
    for (x, y, w, h) in faces:
        if edit == 'blur':
            # Blur the face
            face_region = img[y:y+h, x:x+w]
            face_region = cv2.GaussianBlur(face_region, (199, 199), 50)
            img[y:y+h, x:x+w] = face_region
        elif edit == 'black':
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                # Calculate the bounding box for all eyes
                ex_min = min([ex for (ex, ey, ew, eh) in eyes])
                ey_min = min([ey for (ex, ey, ew, eh) in eyes])
                ex_max = max([ex + ew for (ex, ey, ew, eh) in eyes])
                ey_max = max([ey + eh for (ex, ey, ew, eh) in eyes])

                # Draw a black rectangle
                cv2.rectangle(roi_color, (ex_min, ey_min), (ex_max, ey_max), (0, 0, 0), -1)

        else:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(roi_color, 'Eye', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

def detect_plates_in_photo(img, edit, plates):
    for (x, y, w, h) in plates:
        if edit == 'blur':
            # Blur the license plate
            plate_region = img[y:y+h, x:x+w]
            plate_region = cv2.GaussianBlur(plate_region, (199, 199), 50)
            img[y:y+h, x:x+w] = plate_region

        elif edit == 'black':
            # Draw a black rectangle
            plate_region = img[y:y+h, x:x+w]
            cv2.rectangle(plate_region, (0, 0), (plate_region.shape[1], plate_region.shape[0]), (0, 0, 0), -1)
            img[y:y+h, x:x+w] = plate_region

        else:
            # Draw rectangle around the license plate
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, 'License Plate', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Extract the license plate region
            plate_region = img[y:y+h, x:x+w]

            # Use EasyOCR to extract text
            result = reader.readtext(plate_region)
            if result:
                plate_text = result[0][-2]  # Extract detected text
                cv2.putText(img, plate_text.strip(), (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

def image_detection(image_path, edit, save):
    """
    Detect faces and license plates in an image and draw rectangles around them.
    If edit is 'blur', blur the faces and license plates.
    If edit is 'black', draw black rectangles over them.
    Otherwise, draw green rectangles around them.

    Parameters
    ----------
    image_path : str
        Path to the input image file
    edit : str
        Optional editing mode: 'black' to add a black bar over eyes/plates,
        or 'blur' to blur faces/license plates

    Returns
    -------
    None
    """
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        # Detect license plates
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    except:
        print("Error")

    if len(faces) > 0:
        img = detect_faces_in_photo(img, edit, faces, gray)
    if len(plates) > 0:
        img = detect_plates_in_photo(img, edit, plates)

    # Convert BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

    # Save the processed image
    if save:
        cv2.imwrite('processed_image.jpg', img)
