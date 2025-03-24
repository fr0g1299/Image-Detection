import argparse
from image_detection import image_detection
import os


def main():
    """
    Detect faces and license plates in an image and draw rectangles around them.
    If edit is 'blur', blur the faces and license plates.
    If edit is 'black', draw black rectangles over them.
    Otherwise, draw green rectangles around them.
    """
    parser = argparse.ArgumentParser(
        description="A script to detect faces and license plates in images with optional editing features."
    )

    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input image file (supported formats: .jpg, .jpeg, .png)."
    )
    parser.add_argument(
        "-e", "--edit", 
        required=False, 
        choices=["black", "blur"], 
        help="Optional editing mode: 'black' to add a black bar over eyes/plates, or 'blur' to blur faces/license plates."
    )
    parser.add_argument(
        "-s", "--save", 
        required=False,
        action="store_true",
        help="Optional flag to save the processed image."
    )

    args = parser.parse_args()
    input = args.input
    edit = args.edit
    save = args.save

    # Check if the input file exists
    if not os.path.isfile(input):
        print(f"Error: The file '{input}' does not exist.")
        return

    valid_photo_formats = ['.jpg', '.jpeg', '.png']
    file_extension = os.path.splitext(input)[1].lower()

    if file_extension not in valid_photo_formats:
        print("Invalid file format. Please use one of the following formats:", ' '.join(valid_photo_formats))
        return

    image_detection(input, edit, save)


if __name__ == "__main__":
    main()