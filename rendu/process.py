"""
Tested with Python 3.8.5

Sample processing tool:

- Takes a directory as input
- and a path to the output file to produce
- then classifies each image
- and produces the correct CSV output

=> Just update the "MyClassifier" methods!
"""

from argparse import ArgumentParser
import os
import os.path
import pdb
from typing import Dict, List
from src.main import feature_extractor

# You may use the following imports. Please warn me if you need something else.
import numpy as np
# import sklearn
import cv2
# import skimage
import pickle


class MyClassifier:
    """Sample classifier class to update.
    """

    def __init__(self) -> None:
        self.clf = pickle.load(open("assets/clf.pkl", "rb"))

    def classify_image(self, image_path: str) -> int:
        """Classify the file for which the path is given,
        returning the correct class as output (1-56) or -1 to reject

        Args:
            image_path (str): Full path to the image to process

        Returns:
            int: Class of the symbol contained (1-56 or -1).

        WARNING: `0` is not a valid class here.
                 You may need to adjust your classifier outputs (typically 0-55).
        """
        ponderation = 0.5
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        color_feature_extractor, shape_feature_extractor = feature_extractor(
            [img], load_session=True)

        def fused_feature_extractor(x):
            return np.hstack((color_feature_extractor(x) * ponderation, shape_feature_extractor(x) * (1 - ponderation)))
            
        return self.clf.predict(fused_feature_extractor([img]))[0]

        # cls_id = ((int(image_path[-6:-4]) * 991) % 56) + 1
        # return cls_id


def save_classifications(image_classes: Dict[str, int], output_path: str):
    """Save classification results to a CSV file

    Args:
        image_classes (Dict[str,int]): dict of base filename -> class id
        output_path (str): will store a CSV in the following format:
    ```csv
    filename,class_id
    filename,class_id
    (no header, accept any leading 0 for class_id)
    ```
    """
    with open(output_path, 'w', encoding="utf8") as out_file:
        for filename, cls_id in image_classes.items():
            out_file.write(f"{filename},{cls_id:02d}\n")


def find_png_files_in_dir(input_dir_path: str) -> List[str]:
    """Returns a list of PNG files contained in a directory, without any leading directory.

    Args:
        input_dir_path (str): Path to directory

    Returns:
        List[str]: List of PNG files (no leading directory).
    """
    with os.scandir(input_dir_path) as entry_iter:
        result = [entry.name for entry in entry_iter if entry.name.endswith(
            '.png') and entry.is_file()]
        return result


def main():
    """Main function."""
    # CLI
    parser = ArgumentParser(description="Sample processing program.")
    parser.add_argument("--test_dir", required=True,
                        help="Path to the directory containing the test files.")
    parser.add_argument("--output", required=True,
                        help="Path to output CSV file.")
    args = parser.parse_args()

    # Find files
    files = find_png_files_in_dir(args.test_dir)

    # Create classifier
    clf = MyClassifier()

    # Let's go
    results = {}
    print("Processing files...")
    for file in files:
        print(file)
        file_full_path = os.path.join(args.test_dir, file)
        cls_id = -1
        try:
            cls_id = clf.classify_image(file_full_path)
            print(f"\t-> {cls_id}")
        except:
            print(f"Warning: failed to process file '{file_full_path}'.")
            raise
        results[file] = cls_id
    print("Done processing files.")

    # Save predictions
    save_classifications(results, args.output)
    print(f"Predictions saved in '{args.output}'.")


if __name__ == "__main__":
    main()
