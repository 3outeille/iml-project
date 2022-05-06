
# Submission format:
# filename,class_id
# filename,class_id
# (no header, accept any leading 0 for class_id)

# Ground truth format (reversed, yes this is stupid):
# 43,c18_s04.png
# 43,c27_s07.png

from argparse import ArgumentParser
import os.path


def _load_file(path: str, FILENAME_FID: int, CLSID_FID: int):
    if not os.path.isfile(path):
        raise ValueError(f"'{path} does not exist or is not a file. Expected GT file.")
    filenames_to_clsid = {}

    all_lines = None
    with open(path) as in_file:
        all_lines = in_file.readlines()

    for line in all_lines:
        if len(line) < 4 or line.startswith("#"):
            continue
        fields = line.strip().split(",")
        if len(fields) != 2:
            continue
        try:
            filename = fields[FILENAME_FID]
            clsid = fields[CLSID_FID]
            clsid = int(clsid)
        except Exception as e:
            print(e.with_traceback)
            print(f"Problematic line: '{line}'")
            raise e
        filenames_to_clsid[filename] = clsid
    return filenames_to_clsid


def load_gt(path: str):
    FILENAME_FID = 1
    CLSID_FID = 0
    return _load_file(path, FILENAME_FID, CLSID_FID)


def load_pred(path: str):
    FILENAME_FID = 0
    CLSID_FID = 1
    return _load_file(path, FILENAME_FID, CLSID_FID)


def main():
    # CLI
    parser = ArgumentParser(description="Auto correct submissions from students.")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth file")
    parser.add_argument("--submission", required=True, help="Path to submission file")
    args = parser.parse_args()
    # print(f"GT file: {args.ground_truth}")
    # print(f"PRED file: {args.submission}")
    
    # Load files
    gt = load_gt(args.ground_truth)
    pred = load_pred(args.submission)

    # Identify cases
    MISSING_VAL = -2
    REJECT_VAL = -1
    corrects = 0
    errors = 0
    rejects = 0
    for filename in gt:
        expected_value = gt[filename]
        predicted_value = pred.get(filename, MISSING_VAL)
        if predicted_value == REJECT_VAL:
            rejects += 1
        elif predicted_value == expected_value:
            corrects += 1
        else:
            errors += 1
    
    # Sanity check
    N = len(gt)
    assert(N == (corrects + rejects + errors))

    # Compute final score
    REJECT_COST = 0.5
    ERROR_COST = 1
    score = 1. - 1./N * (REJECT_COST * rejects + ERROR_COST * errors)
    print(f"Score: {score*100:5.2f}%; c={corrects}; r={rejects}; e={errors}; N={N}")

if __name__ == "__main__":
    main()
