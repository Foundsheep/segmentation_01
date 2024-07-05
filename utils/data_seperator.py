from pathlib import Path
import random

def seperate_data(root, train_ratio, val_ratio, test_ratio, shuffle=True):
    failure = 0
    success = 1

    # check the ratio
    sum_ratio = train_ratio + val_ratio + test_ratio
    if str(sum_ratio) != "1.0" and str(sum_ratio) != "1":
        return failure
    
    # get the number of all files
    root_folder = Path(root)
    annotated_folder = root_folder / "annotated"
    raw_folder = root_folder / "raw"

    if not root_folder.exists() or not annotated_folder.exists() or not raw_folder.exists():
        print("necessary folders don't exist")
        return failure
    
    raw_files = list(raw_folder.glob("*"))
    num_files = len(raw_files)

    # get the numbers for each dataset
    num_train = round(num_files * train_ratio)
    num_val = round(num_files * val_ratio)
    num_test = num_files - num_train - num_val
    print(f"numbers of file to make: train=[{num_train}], val=[{num_val}], test=[{num_test}]")

    # prepare the directories for copies
    train_folder = root_folder / "train"
    val_folder = root_folder / "val"
    test_folder = root_folder / "test"

    # if seperated before
    if train_folder.exists():
        print("seperated data exist already!")
        return success

    annotated_files = list(annotated_folder.glob("*.png")) + list(annotated_folder.glob("*.jpg"))
    labelmap_txt = list(annotated_folder.glob("labelmap.txt"))[0]

    # make commonly necessary folders
    if num_train > 0:
        train_folder.mkdir()
        (train_folder / "annotated").mkdir()
        (train_folder / "raw").mkdir()

    if num_val > 0:
        val_folder.mkdir()
        (val_folder / "annotated").mkdir()
        (val_folder / "raw").mkdir()

    if num_test > 0:
        test_folder.mkdir()
        (test_folder / "annotated").mkdir()
        (test_folder / "raw").mkdir()

    # shuffle
    if shuffle:
        pass
        # for f in raw_files:
