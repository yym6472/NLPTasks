"""
Shuffle lines, and separate these shuffled lines into (train, valid, test) dataset.
"""


import random
random.seed(1)

def shuffle_and_separate(source_file_path: str,
                         train_output_path: str,
                         valid_output_path: str,
                         test_output_path: str,
                         separate_rates: str = "8:1:1") -> None:
    with open(source_file_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()
    random.shuffle(lines)
    r_train, r_valid, r_test = [int(rate) for rate in separate_rates.split(":")]
    levels = [r_train, r_train + r_valid, r_train + r_valid + r_test]
    r_total = levels[2]
    levels = [x * len(lines) // r_total for x in levels]
    train_lines = lines[0:levels[0]]
    valid_lines = lines[levels[0]:levels[1]]
    test_lines = lines[levels[1]:levels[2]]
    with open(train_output_path, "w", encoding="utf-8") as f_out:
        f_out.write("".join(train_lines))
    with open(valid_output_path, "w", encoding="utf-8") as f_out:
        f_out.write("".join(valid_lines))
    with open(test_output_path, "w", encoding="utf-8") as f_out:
        f_out.write("".join(test_lines))


if __name__ == "__main__":
    kwargs = {
        "source_file_path": "./data/source.txt",
        "train_output_path": "./data/train.txt",
        "valid_output_path": "./data/valid.txt",
        "test_output_path": "./data/test.txt",
        "separate_rates": "8:1:1"
    }
    shuffle_and_separate(**kwargs)