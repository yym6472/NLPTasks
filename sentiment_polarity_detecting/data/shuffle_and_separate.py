"""
This script will read `.pos` and `.neg` files, shuffle the positive
samples and negative samples together, and separate these shuffled samples
into (train, valid, test) dataset.

Note that in output files, one line contains only one sample, whose format
is as follows:

    `"[token1] [token2] ... [tokenN]\t[label]"`

in which `[label]` should be either "pos" for positive samples or "neg" for
negative samples.
"""


import random
random.seed(1)

def shuffle_and_separate(pos_file_path: str,
                         neg_file_path: str,
                         train_output_path: str,
                         valid_output_path: str,
                         test_output_path: str,
                         separate_rates: str = "8:1:1") -> None:
    lines = []
    with open(pos_file_path, "r", encoding="utf-8") as f_in:
        for pos_line in f_in:
            if not pos_line.strip():  # for empty lines
                continue
            lines.append(pos_line.strip() + "\tpos\n")
    with open(neg_file_path, "r", encoding="utf-8") as f_in:
        for neg_line in f_in:
            if not neg_line.strip():  # for empty lines
                continue
            lines.append(neg_line.strip() + "\tneg\n")
    
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
        "pos_file_path": "./data/For Detecting sentiment polarity-Data 2/rt-polaritydata/rt-polarity.pos",
        "neg_file_path": "./data/For Detecting sentiment polarity-Data 2/rt-polaritydata/rt-polarity.neg",
        "train_output_path": "./data/train.txt",
        "valid_output_path": "./data/valid.txt",
        "test_output_path": "./data/test.txt",
        "separate_rates": "8:1:1"
    }
    shuffle_and_separate(**kwargs)