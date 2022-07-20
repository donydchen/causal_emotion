import sys
import os
from collections import OrderedDict


def load_data(root_dirs):
    raw_data = []
    for root_dir in root_dirs:
        cur_path = os.path.join(root_dir, 'emo_acc.txt')
        with open(cur_path, 'r') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines if x.strip()]
        line = lines[-1]  # take the newest test results
        line = line[line.find('Anger'):]  # remove prefix
        raw_data.append(line)

    return raw_data


def parse_data(raw_data):
    raw_dict = OrderedDict()
    for line in raw_data:
        items = [x.strip() for x in line.split('|') if x.strip()]
        for item in items:
            emo, num = item.split(':')
            emo = emo.strip()
            if emo not in raw_dict:
                raw_dict[emo] = [[], []]
            num = num[num.find('(') + 1:num.find(')')]
            num = num.split('/')
            raw_dict[emo][0].append(int(num[0]))
            raw_dict[emo][1].append(int(num[1]))

    return raw_dict


def main(root_dirs):
    assert len(root_dirs) == 3, "The number of fold should be 3, but gets only %d" % len(root_dirs)
    raw_data = load_data(root_dirs)
    raw_dict = parse_data(raw_data)

    all_list = []
    for emo, v in raw_dict.items():
        tp = sum(v[0])  # summarize number of true positive of three folds for specific emotion
        gt = sum(v[1])  # summarize number of all samples of three folds for specific emotion
        avg = float(tp) / float(gt)
        print("==> %-9s: %.6f" % (emo, avg))
        all_list.append(avg)

    # print(all_list)
    s = ["%.2f%%" % (x * 100) for x in all_list]
    print(' '.join(s))


if __name__ == '__main__':
    main(sys.argv[1:])
