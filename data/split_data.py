import os
DATA_DIR = 'origin'
TRAIN_FILE = 'r8-train-all-terms.txt'
TEST_FILE = 'r8-test-all-terms.txt'
TRAID_DIR = 'train_txt'
TEST_DIR = 'test_txt'

if __name__=='__main__':
    train_file = []
    fp = open(os.path.join(DATA_DIR, TRAIN_FILE), 'r')
    labels = {}
    count = 0
    train_label = []
    train_file = []
    for lines in fp:
        label = lines.split()[0].strip()
        txt = lines.replace(label, '')
        if label not in labels:
            labels[label] = len(labels)
        count += 1
        # writing '#count.txt' file
        filename = str(count)+'.txt'
        fp_train = open(os.path.join(TRAID_DIR, filename), 'wb')
        train_file.append(filename)
        fp_train.write(txt)
        fp_train.close()
        # record #count label
        train_label.append(labels[label])
    fp_file = open('train_txt.txt', 'w')
    for file in train_file:
        fp_file.write(file + '\n')
    fp_file.close()
    fp_label = open('train_label.txt', 'w')
    for t in train_label:
        fp_label.write(str(t) + '\n')
    fp_label.close()

    fp.close()
    print(labels)
    fp = open(os.path.join(DATA_DIR, TEST_FILE), 'r')
    count = 0
    test_label = []
    test_file = []
    for lines in fp:
        label = lines.split()[0].strip()
        txt = lines.replace(label, '')
        count += 1
        # writing '#count.txt' file
        filename = str(count)+'.txt'
        fp_test = open(os.path.join(TEST_DIR, filename), 'wb')
        test_file.append(filename)
        fp_test.write(txt)
        fp_test.close()
        # record #count label
        test_label.append(labels[label])
    fp_file = open('test_txt.txt', 'w')
    for file in test_file:
        fp_file.write(file + '\n')
    fp_file.close()

    fp_label = open('test_label.txt', 'w')
    for t in test_label:
        fp_label.write(str(t) + '\n')
    fp_label.close()

    fp.close()

