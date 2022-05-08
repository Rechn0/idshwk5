from sklearn.ensemble import RandomForestClassifier


class Domain:
    def __init__(self, _name, _label):
        self.name = _name
        self.label = _label
        self.domain_len = len(_name)
        sum_of_num = 0
        for c in _name:
            if c.isdigit():
                sum_of_num += int(c)
        self.sum_of_num = sum_of_num

    def returnData(self):
        return [self.domain_len, self.sum_of_num]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1


def initTrainData(filename):
    domain_list = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            domain_list.append(Domain(name, label))
            return domain_list


def initTestData(filename):
    domain_list = []
    feature_list = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            num = 0
            for c in line:
                if c.isdigit():
                    num += int(c)
            testData = [len(line), num]
            domain_list.append(line)
            feature_list.append(testData)
    return domain_list, feature_list


def output(content, filename):
    with open(filename, "w") as f:
        for i in content:
            f.write(i + "\n")


def main():
    train_set = initTrainData("train.txt")
    test_domain_name, test_feature_set = initTestData("test.txt")
    feature_matrix = []
    label_list = []
    for item in train_set:
        feature_matrix.append(item.returnData())
        label_list.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(feature_matrix, label_list)
    classification = clf.predict(test_feature_set)
    result = []
    for i in range(len(classification)):
        if classification[i] == 1:
            result.append(test_domain_name[i] + ",dga")
        else:
            result.append(test_domain_name[i] + ",notdga")
    output(result, "result.txt")


if __name__ == '__main__':
    main()
