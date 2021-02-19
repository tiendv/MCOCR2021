
def load_class_names(filename):
    with open(filename, 'r', encoding='utf8') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
