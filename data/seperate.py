# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a loaded document into sentences
def to_sentences(doc):
    return doc.strip().split('\n')

def seperate(lines):
    b = []
    e = []
    for line in lines:
        if "\t" in line:
            bo, en = line.split("\t")
            if bo and en:
                b.append(bo)
                e.append(en)
    return b, e

    
# save a list of clean sentences to file
def save_to_file(sentences, filename):
    with open(filename, 'w') as filehandle:
        filehandle.writelines("%s\n" % sentence for sentence in sentences)
    print('Saved: %s' % filename)
    
if __name__ == "__main__":
    doc = load_doc("bo-en.txt")
    sentences = to_sentences(doc)
    bo, en = seperate(sentences)
    save_to_file(bo, "bo.txt")
    save_to_file(en, "en.txt")