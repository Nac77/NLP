import sys
import os
import hw2_corpus_tool as inputtool


def sent2labels(utterance):
    finallist = []
    for labelutterance in utterance:
        finallist.append(labelutterance.act_tag)
    return finallist


def main():
    devdir = sys.argv[1]
    outputfile = sys.argv[2]
    correctcount = 0
    totalcount = 0
    output = open(outputfile, "r")
    content = output.read().splitlines()
    for root, dirs, files in os.walk(devdir):
        for filename in files:
            if filename.endswith(".csv"):
                var = 0
                filepath = os.path.abspath(os.path.join(root, filename))
                utterances = inputtool.get_utterances_from_filename(filepath)
                checklist = sent2labels(utterances)
                filenamevar = 'Filename="'+filename+'"'
                for x in range(0, len(content)):
                    if filenamevar == content[x]:
                        for y in range(x+1, len(checklist)+x+1):
                            if content[y] == checklist[var]:
                                correctcount += 1
                            totalcount += 1
                            var += 1
    print(correctcount)
    print(totalcount)
    print('Accuracy: ', float(correctcount/totalcount))

if __name__ == '__main__':
    main()