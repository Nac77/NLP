import sys
import os
import pycrfsuite
import hw2_corpus_tool as inputtool

def sent2features(utterance):
    speakerlist = []
    finallist = []
    for featureutterance in utterance:
        tokenlist = []
        poslist = []
        xfilelist = []
        speakerlist.append(featureutterance.speaker)
        if len(speakerlist) > 1 and speakerlist[-2] == speakerlist[len(speakerlist)-1]:
            speaker = '0'
        else:
            speaker = '1'
        if len(speakerlist) == 1:
            first = '1'
        else:
            first = '0'
        if not featureutterance.pos:
            tokenlist.append('token_NONE')
            poslist.append('pos_NONE')
            xfilelist.append(speaker)
            xfilelist.append(first)
            for token in tokenlist:
                xfilelist.append(token)
            for pos in poslist:
                xfilelist.append(pos)
        else:
            for x in featureutterance.pos:
                tokenlist.append('token_'+x.token)
                poslist.append('pos_'+x.pos)
            xfilelist.append(speaker)
            xfilelist.append(first)
            for token in tokenlist:
                xfilelist.append(token)
            for pos in poslist:
                xfilelist.append(pos)
        finallist.append(xfilelist)
    return finallist


def sent2labels(utterance):
    finallist = []
    for labelutterance in utterance:
        finallist.append(labelutterance.act_tag)
    return finallist


def main():
    inputdir = sys.argv[1]
    testdir = sys.argv[2]
    outputfile = sys.argv[3]
    x_list = []
    y_list = []
    for root, dirs, files in os.walk(inputdir):
        for filename in files:
            if filename.endswith(".csv"):
                filepath = os.path.abspath(os.path.join(root, filename))
                utterances = inputtool.get_utterances_from_filename(filepath)
                x_train = sent2features(utterances)
                y_train = sent2labels(utterances)
                for x in x_train:
                    x_list.append(x)
                for y in y_train:
                    y_list.append(y)
    
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.append(x_list, y_list)
    trainer.set_params({
        'c1': 1,
        'c2': 1e-3,
        'max_iterations': 85,
        'feature.possible_states': True,
        'feature.possible_transitions': True
    })
    trainer.train('baseline.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('baseline.crfsuite')
    f = open(outputfile, "a")
    f.truncate(0)
    for root, dirs, files in os.walk(testdir):
        for filename in files:
            if filename.endswith(".csv"):
                filepath = os.path.abspath(os.path.join(root, filename))
                utterances = inputtool.get_utterances_from_filename(filepath)
                x_tag = sent2features(utterances)
                outputlist = tagger.tag(x_tag)
                f.write('Filename="')
                f.write(filename)
                f.write('"')
                f.write('\n')
                for y in outputlist:
                    f.write(y)
                    f.write('\n')
                f.write('\n')
    f.close()


if __name__ == '__main__':
    main()
