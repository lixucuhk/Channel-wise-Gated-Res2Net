import argparse
import os

os.makedirs('scoring/class_far', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--type', action='store', type=str, default='LA')
parser.add_argument('--score-file', action='store', type=str, default='scoring/cm_scores/SEGatedLinearRes2Net5026w4s4rLACQT0-epoch20-dev_scores.txt')
parser.add_argument('--threshold', action='store', type=float)
args = parser.parse_args()

if args.type == 'LA':
    class_labels = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
elif args.type == 'PA':
    class_labels = ['-', 'AA', 'AB', 'AC', 'BA', 'BB', 'BC', 'CA', 'CB', 'CC']

class_correct = [0 for i in range(len(class_labels))]
class_total = [0 for i in range(len(class_labels))]
threshold = args.threshold
scorefile = args.score_file
wfile = scorefile+'.anls'
wfile = wfile.replace('cm_scores', 'class_far')

with open(scorefile, 'r') as rf:
    for line in rf.readlines():
        uttid, audio_category, _, score = line.split()
        class_index = class_labels.index(audio_category)
        score = float(score)
        if audio_category == '-':
            if score >= threshold:
                class_correct[class_index] += 1
        else:
            if score < threshold:
                class_correct[class_index] += 1
        class_total[class_index] += 1


with open(wfile, 'w') as wf:
    wf.write("===> Results for each class\n")
    for i in range(len(class_labels)):
        if class_total[i] != 0:
            wf.write('Accuracy of %8s : %6f %%\n' % (class_labels[i], 100. * class_correct[i] / class_total[i]))

