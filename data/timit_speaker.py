import argparse
import os

from collections import defaultdict, OrderedDict

parser = argparse.ArgumentParser(description='Processes timit speaker.')
parser.add_argument('--target_dir', default='timit_dataset/', help='Path to save dataset')
args = parser.parse_args()


def process_manifests(lines, train_file, val_file, path):
    speakers = defaultdict(int)
    for line in range(len(lines)):
        audio = lines[line].split(',')[0]
        speaker = audio.strip(os.path.abspath(path)).split('/')[1]
        speakers[speaker] += 1
        if speakers[speaker] % 5 != 4:
            train_file.write(audio + "," + speaker + "\n")
        else:
            val_file.write(audio + "," + speaker + "\n")
    #print(OrderedDict(speakers))

def main():
    name = 'timit'
    train_path = args.target_dir + '/TRAIN/'
    test_path = args.target_dir + '/TEST/'
    with open(name + "_train_manifest_speaker.csv", 'w') as train_file:
        with open(name + "_val_manifest_speaker.csv", 'w') as val_file:
            with open(name + "_train_manifest.csv", 'r') as file:
                lines = file.readlines()
                process_manifests(lines, train_file, val_file, train_path)
            with open(name + "_val_manifest.csv", 'r') as file:
                lines = file.readlines()
                process_manifests(lines, train_file, val_file, test_path)


if __name__ == '__main__':
    main()
