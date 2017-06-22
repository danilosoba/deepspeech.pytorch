import argparse
import os

parser = argparse.ArgumentParser(description='Processes timit speaker.')
parser.add_argument('--target_dir', default='timit_dataset/', help='Path to save dataset')
args = parser.parse_args()


def process_manifests(lines, train_file, val_file, path):
    speakers = {}
    label = 1
    for line in range(len(lines)):
        audio = lines[line].split(',')[0]
        speaker = audio.strip(os.path.abspath(path)).split('/')[1]
        if speaker in speakers:
            speakers[speaker][1] += 1
        else:
            speakers[speaker] =[label,1]
            label += 1
        if speakers[speaker][1] % 5 != 4:
            train_file.write(audio + "," + str(speakers[speaker][0]) + "\n")
        else:
            val_file.write(audio + "," + str(speakers[speaker][0]) + "\n")


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
