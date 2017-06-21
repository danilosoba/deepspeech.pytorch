import argparse

from utils import create_manifest2

parser = argparse.ArgumentParser(description='Processes timit.')
parser.add_argument('--target_dir', default='timit_dataset/', help='Path to save dataset')
args = parser.parse_args()


def main():
    train_path = args.target_dir + '/TRAIN/'
    test_path = args.target_dir + '/TEST/'
    print ('\n', 'Creating manifests...')
    create_manifest2(train_path, 'timit_train')
    create_manifest2(test_path, 'timit_val')


if __name__ == '__main__':
    main()
