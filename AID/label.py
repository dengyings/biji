import os
import sys
import numpy as np
import sklearn.preprocessing as sp


def main(argc, argv, envp):
    raw_labels = np.array([
        'audi', 'ford', 'audi', 'toyota', 'ford', 'bmw',
        'toyota', 'ford', 'audi'])
    print(raw_labels)
    codec = sp.LabelEncoder()
    enc_labels = codec.fit_transform(raw_labels)
    print(enc_labels)
    dec_labels = codec.inverse_transform(enc_labels)
    print(dec_labels)
    return 0

if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
