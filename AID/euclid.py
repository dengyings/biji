import os
import sys
import json
import numpy as np


def calc_es(ratings, user1, user2):
    movies = set()
    for movie in ratings[user1]:
        if movie in ratings[user2]:
            movies.add(movie)
    if len(movies) == 0:
        return 0
    diffs = []
    for move in movies:
        diffs.append(
            np.square(ratings[user1][move] - ratings[user2][move]))
    diffs = np.array(diffs)
    euclidean_score = 1 / (1 + np.sqrt(diffs.sum()))
    return euclidean_score


def read_data(filename):
    with open(filename, 'r') as f:
        ratings = json.loads(f.read())
    return ratings


def eval_es(ratings):
    users, esmat = list(ratings.keys()), []
    for user1 in users:
        esrow = []
        for user2 in users:
            esrow.append(calc_es(ratings, user1, user2))
        esmat.append(esrow)
    users = np.array(users)
    esmat = np.array(esmat)
    return users, esmat


def main(argc, argv, envp):
    ratings = read_data('ratings.json')
    users, esmat = eval_es(ratings)
    print(users)
    print(esmat)
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
