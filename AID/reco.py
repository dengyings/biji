import os
import sys
import json
import numpy as np


def calc_ps(ratings, user1, user2):
    movies = set()
    for movie in ratings[user1]:
        if movie in ratings[user2]:
            movies.add(movie)
    n = len(movies)
    if n == 0:
        return 0
    x = np.array([ratings[user1][move] for move in movies])
    y = np.array([ratings[user2][move] for move in movies])
    sx = x.sum()
    sy = y.sum()
    xx = (x**2).sum()
    yy = (y**2).sum()
    xy = (x * y).sum()
    sxx = xx - sx**2 / n
    syy = yy - sy**2 / n
    sxy = xy - sx * sy / n
    if sxx * syy == 0:
        return 0
    pearson_score = sxy / np.sqrt(sxx * syy)
    return pearson_score


def read_data(filename):
    with open(filename, 'r') as f:
        ratings = json.loads(f.read())
    return ratings


def eval_ps(ratings):
    users, psmat = list(ratings.keys()), []
    for user1 in users:
        psrow = []
        for user2 in users:
            psrow.append(calc_ps(ratings, user1, user2))
        psmat.append(psrow)
    users = np.array(users)
    psmat = np.array(psmat)
    return users, psmat


def find_similars(users, psmat, user, n_similars=None):
    user_index = np.arange(len(users))[users == user][0]
    sorted_indices = psmat[user_index].argsort()[::-1]
    similar_indices = sorted_indices[
        sorted_indices != user_index][:n_similars]
    similar_users = users[similar_indices]
    similar_scores = psmat[user_index][similar_indices]
    return similar_users, similar_scores


def calc_reco(ratings, user):
    users, psmat = eval_ps(ratings)
    similar_users, similar_scores = find_similars(
        users, psmat, user)
    positive_mask = similar_scores > 0
    similar_users = similar_users[positive_mask]
    similar_scores = similar_scores[positive_mask]
    score_sums, weight_sums = {}, {}
    for i, similar_user in enumerate(similar_users):
        for movie, score in ratings[similar_user].items():
            if movie not in ratings[user].keys() or \
                    ratings[user][movie] == 0:
                if movie not in score_sums.keys():
                    score_sums[movie] = 0
                score_sums[movie] += score * similar_scores[i]
                if movie not in weight_sums.keys():
                    weight_sums[movie] = 0
                weight_sums[movie] += similar_scores[i]
    movie_ranks = {movie: score_sum / weight_sums[movie]
                   for movie, score_sum in score_sums.items()}
    sorted_indices = np.array(
        list(movie_ranks.values())).argsort()[::-1]
    reco = np.array(
        list(movie_ranks.keys()))[sorted_indices]
    return reco


def main(argc, argv, envp):
    ratings = read_data('ratings.json')
    for user in ratings.keys():
        reco = calc_reco(ratings, user)
        print('{}: {}'.format(user, reco))
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
