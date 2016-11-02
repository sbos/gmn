import numpy as np
import sys
from utils import draw_episode


def cos_sim(x):
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    sim = np.dot(x, x.T)
    return sim


def one_shot_classification(test_data, num_shots, num_classes, compute_similarities, k_neighbours=1,
                            num_episodes=10000):
    data_shape = np.prod(test_data[0][0].shape)
    episode_length = num_shots * num_classes + 1
    batch = np.zeros([num_classes, episode_length, data_shape], dtype=np.float32)

    accuracy = 0.
    votes = np.zeros(num_classes)

    for episode in xrange(num_episodes):
        classes = np.random.choice(test_data.shape[0], num_classes, False)
        classes_idx = np.repeat(classes[:, np.newaxis], num_shots, 1).flatten()
        idx = []
        for k in xrange(num_classes):
            idx.append(np.random.choice(test_data.shape[1], num_shots + 1, False))
        idx = np.vstack(idx)
        y = np.repeat(np.arange(num_classes)[:, np.newaxis], num_shots, 1).flatten()

        # print batch[:, :-1, :].shape, idx[:, :-1].flatten().shape
        batch[:, :-1, :] = test_data[classes_idx, idx[:, :-1].flatten(), :]
        batch[:,  -1, :] = test_data[classes,     idx[:,  -1].flatten(), :]

        # np.true_divide(batch, 255., out=batch, casting='unsafe')

        # sim[i, j] -- similarity between batch[i, -1] and batch[i, j]
        sim = compute_similarities(batch)

        for k in xrange(num_classes):
            votes[:] = 0.
            nearest = sim[k].argsort()[-k_neighbours:]
            for j in nearest:
                votes[y[j]] += sim[k, j]
            y_hat = votes.argmax()
            if y_hat == k:
                accuracy += 1

        status = 'episode: %d, accuracy: %f' % (episode, accuracy / num_classes / (episode + 1))
        sys.stdout.write('\r' + status)
        sys.stdout.flush()

    return accuracy / num_episodes / num_classes


def blackbox_classification(test_data, num_shots, num_classes, classify,
                            num_episodes=10000, num_samples=100):
    data_dim = np.prod(test_data[0][0].shape)
    batch = np.zeros([num_samples, num_shots + 1, data_dim], dtype=np.float32)

    accuracy = 0.

    for episode in xrange(num_episodes):
        classes = np.random.choice(test_data.shape[0], num_classes, False)

        idx = []
        for k in xrange(num_classes):
            idx.append(np.random.choice(test_data.shape[1], num_shots + 1, False))
        idx = np.vstack(idx)

        def score(k):
            batch[:, -1, :] = test_data[classes[k], idx[k, -1]]
            for j in xrange(1, batch.shape[0]):
                batch[j] = batch[0]
            scores = np.zeros(num_classes)
            for c in xrange(num_classes):
                batch[:, :-1, :] = test_data[classes[c], idx[c, :-1]]
                scores[c] = classify(batch)
            return scores

        for k in xrange(num_classes):
            ll = score(k)
            y_hat = ll.argmax()
            if y_hat == k:
                accuracy += 1
            elif False:
                print classes[y_hat], classes[k]
                wrong = np.vstack([test_data[classes[y_hat], idx[y_hat, :-1]], test_data[classes[k], idx[k, -1]]])
                draw_episode(wrong)
                right = np.vstack([test_data[classes[k], idx[k, :-1]], test_data[classes[k], idx[k, -1]]])
                draw_episode(right)

        status = 'episode: %d, accuracy: %f' % (episode, accuracy / num_classes / (episode + 1))
        sys.stdout.write('\r' + status)
        sys.stdout.flush()

    return accuracy / num_episodes / num_classes
