import numpy as np
import sys


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

        np.true_divide(batch, 255., out=batch, casting='unsafe')

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

    return accuracy / num_episodes / num_classes


def blackbox_classification(test_data, num_shots, num_classes, classify,
                            num_episodes=10000, num_samples=100):
    data_dim = np.prod(test_data[0][0].shape)
    batch = np.zeros([num_samples * num_classes, num_shots + 1, data_dim], dtype=np.float32)

    accuracy = 0.

    for episode in xrange(num_episodes):
        classes = np.random.choice(test_data.shape[0], num_classes, False)

        classes_idx = classes[:, np.newaxis].repeat(num_shots, 1).repeat(num_samples, 0)
        test_classes_idx = classes[:, np.newaxis].repeat(num_samples, 0)

        idx = []
        for k in xrange(num_classes):
            idx.append(np.random.choice(test_data.shape[1], num_shots + 1, False))
        idx = np.vstack(idx)
        train_idx = idx[:, :-1].repeat(num_samples, 0)

        batch[:, :-1, :] = test_data[classes_idx, train_idx, :]

        for k in xrange(num_classes):
            batch[:, -1, :] = test_data[test_classes_idx, idx[k, -1], :].squeeze()
            actual_batch = batch / 255.
            # y_hat = classify(actual_batch)
            ll = classify(actual_batch)
            # print k, ll, '\n'
            y_hat = ll.argmax()
            if y_hat == k:
                accuracy += 1

        status = 'episode: %d, accuracy: %f' % (episode, accuracy / num_classes / (episode + 1))
        sys.stdout.write('\r' + status)

    return accuracy / num_episodes / num_classes
