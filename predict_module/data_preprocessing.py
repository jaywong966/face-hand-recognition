import numpy as np

class data_preprocessing():
    def __init__(self):
        self.point_pairs = [[0, 1], [1, 2], [2, 3], [3, 4],
                            [0, 5], [5, 6], [6, 7], [7, 8],
                            [0, 9], [9, 10], [10, 11], [11, 12],
                            [0, 13], [13, 14], [14, 15], [15, 16],
                            [0, 17], [17, 18], [18, 19], [19, 20]]

    def __call__(self, right_hand, left_hand):
        get_hand = []
        if right_hand.size == 1:
            return np.zeros((2,20,2), dtype='float32')
        get_hand.append(right_hand)
        get_hand.append(left_hand)
        get_hand = self.unit_vector(get_hand)
        x = get_hand[0].flatten()
        get_hand = x[np.newaxis, :]
        return get_hand

    def unit_vector(self,hand):
        vector = np.ones((2, 20, 2), dtype='float32')
        for i in range(0,2):
            for j in range(0,20):
                vector[i][j][0] = hand[i][self.point_pairs[j][0]][0] - hand[i][self.point_pairs[j][1]][0]   # convert into vector
                vector[i][j][1] = hand[i][self.point_pairs[j][0]][1] - hand[i][self.point_pairs[j][1]][1]
                norm = pow(pow(vector[i][j][0], 2) + pow(vector[i][j][1], 2), 0.5)
                if norm!=0:
                    norm = 1/norm
                vector[i][j][0] = norm * vector[i][j][0]    # convert into unit vector
                vector[i][j][1] = norm * vector[i][j][1]
        return vector

