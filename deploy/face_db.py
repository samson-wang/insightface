from __future__ import print_function
import cv2
import numpy as np

class FaceDB(object):
    '''
    Face DataBase.
    To store, face id / feature vector, image list, face box and points.
    '''
    def __init__(self, db_path):
        '''
        Initialization
        '''
        # person's feature, (P x (N+1))
        self.fea_vector = None
        # map bbox to person and image, (B x (4 + 1))
        self.bbx = None
        # points, (B x pts)
        self.points = None
        # Inverse Map from person to bbx, (P x List)
        self.inv_map = None
        # Avatar image of bbox, (B x 64 x 64 x 3)
        self.avatar = None
        # Image list, (I x 1)
        self.image_list = None

        self.p_num = 0
        self.bb_num = 0
        self.i_num = 0

    def show(self):
        print("Unique Persons {}-{}" . format(self.p_num, self.fea_vector.shape))
        print("Images {}, bbox {}, avatars {}" . format(len(self.image_list), self.bbx.shape, self.avatar.shape))

        for k, v in enumerate(self.inv_map):
            print("{}'th person - {} boxes" . format(k, len(v)))
            for kk, idx in enumerate(v):
                cv2.imshow("{}-{}" . format(k, kk), self.avatar[idx])
            cv2.waitKey()

    def check(self, image_info, boxes, points, feas, avatar):
        '''
        To check and update facedb.
        '''
        if self.fea_vector is None:
            b_idx = np.arange(boxes.shape[0], dtype=boxes.dtype).reshape((-1,1))
            self.fea_vector = np.concatenate((feas.copy(), b_idx), axis=1)
            self.image_list = [image_info['fn']]
            self.inv_map = [set([k]) for k in range(feas.shape[0])]

            self.bbx = np.concatenate((boxes, b_idx, np.zeros((boxes.shape[0],1), dtype=boxes.dtype)), axis=1)

            self.avatar = avatar.transpose(0,2,3,1).copy()
            self.points = points.copy()
            self.p_num = len(self.fea_vector)
            self.bb_num = len(self.bbx)
            self.i_num = 1
        else:
            if image_info['fn'] in self.image_list:
                return
            dist = np.dot(self.fea_vector[:, :128], feas.T)
            max_idx = np.argmax(dist, axis=0)
            for fi, pi in enumerate(max_idx):
                cur_bb_idx = self.bb_num + fi
                tmp_fea = np.concatenate((feas[fi], np.array([cur_bb_idx],dtype=feas.dtype)), axis=0).reshape((1,-1))
                if dist[pi, fi] < 0.5:
                    print("New person", self.fea_vector.shape, tmp_fea.shape)
                    self.fea_vector = np.concatenate((self.fea_vector, tmp_fea), axis=0)
                    self.inv_map.append(set([cur_bb_idx,]))
                    self.p_num += 1
                else:
                    self.inv_map[pi].add(cur_bb_idx)
                    cur_p_num = len(self.inv_map[pi])
                    self.fea_vector[pi][:128] = (cur_p_num / (cur_p_num + 1.)) * self.fea_vector[pi][:128] + (1 / (cur_p_num + 1.)) * feas[fi]
            tmp_bbx = np.concatenate((boxes, np.arange(self.bb_num, self.bb_num+boxes.shape[0], dtype=boxes.dtype).reshape((-1,1)), np.ones((boxes.shape[0],1), dtype=boxes.dtype) * self.i_num), axis=1)
            self.bbx = np.concatenate((self.bbx, tmp_bbx), axis=0)
            self.avatar = np.concatenate((self.avatar, avatar.transpose(0,2,3,1).copy()), axis=0)
            self.points = np.concatenate((self.points, points), axis=0)
            self.bb_num = len(self.bbx)
            self.image_list.append(image_info['fn'])
            self.i_num += 1
            pass

