from __future__ import print_function


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
        # Inverse map bbox to person and image, (B x (1 + 1))
        self.inv_map = None
        # Avatar image of bbox, (B x 64 x 64 x 3)
        self.avatar = None
        # Image list, (I x 1)
        self.image_list = None
       
    def check(self, image_info, boxes, points, feas):
        '''
        To check and update facedb.
        '''
        if self.fea_vector is None:
            self.fea_vector = feas.copy()
            self.image_list = [image_info['fn']]
            self.inv_map = dict(zip(range(feas.shape[0]), range(feas.shape[0])))
        
