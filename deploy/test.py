import face_model
import argparse
import cv2
import sys
import numpy as np
from face_db import FaceDB

def get_image_fea(model, image):
    img = cv2.imread(args.image)
    imgs, bbox, points = model.get_input(img)
    #feas = np.array([model.get_feature(img) for img in imgs])
    feas = model.get_feature(imgs)

    return feas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--image', default=['Tom_Hanks_54745.png'], type=str, nargs='+', help='image path')
    args = parser.parse_args()
    
    model = face_model.FaceModel(args)
    total_feas = []
    face_images = []

    db = FaceDB('hehe')
    for image in args.image:
        img = cv2.imread(image)
        max_size = max(img.shape)
        scale = 768. / max_size
        img = cv2.resize(img, None, fx=scale, fy=scale)
        imgs, bbox, points = model.get_input(img)
        face_images.append(imgs)
        #feas = np.array([model.get_feature(img) for img in imgs])
        feas = model.get_feature(imgs)
        
        print(feas.shape)
        print(bbox)
        db.check({'fn': image}, bbox, points, feas, imgs)
#        db.show()
        
#        print(np.sum(np.square(feas[0] - feas[1])))
#        print(np.dot(feas, feas.T))
#        for b in bbox:
#            bb = b.astype(np.int).tolist()
#            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), 255, 2)
        #cv2.imshow("bbox", img)
        #cv2.waitKey()
#        total_feas.append(feas)
#    total_feas = np.concatenate(total_feas, axis=0)
#    face_images = np.concatenate(face_images, axis=0)
#    dist = np.dot(total_feas, total_feas.T) * (1. - np.eye(total_feas.shape[0]))
#    print(np.where(dist > 0.5))
#    for x,y in np.array(np.where(dist > 0.5)).T:
#        print(x,y)
#        if x < y:
#            continue
#        cv2.imshow("l", face_images[x].transpose((1,2,0)))
#        cv2.imshow("r", face_images[y].transpose((1,2,0)))
#        print(dist[x,y])
#        cv2.waitKey()
        #gender, age = model.get_ga(img)
        #print(gender)
        #print(age)

