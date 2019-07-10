import face_model
import argparse
import cv2
import sys
import numpy as np

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
    img = cv2.imread(args.image)
    imgs, bbox, points = model.get_input(img)
    #feas = np.array([model.get_feature(img) for img in imgs])
    feas = model.get_feature(imgs)
    
    print(feas.shape)
    print(np.sum(np.square(feas[0] - feas[1])), np.dot(feas[0], feas[1]))
    #gender, age = model.get_ga(img)
    #print(gender)
    #print(age)
    sys.exit(0)
    img = cv2.imread('/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
    f2 = model.get_feature(img)
    dist = np.sum(np.square(f1-f2))
    print(dist)
    sim = np.dot(f1, f2.T)
    print(sim)
    #diff = np.subtract(source_feature, target_feature)
    #dist = np.sum(np.square(diff),1)
