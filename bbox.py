from util import *

import matplotlib.pyplot as plt

for _ in range(5):
    img, id = get_img(ret_id=True)
    bbs = load_all_bboxs(id)
    gray_imshow(plt, img)

    #corners = [0,1,2,3,0]
    #for i in range(len(corners)-1):
    #    plt.plot(bbs[:,corners[i]], bbs[:,corners[i+1]], c='r')
    for bb in bbs:
        plt.plot([bb[1], bb[3]], [bb[0], bb[0]], c='r')
        plt.plot([bb[1], bb[3]], [bb[2], bb[2]], c='r')
        plt.plot([bb[1], bb[1]], [bb[0], bb[2]], c='r')
        plt.plot([bb[3], bb[3]], [bb[0], bb[2]], c='r')

    plt.show()
