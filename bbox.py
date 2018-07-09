from util import *

import matplotlib.pyplot as plt

for _ in range(5):
    img, id = get_img(ret_id=True)
    bbs = load_all_bboxs(id)

    _, ax = plt.subplots(1, 2)
    gray_imshow(ax[0], img)

    #corners = [0,1,2,3,0]
    #for i in range(len(corners)-1):
    #    plt.plot(bbs[:,corners[i]], bbs[:,corners[i+1]], c='r')
    for bb in bbs:
        ax[0].plot([bb[1], bb[3]], [bb[0], bb[0]], c='r')
        ax[0].plot([bb[1], bb[3]], [bb[2], bb[2]], c='r')
        ax[0].plot([bb[1], bb[1]], [bb[0], bb[2]], c='r')
        ax[0].plot([bb[3], bb[3]], [bb[0], bb[2]], c='r')

    ax[0].show()
