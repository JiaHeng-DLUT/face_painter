import cv2
import numpy as np
import os
import sys
from inference import inference


def mask_img(img, mask):
    masked_img = img * (mask > 0)
    return masked_img


def merge_images(img1, img2, overlay=True):
    '''
    img1: RGB
    img2: RGB
    '''
    H, W = img1.shape[:2]
    if overlay:
        # img2 优先
        for h in range(H):
            for w in range(W):
                if (img2[h, w, :] == 0).all():
                    img2[h, w] = img1[h, w]
        return img2
    else:
        # img1 优先
        for h in range(H):
            for w in range(W):
                if (img1[h, w, :] == 0).all():
                    img1[h, w] = img2[h, w]
        return img1


whole_mask = cv2.imread('input_159_gray.png')
print(whole_mask.shape)
whole_mask = np.array(cv2.resize(whole_mask, (W, H))).astype(np.uint8)
region_list = [17, 5, 4, 3, 2, 10, 12, 13, 1, 8, 7, 16]
for pi in region_list:
    index = np.where(whole_mask == pi)
    mask = np.zeros_like(whole_mask)
    print(mask.shape)
    mask[index[0], index[1], :] = [1, 1, 1]
    input = mask_img(img, mask)
    print(input.shape)
    name = f'face_parsing_{pi}.png'
    cv2.imwrite(name, input)
    try:
        inference(input_path=name, model_path='model.pth', output_dir='./output', need_animation=False)
        result = merge_images(result, cv2.imread(f'./output/{name}'), overlay=False)
        cv2.imwrite(f'part_{cnt}.png', result)
        cnt += 1
    except:
        print(pi)
        pass
'''
1. 
progressive
1. Mask input -> 
2. Fix output -> 图像稳定，不会大幅变化，但质量可能有限？
第一种更符合直觉

啊.....因为不知道你想画现代油画还是古典油画... 你是想画在木板上还是油画布或者油画纸上，所以.....就大致写一下基本的古典油画步骤吧！

1，一般都是要在素描纸上打底稿也就是构图，千万不要因为他是底稿就马马虎虎，一定要认真的打。
2，底料的问题，现在很多油画布上面都是有一层白乳胶的，如果你买的是另一种很原始的油画布，那你就要自己上一层涂料哦，文具店都有卖的！
3，然后你要把大面积的色块涂出来，比如山脉的大体形状，或者沙漠之类的。这里要注意，油画的上色是“由深到浅”也就是钛白是非常厉害的颜色，它可以覆盖下面的底色.....如果你按照水彩里的“由浅到深”你会发现到后面颜色上不上去了！
4，油画的纹理是十分重要的，你在上大面积底色时，可以用油画刀刮出纹理，干了后的浮雕状非常好看！
5，完善细节，这个就不多说了～
6，等大概三四周左右你的画干完了，有的还要上一层龟裂油，当然这个随意～ps：关于松节油！一定要看清楚再买！无味的无味的！不然整个房子都会弥漫着不可描述的味道！！！！

作者：逛公园爱好者
链接：https://www.zhihu.com/question/49407165/answer/127513625
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

几个问题：
1. 油画是画在白色画布上的？
2. 先上深色，在上浅色
'''

def merge_images(img1, img2, overlay=True):
    '''
    img1: RGB
    img2: RGB
    '''
    H, W = img1.shape[:2]
    '''
    if overlay:
        # img2 优先
        for h in range(H):
            for w in range(W):
                if (img2[h, w, :] == 0).all():
                    img2[h, w] = img1[h, w]
        return img2
    else:
        # img1 优先
        for h in range(H):
            for w in range(W):
                if (img1[h, w, :] == 0).all():
                    img1[h, w] = img2[h, w]
        return img1
    '''
    w1 = img1.sum(axis=-1)
    w2 = img2.sum(axis=-1)
    # print(w1.shape)
    # print(w2.shape)
    # print(set(w1.reshape(-1).tolist()))
    # print(set(w2.reshape(-1).tolist()))
    w1 = (w1 != 0)
    w2 = (w2 != 0)
    # print(w1.shape)
    # print(w2.shape)
    # print(set(w1.reshape(-1).tolist()))
    # print(set(w2.reshape(-1).tolist()))
    w = ((w1 & w2) / 2)
    # print(w.shape)
    # print(set(w.reshape(-1).tolist()))
    w1 = w1 - w
    w2 = w2 - w
    w1 = np.stack([w1, w1, w1], axis=-1)
    w2 = np.stack([w2, w2, w2], axis=-1)
    # print(w1.shape)
    # print(w2.shape)
    # print(set(w1.reshape(-1).tolist()))
    # print(set(w2.reshape(-1).tolist()))
    img = img1 * w1 + img2 * w2
    return img.astype(np.uint8)
    '''
    for h in range(H):
        for w in range(W):
            if sum(img1[h, w]) == 0:
                img1[h, w] = img2[h, w]
            else:
                if sum(img2[h, w]) != 0:
                    img1[h, w] = 0.5 * img1[h, w] + 0.5 * img2[h, w]
    return img1.astype(np.uint8)
    '''