import cv2
import numpy as np
from queue import Queue
#脊波变换-->用于图像增强
#def ridgelet_transform(img):
#截取矩形兴趣域
def get_ROI(img):
    #去除部分无关区域
    img=img[30:226,50:380]
    # cv2.imshow('quchu',img)
    #提取边缘
    bimg1=cv2.Canny(img,20,240)
    # cv2.imshow('canny提取边缘', bimg1)
    h,w=bimg1.shape
    y1=0
    y2=0
    for k in range(w):
        for i in range(h//2,0,-1):
            if(bimg1[i][k]==255):
                y1+=i
        for j in range(h//2,h):
            if(bimg1[j][k]==255):
                y2+=j
    y1=y1//(w)
    y2=y2//(w)
    roi_img=img[y1:y2,:]
    #尺度归一化
    # print(roi_img.shape)
    print(roi_img.shape)
    if roi_img.shape[0]==0 or roi_img.shape[1]==0:
        roi_img=img[:,50:202]
    roi_unimg=cv2.resize(roi_img,(330,144),cv2.INTER_LINEAR)
    # cv2.imshow('uni', roi_unimg)
    return roi_unimg


#截取矩形兴趣域
def get_ROI0(img):
    #去除部分无关区域
    img=img[0:320,200:460]
    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)
    img=cv2.medianBlur(img,3)
    cv2.imshow('caij',img)
    #提取边缘
    bimg=cv2.Canny(img,20,120)
    cv2.imshow('tst', bimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #裁剪：分为左右两部分，分别计算平均边缘坐标
    h,w=bimg.shape
    x1=0
    x2=0
    for k in range(h):
        for i in range(w//2,0,-1):
            if(bimg[k][i]==255):
                x1+=i
        for j in range(w//2,w):
            if(bimg[k][j]==255):
                x2+=j
    x1=x1//h
    x2=x2//h
    # print(x1)
    # print(x2)
    roi_img=img[:,x1:x2]
    if roi_img[1]==0:
        roi_img=img[:,50:202]
    #尺度归一化
    roi_unimg=cv2.resize(roi_img,(150,320),cv2.INTER_LINEAR)
    # print("???")
    # print(roi_img.shape)
    return roi_unimg

#图像增强
def clahe_gabor(roi_img):
    #CLAHE:限制对比度的自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,5))
    clahe_img=clahe.apply(roi_img)
    # cv2.imwrite('D:/finger_vein_recognition/clahe_roi_img.jpg',clahe_img)
    #Gabor滤波和融合都没有调试好
    # res=np.zeros(roi_img.shape,np.uint8)
    # for i in range(4):
    #     gabor = cv2.getGaborKernel(ksize=(5, 5), sigma=20, theta=i*45, lambd=30, gamma=0.375)
    #     gabor_img = cv2.filter2D(src=clahe_img, ddepth=cv2.CV_8UC3, kernel=gabor)
    #     cv2.imwrite('D:/finger_vein_recognition/'+str(i)+'.jpg', gabor_img)
    #     # gabor_img=cv2.GaussianBlur(gabor_img,ksize=(3,3),sigmaX=0.8)
    #     # cv2.imshow('1', gabor_img)
    #     # ret,bimg=cv2.threshold(gabor_img,30,255,cv2.THRESH_BINARY)
    #     # cv2.imshow('2',bimg)
    #     # res=cv2.add(res,bimg)

    return clahe_img
'''
sigma_x:标准差
dnum:方向的数目
s:尺度
L:tje length od y-direction
'''

#得到多尺度匹配滤波核
def getMultiMatchFilterKernel(sigma,L,theta,s):
    width=int(np.sqrt((6*sigma+1)**2+L**2))
    mutilMatchFilter=np.zeros((width,width))
    if np.mod(width,2)==0:
        width=width+1
    halfL=int((width-1)/2)
    row=1
    for y in range(halfL,-halfL,-1):
        col=1
        for x in range(-halfL,halfL):
            p=x*np.cos(theta)+y*np.sin(theta)
            q=x*np.cos(theta)-y*np.sin(theta)
            if np.abs(p)>3*sigma or np.abs(q)>s*L/2:
                mutilMatchFilter[row][col]=0
            else:
                # mutilMatchFilter[row][col]=-np.exp(-(p**2)/(s*sigma**2))
                mutilMatchFilter[row][col] = -np.exp(-5*(p/sigma)**2/(np.sqrt(2*np.pi)*sigma*s))
            col=col+1
        row=row+1
    mean=np.sum(mutilMatchFilter)/np.count_nonzero(mutilMatchFilter)
    mutilMatchFilter[mutilMatchFilter!=0]=mutilMatchFilter[mutilMatchFilter!=0]-mean
    return mutilMatchFilter
def applyMultiMatchFilter(img,sigma_x,L,dnum,s):
    h,w=img.shape
    mf_img=np.zeros((h,w,dnum),dtype=np.uint8)
    # res = np.zeros((h, w, dnum), dtype=np.uint8)
    for i in range(dnum):
        multiMatchFilter=getMultiMatchFilterKernel(sigma_x,L,(np.pi/dnum)*i,s)
        # print(multiMatchFilter)
        mf_img[:,:,i]=cv2.filter2D(img,ddepth=cv2.CV_8UC3,kernel=multiMatchFilter)
    # print(mf_img.shape)
    res=np.max(mf_img,axis=2)
    return res

def MMF(enhance_img):
    # 应用多尺度匹配滤波提取静脉纹路
    mutil_img1 = applyMultiMatchFilter(enhance_img, 5, 5, 12, 0.03)
    # cv2.imshow('s=0.03 filter response', mutil_img1)
    # cv2.imwrite('D:/finger_vein_recognition/003.jpg',mutil_img1)
    mutil_img2 = applyMultiMatchFilter(enhance_img, 5, 5, 12, 0.06)
    # cv2.imshow('0.06 filter response', mutil_img2)
    # cv2.imwrite('D:/finger_vein_recognition/006.jpg',mutil_img2)
    mutil_img3 = applyMultiMatchFilter(enhance_img, 5, 5, 12, 0.09)
    # cv2.imshow('0.09 filter response', mutil_img3)
    # 三个尺度加权乘积
    res = cv2.multiply(mutil_img1, mutil_img2, scale=0.1)
    res = cv2.multiply(res, mutil_img3, scale=0.1)
    #进行形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=1)
    res = cv2.medianBlur(res, 5)
    # cv2.imshow('multi-scale matched filter response', res)
    #二值化
    ret, res = cv2.threshold(res, 40, 255, cv2.THRESH_BINARY)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow('bi', res)
    return res

#细化 zhng-suen细化算法
# 定义像素点周围的8邻域
#                P9 P2 P3
#                P8 P1 P4
#                P7 P6 P5
def neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9

# 计算邻域像素从0变化到1的次数
def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2,P3,...,P8,P9,P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)

def delete(img,flag):
    rows, cols = img.shape
    for y in range(cols):
        for x in range(rows):
            if flag[x][y]==1:
                img[x,y]=0
    return img

def ZhangSuen(img):
    flag = np.zeros(img.shape)
    rows,cols=img.shape
    #step one
    for y in range(1,cols-1):
        for x in range(1,rows-1):
            if img[x][y]==1:#前景点
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img)
                if(2<=sum(n)<=6 and transitions(n)==1 and
                        P2*P4*P6==0 and  P4*P6*P8==0):
                    flag[x,y]=1
    if np.sum(flag)>0:
        img=delete(img,flag)
        #flag清零
        flag = np.zeros(img.shape)
        #step two
        for y in range(1,cols -1) :
            for x in range(1,rows -1):
                if img[x][y] == 1:  # 前景点
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img)
                    if (2 <= sum(n) <= 6 and transitions(n) == 1 and
                            P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                        flag[x, y] = 1
        if np.sum(flag)>0:
            img=delete(img,flag)
            img=ZhangSuen(img)
            return img
        else:
            return img
    else:
        return img


def findBurr(img,i,j,q):
    if q.full():
        while(not q.empty()):
            index=q.get()
            img[index[0]][index[1]]=1
        # print(q.empty())
        return img
    else:
        n = neighbours(i, j, img)
        if sum(n) > 1:
            return img
        else:
            if sum(n) == 0:
                img[i][j] = 0
                return img
            else:
                q.put([i, j])
                img[i][j] = 0
                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if img[x][y] == 1:
                            i = x
                            j = y
                            img=findBurr(img,i,j,q)
                            return img
#细化的毛刺去除
def removeBurr(img):
    rows,cols=img.shape
    # print(rows,cols)
    thresh=25
    q = Queue(thresh)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            # print(img[i][j])
            # print(i,j)
            if img[i][j]==1:
                img=findBurr(img,i,j,q)
                while (not (q.empty())):
                    q.get()
                # n = neighbours(i, j, img)
                # if sum(n)>1:
                #     continue
                # else :
                #     if sum(n)==0:
                #         img[i][j]=0
                #     else:
                #         L=L+1
                #         q.put([i,j])
                #         img[i][j]=0
                #         for x in range(i-1,i+2):
                #             for y in range(j-1,j+2):
                #                 if img[x][y]==1:
                #                     i=x
                #                     j=y
    img = black(img)
    return img

#
# #先进先出队列
# q=Queue(maxsize=5)

#四周全黑
def black(img):
    rows,cols=img.shape
    for i in range(rows):
        img[i][0]=0
        img[i][cols-1]=0
    for j in range(cols):
        img[0][j]=0
        img[rows-1][j]=0
    return img

def enhanceImage(filename):
    img = cv2.imread(filename, 0)
    roi_img = get_ROI(img)
    # 使用clahe图像增强并中值滤波
    enhance_img = clahe_gabor(roi_img)
    enhance_img = cv2.medianBlur(enhance_img, 3)
    # cv2.imshow('enhance', enhance_img)
    return enhance_img

def preImgge(filename):
    img = cv2.imread(filename, 0)
    roi_img = get_ROI(img)
    # 使用clahe图像增强并中值滤波
    enhance_img = clahe_gabor(roi_img)
    enhance_img = cv2.medianBlur(enhance_img, 3)
    # cv2.imshow('enhance', enhance_img)
    #多尺度匹配滤波
    res = MMF(enhance_img)
    res = res / 255
    #细化
    xihua = ZhangSuen(res)
    # cv2.imshow('Zhangsuen', xihua)
    #细化毛刺去除
    remove = removeBurr(xihua)
    # cv2.imshow('remove', remove)
    # cv2.imwrite('D:/finger_vein_recognition/remove.jpg', remove * 255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return remove*255

# preImgge("./data/115.bmp")


################代码调试######################
# img=cv2.imread('./data/115.bmp',0)
# # equ = cv2.equalizeHist(img)
# # cv2.imshow('test',equ)
# # print(img)
# # gray_normalization(img)
# #得到ROI
# roi_img=get_ROI(img)
#
# #
# # print('ROI')
# # cv2.imshow('roi',roi_img)
# # print(roi_img.shape)
# #使用clahe图像增强并中值滤波
# enhance_img=clahe_gabor(roi_img)
# enhance_img=cv2.medianBlur(enhance_img,3)
# cv2.imshow('enhance',enhance_img)
# # cv2.imwrite('D:/finger_vein_recognition/enhance1.jpg',enhance_img)
# # enhance_img=roi_img
# res=MMF(enhance_img)
# # #应用多尺度匹配滤波提取静脉纹路
# # mutil_img1=applyMultiMatchFilter(enhance_img,5,5,12,0.03)
# # cv2.imshow('s=0.03 filter response',mutil_img1)
# # # cv2.imwrite('D:/finger_vein_recognition/003.jpg',mutil_img1)
# # mutil_img2=applyMultiMatchFilter(enhance_img,5,5,12,0.06)
# # cv2.imshow('0.06 filter response',mutil_img2)
# # # cv2.imwrite('D:/finger_vein_recognition/006.jpg',mutil_img2)
# # mutil_img3=applyMultiMatchFilter(enhance_img,5,5,12,0.09)
# # cv2.imshow('0.09 filter response',mutil_img3)
# # # cv2.imwrite('D:/finger_vein_recognition/009.jpg',mutil_img3)
# # # res=np.multiply(mutil_img1,mutil_img2)
# # # res=mutil_img1+mutil_img2+mutil_img3
# # # for i in range(141):
# # #     for j in range(350):
# # #         temp=mutil_img3[i][j]*mutil_img2[i][j]*mutil_img1[i][j]
# # #         if temp>255:
# # #             res[i][j]=255
# # #         if temp<0:
# # #             res[i][j]=0
# #
# # # res=np.multiply(mutil_img3,res)
# # #三个尺度加权乘积
# # res=cv2.multiply(mutil_img1,mutil_img2,scale=0.1)
# # res=cv2.multiply(res,mutil_img3,scale=0.1)
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# # # res=cv2.erode(res,kernel,iterations=2)
# #
# # # res=cv2.dilate(res,kernel)
# # # res=cv2.erode(res,kernel,iterations=2)
# # # res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, iterations=2)
# # res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=1)
# # res=cv2.medianBlur(res,5)
# # cv2.imshow('multi-scale matched filter response',res)
# # # cv2.imwrite('D:/finger_vein_recognition/mutil.jpg',res)
# #
# # # ret2,res = cv2.threshold(res,24,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # ret,res=cv2.threshold(res,40,255,cv2.THRESH_BINARY)
# #
# #
# # res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel2, iterations=1)
# # # res=cv2.erode(res,kernel,iterations=1)
# #
# res=res/255
# #
# # cv2.imshow('binarization',res)
#
#
#
#
# xihua=ZhangSuen(res)
# cv2.imshow('Zhangsuen',xihua)
# # # cv2.imwrite('D:/finger_vein_recognition/xihua.jpg',xihua)
# remove=removeBurr(xihua)
# # # res=res*255
# # # res=np.uint8(res)
# # # print(np.max(res))
# # # print(type(res))
# # # res = res.astype(np.uint8)
# # # print(np.max(res))
#
# # # cv2.imwrite('D:/finger_vein_recognition/remove.jpg',remove*255)
# cv2.imshow('remove',remove)
# #
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()