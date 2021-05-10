import cv2
import numpy as np
import pretreatment as pt

from matplotlib import pyplot as plt
#LBP
def LBP(img):
    rows,cols=img.shape
    lbp=np.zeros(img.shape,dtype=np.uint8)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            for x in range(i-1,i+2):
                for y in range(j-1,j+2):
                    if img[x][y]>img[i][j]:
                        lbp[x][y]=255
                    else:
                        lbp[x][y]=0
    return lbp
def LBP_match(img,img2):
    # img = cv2.imread(filename1, 0)
    # img2=cv2.imread(filename2, 0)
    lbp=LBP(img)
    lbp2=LBP(img2)
    score=1-(np.sum(lbp^lbp2))/(330*144*255)
    # print(score)
    # cv2.imshow("lbp",lbp)
    # cv2.imshow("lbp2",lbp2)
    return score

#img1,img2为同一个体的两份指静脉图像样本
def PBBM(img1,img2):
    lbp1=LBP(img1)
    lbp2=LBP(img2)
    # lbp3=LBP(img3)
    pbbm=np.zeros(img1.shape,dtype=np.uint8)
    rows,cols=img1.shape
    for i in range(rows):
        for j in range(cols):
            if lbp1[i][j]==lbp2[i][j]:
                # if lbp3[i][j]==lbp2[i][j]:
                 pbbm[i][j]=lbp1[i][j]
            else:
                pbbm[i][j]=-1
    return pbbm
def PBBM_match(lbp,pbbm):
    # lbp=LBP(img1)
    # pbbm=PBBM(img2,img3)
    rows, cols = lbp.shape
    score=0
    count=0
    for i in range(rows):
        for j in range(cols):
            if pbbm[i][j] == -1:
                continue
            else:
                count=count+1
                if lbp[i][j]==pbbm[i][j]:
                    score=score+1
    print(score)
    print(count)
    score=score/count
    return score
#基于细节点和MHD算法


'''
Opencv中的函数cv2.goodFeatureToTrack()用来进行Shi-Tomasi角点检测，其参数说明如下所示：
第一个参数：通常情况下，其输入的应是灰度图像；
第二个参数N：是想要输出的图像中N个最好的角点；
第三个参数：设置角点的质量水平，在0~1之间；代表了角点的最低的质量，小于这个质量的角点，则被剔除；
最后一个参数：设置两个角点间的最短欧式距离；也就是两个角点的像素差的平方和；
'''
def corner(img):
    corners = cv2.goodFeaturesToTrack(img, 30, 0.5, 10)  # 返回的结果是 [[ a., b.]] 两层括号的数组。
    # print(corners)
    corners = np.int0(corners)
    # print(corners)
    for i in corners:
        x, y = i.ravel()
        print((x,y))
        cv2.circle(img, (x, y), 4, 255, -1)  # 在角点处画圆，半径为2，红色，线宽默认，利于显示
    cv2.imshow('imgcor',img)
    return corners

def end_cross_point(img2):
    rows,cols=img2.shape
    # print(img2.shape)
    tpoint=[]
    img=img2.copy()
    img=img/255
    img=img.astype(int)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if img[i][j]==1:
                # val=(np.abs(img[i-1][j]-img[i-1][j-1])+np.abs(img[i-1][j+1]-img[i-1][j])+np.abs(img[i][j+1]-img[i-1][j+1])+np.abs(img[i+1][j+1]-img[i][j+1])+np.abs(img[i+1][j]-img[i+1][j+1])+np.abs(img[i+1][j-1]-img[i+1][j])+np.abs(img[i][j-1]-img[i+1][j-1])+np.abs(img[i-1][j-1]-img[i][j-1]))
                p1=img[i-1][j-1]
                p2=img[i-1][j]
                p3=img[i-1][j+1]
                p4=img[i][j+1]
                p5=img[i+1][j+1]
                p6=img[i+1][j]
                p7=img[i+1][j-1]
                p8=img[i][j-1]
                val=np.abs(p2-p1)+np.abs(p3-p2)+np.abs(p4-p3)+np.abs(p5-p4)+\
                    np.abs(p6-p5)+np.abs(p7-p6)+np.abs(p8-p7)+np.abs((p1-p8))
                if val==2 or val>=6:
                    tpoint.append([i,j])
                else:
                    continue
    #             if val==2:
    #                 end_point.append([i,j])
    #             else:
    #                 if val>=6:
    #                     cross_point.append([i,j])
    #                 else:
    #                     continue
    # for i in range(len(end_point)):
    #     print(end_point[i])
    #     x=end_point[i][0]
    #     y = end_point[i][1]
    #     cv2.circle(img2, (y,x), 4,255, -1)# 在角点处画圆，半径为2，红色，线宽默认，利于显示
    # cv2.imshow('img2', img2)
    # # plt.imshow(img2), plt.show()
    # for i in cross_point:
    #     x, y = i[0],i[1]
    #     cv2.circle(img2, (y, x), 4, 255, -1)#自己定义的y、x和circle函数定义的相反
    # # plt.imshow(img2), plt.show()
    # cv2.imshow('img1',img2)
    return tpoint

#输入细化后的图像
def MHD(img1, img2):
    score = 0
    point1=end_cross_point(img1)
    point2=end_cross_point(img2)
    N=len(point1)
    dis=0
    for i in range(N):
        d0=np.sqrt(np.power((point1[i][0]-point2[0][0]),2)+np.power((point1[i][1]-point2[0][1]),2))
        for j in range(1,len(point2)):
            d=np.sqrt(np.power((point1[i][0]-point2[j][0]),2)+np.power((point1[i][1]-point2[j][1]),2))
            if d<d0:
                d0=d
        dis=dis+d0
    dis=dis/N
    return dis
def match(num1,num2):
    filename1="D:/finger_vein_recognition/data/"+num1+".bmp"
    filename2 = "D:/finger_vein_recognition/data/" + num2 + ".bmp"
    print(filename1)
    num1=int(num1)
    num2=int(num2)

    ismatch = False
    # 真实匹配情况
    if num1 % 40 == num2 % 40:
        ismatch = True
    print(ismatch)
    num3=str((num1+40)%120)
    # num4=str((num1+80)%120)
    filename3 = "D:/finger_vein_recognition/data/" + num3 + ".bmp"
    # filename4 = "D:/finger_vein_recognition/data/" + num4 + ".bmp"


    ##LBP
    img1=pt.enhanceImage(filename1)
    img2=pt.enhanceImage(filename2)
    LBP_score=LBP_match(img1,img2)
    print("LBP:")
    print(LBP_score)

    ##PBBM
    img3=pt.enhanceImage(filename3)
    # img4=pt.enhanceImage(filename4)
    pbbm=PBBM(img1,img3)
    lbp=LBP(img2)
    PBBM_score=PBBM_match(lbp,pbbm)
    print("PBBM")
    print(PBBM_score)

    ##细节点MHD
    img5=pt.preImgge(filename1)
    img6=pt.preImgge(filename2)
    dis=MHD(img5,img6)
    print("MHD")
    print(dis)
    return LBP_score,PBBM_score,dis,ismatch

#计算系统准确率
#score>70
#distance<12
#认为是匹配的
def is_match_score(score):
    if score>0.7:
        return True
    else:
        return False
def is_match_dis(dis):
    if dis<15:
        return True
    else:
        return False
def vote(lbp,pbbm,dis):
    if (is_match_score(lbp) and is_match_score(pbbm)) or (is_match_score(lbp) and is_match_dis(dis)) or (is_match_score(pbbm) and is_match_dis(dis)):
        return True
    else:
        return False

# match_num=0
# # mismatch_num=[0,0,0]
# # print(match_num[2])
# for i in range(1,3):
#     lbp1,pbbm1,dis1,m=match(str(i),str(i+40))
#     lbp2,pbbm2,dis2,m=match(str(i),str(i+80))
#     lbp3,pbbm3,dis3,m=match(str(i+40),str(i+80))
# #     if vote(lbp1,pbbm1,dis1):
# #         match_num=match_num+1
# #     if vote(lbp2, pbbm2, dis2):
# #         match_num = match_num + 1
# #     if vote(lbp3, pbbm3, dis3):
# #         match_num = match_num + 1
#     #LBP
#     if is_match_score(lbp1):
#         match_num[0]=match_num[0]+1
#     # else:
#     #     mismatch_num[0]=mismatch_num[0]+1
#     if is_match_score(lbp2):
#         match_num[0]=match_num[0]+1
#     if is_match_score(lbp3):
#         match_num[0]=match_num[0]+1
#
#     #PBBM
#     if is_match_score(pbbm1):
#         match_num[1]=match_num[1]+1
#     if is_match_score(pbbm2):
#         match_num[1]=match_num[1]+1
#     if is_match_score(pbbm3):
#         match_num[1]=match_num[1]+1
#
#     #MHD
#     if is_match_dis(dis1):
#         match_num[2] = match_num[2] + 1
#     if is_match_dis(dis2):
#         match_num[2] = match_num[2] + 1
#     if is_match_dis(dis3):
#         match_num[2] = match_num[2] + 1
# print(match_num)


# print(match_num/120)

# print("请输入1-150之间的两个数字来选择两幅指静脉图像\n")
# num1=input("输入第一个数字：")
# num2=input("输入第二个数字：")
# match(num1,num2)
##########代码调试################
# img=pt.preImgge("./data/115.bmp")
# ##基于LBP的评分
# img1=pt.enhanceImage("./data/14.bmp")
# img2=pt.enhanceImage("./data/94.bmp")
# img3=pt.enhanceImage("./data/54.bmp")
# # score=LBP_match(img1,img2)
# score=PBBM_match(img1,img2,img3)
# print(score)

# img1=pt.preImgge("./data/24.bmp")
# img2=pt.preImgge("./data/64.bmp")
# dis=MHD(img1,img2)
# print(dis)

# img=cv2.imread("D:/finger_vein_recognition/remove.jpg",0)
# ret,img=cv2.threshold(img,200,255,cv2.THRESH_BINARY)
# # cv2.imshow('bi',img)
# # end,cross=end_cross_point(img)
# # print(end)
# # print(cross)
# corners=corner(img)
# # # print(corners)
# # # print(corners[:,0])
# #
# #
# # # # img=img*255
# # # # print(np.max(img))
# # # cv2.imshow("img",img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()