'''
@Author: 刘施恩
@date:2023-11-22-22:22
@function: 用于pytorch转换onnx模型的推理预测返回四个框的位置置信度和类别,同时可以进行量化
'''
import os
import cv2
import numpy as np
import onnxruntime

CLASSES = ['E2', 'J20', 'B2', 'F14', 'Tornado', 'F4', 'B52', 'JAS39', 'Mirage2000']

class YOLOV5():
    def __init__(self,onnxpath):
        self.onnx_session=onnxruntime.InferenceSession(onnxpath)
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()

    def get_input_name(self):
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self,img_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=img_tensor
        return input_feed

    def inference(self,img_path):
        img=cv2.imread(img_path)
        or_img=cv2.resize(img,(640,640))
        img=or_img[:,:,::-1].transpose(2,0,1)  
        img=img.astype(dtype=np.float16)
        img/=255.0
        img=np.expand_dims(img,axis=0)
        input_feed=self.get_input_feed(img)
        pred=self.onnx_session.run(None,input_feed)[0]
        return pred,or_img

def getx_c(width, xmin, xmax):
    x_c = (xmin + xmax) / 2;
    x_c = '%.16f' % (x_c / width)
    return x_c


def gety_c(height, ymin, ymax):
    y_c = (ymin + ymax) / 2
    y_c = '%.16f' % (y_c / height)
    return y_c


def getbbox_width(width, xmin, xmax):
    bbox_width = xmax - xmin
    bbox_width = '%.16f' % (bbox_width / width)
    return bbox_width


def getbbox_height(height, ymin, ymax):
    bbox_height = ymax - ymin
    bbox_height = '%.16f' % (bbox_height / height)
    return bbox_height

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1] 

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]]) 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)                              
        h = np.maximum(0, y22 - y11 + 1) 

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):

    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def filter_box(org_box,conf_thres,iou_thres):

    org_box=np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]

    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))     

    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])
        curr_cls_box = np.array(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box,iou_thres)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output

def get_result(image,box_data):
    boxes=box_data[...,:4].astype(np.int32) 
    scores=box_data[...,4]
    classes=box_data[...,5].astype(np.int32)
    res_list = []
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        # 转化
        x_c = getx_c(650,top,right)
        y_c = gety_c(650,left,bottom)
        get_width = getbbox_width(650,top,right)
        get_height = getbbox_height(650,left,bottom)
        res_list.append([x_c,y_c,get_width,get_height,score,cl])
        # print('----------------------------------------')
        # print(x_c,y_c,get_width,get_height)
    return res_list

        # cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        # cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
        #             (top, left ),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6, (0, 0, 255), 2)



if __name__=="__main__":
    onnx_path=r'last.onnx'
    model=YOLOV5(onnx_path)
    # 结果，转换的图片
    output,or_img=model.inference(r"main.jpg")
    outbox=filter_box(output,0.5,0.5)#返回框和置信度和类别
    res = get_result(or_img,outbox)
    print(res)
    # cv2.imwrite('res.jpg',or_img)
    
    

