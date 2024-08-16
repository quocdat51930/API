import random
import torch
import numpy as np
import math
import cv2
# import streamlit

from yolov6.utils.events import load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression

class my_yolov6():
    color_global = ''
    def __init__(self, weights, device, yaml, img_size, half):
        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size

        # Half precision
        if half & (self.device.type != 'cpu'):
            self.model.model.half()
            self.half = True
        else:
            self.model.model.float()
            self.half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(
                next(self.model.model.parameters())))  # warmup

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

    # @staticmethod
    # def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    #     # Add one xyxy box to image with label
    #     p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    #     cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    #     if label:
    #         tf = max(lw - 1, 1)  # font thickness
    #         w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    #         outside = p1[1] - h - 3 >= 0  # label fits outside box
    #         p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    #         cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    #         cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
    #                     thickness=tf, lineType=cv2.LINE_AA)
    
    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        global color_global;
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color_global = color
        print(color)
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    @staticmethod
    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

    def precess_image(self,img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    # def infer(self, source, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
    #     img, img_src = self.precess_image(source, self.img_size, self.stride, self.half)
    #     img = img.to(self.device)

    #     if len(img.shape) == 3:
    #         img = img[None]
    #         # expand for batch dim

    #     pred_results = self.model(img)
    #     det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    #     label_name = []
    #     percentage = []
    #     sick_name = []
    #     if len(det):
    #         det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
    #         for *xyxy, conf, cls in reversed(det):
    #             class_num = int(cls)  # integer class
    #             label = f'{self.class_names[class_num]} {conf:.2f}'
    #             self.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=(255,0,0))
    #             label_name.append(label)
    #             percentage.append(f'{conf:.2f}')
    #             sick_name.append(f'{self.class_names[class_num]}')
    #     img_src = np.asarray(img_src)

    #     return img_src, len(det), label_name, percentage, sick_name

    def infer(self, source, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
        global color_global;
        img, img_src = self.precess_image(source, self.img_size, self.stride, self.half)
        img = img.to(self.device)

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred_results = self.model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        labels_vn = ['Phình động mạch chủ', 'Xẹp phổi', 'Vôi hóa', 'Tim to', 'Đông đặc phổi', 'Phổi kẽ', 'Thâm nhiễm', 'Mờ phổi', 'Khối u', 'Tổn thương khác', 'Tràn dịch màn phổi', 'Màng phổi dày', 'Tràn khí phế mạc', 'Xơ phổi']
        label_mapping = {i: label for i, label in enumerate(labels_vn)}
        labels = []  # List to store predicted labels
        percentage = []
        sick_name = []
        sick_name_eng = []
        colors = []

        if len(det):
            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)  # integer class
                label = f'{self.class_names[class_num]} {conf:.2f}'
                # label_name = f'{label_mapping.get(class_num)} {conf:.2f}'
                label_name = f'{self.class_names[class_num]} ({label_mapping.get(class_num)})  {conf:.2f}'
                print(label.encode('utf-8').decode('utf-8'))
                self.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=(255,0,0))
                print(color_global)
                labels.append(label_name)  # Append label to the list
                percentage.append(f'{conf:.2f}')
                sick_name.append(f'{self.class_names[class_num]}')
                sick_name_eng.append(f'{label_mapping.get(class_num)}')
                colors.append(color_global)
                
                # sick_name.append(label_name)
        img_src = np.asarray(img_src)

        return img_src, len(det), labels, percentage, sick_name, sick_name_eng, colors
    

    # def infer(self, source, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
    #     img, img_src = self.precess_image(source, self.img_size, self.stride, self.half)
    #     img = img.to(self.device)

    #     if len(img.shape) == 3:
    #         img = img[None]  # expand for batch dim

    #     pred_results = self.model(img)
    #     det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    #     labels_vn = ['Phình động mạch chủ', 'Xẹp phổi', 'Vôi hóa', 'Tim to', 'Đông đặc phổi', 'Phổi kẽ', 'Thâm nhiễm', 'Mờ phổi', 'Khối u', 'Tổn thương khác', 'Tràn dịch màng phổi', 'Màng phổi dày', 'Tràn khí phế mạc', 'Xơ phổi']
    #     label_mapping = {i: label for i, label in enumerate(labels_vn)}
    #     labels = []  # List to store predicted labels

    #     if len(det):
    #         det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
    #         for *xyxy, conf, cls in reversed(det):
    #             class_num = int(cls)  # integer class
    #             label = f'{self.class_names[class_num]} {conf:.2f} '
    #             label_name = f'{self.class_names[class_num]} ({label_mapping.get(class_num)})  {conf:.2f}'
    #             print(label.encode('utf-8').decode('utf-8'))
    #             self.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=(255,0,0))
    #             labels.append(label_name)  # Append label to the list

    #     img_src = np.asarray(img_src)

    #     return img_src, len(det), labels




# # -*- coding: utf-8 -*-
# import torch
# import numpy as np
# import math
# import cv2
# import yaml  # Import the PyYAML library
# import random

# from yolov6.utils.events import load_yaml
# from yolov6.layers.common import DetectBackend
# from yolov6.data.data_augment import letterbox
# from yolov6.utils.nms import non_max_suppression
# from PIL import Image, ImageDraw, ImageFont

# class my_yolov6():
#     def __init__(self, weights, device, yaml_path, img_size, half):
#         self.__dict__.update(locals())

#         # Init model
#         self.device = device
#         self.img_size = img_size
#         cuda = self.device != 'cpu' and torch.cuda.is_available()
#         self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
#         self.model = DetectBackend(weights, device=self.device)
#         self.stride = self.model.stride

#         # Read YAML file with UTF-8 encoding
#         with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
#             yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

#         self.class_names = yaml_data['names']
#         self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size

#         # Half precision
#         if half & (self.device.type != 'cpu'):
#             self.model.model.half()
#             self.half = True
#         else:
#             self.model.model.float()
#             self.half = False

#         if self.device.type != 'cpu':
#             self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(
#                 next(self.model.model.parameters())))  # warmup

#         # Switch model to deploy status
#         self.model_switch(self.model.model, self.img_size)

#     # -*- coding: utf-8 -*-
#     @staticmethod
#     def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
#         # Add one xyxy box to image with label
#         p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
#         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
#         if label:
#             tf = max(lw - 1, 1)  # font thickness
#             w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
#             outside = p1[1] - h - 3 >= 0  # label fits outside box
#             p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#             cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
#             cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
#                         thickness=tf, lineType=cv2.LINE_AA)
        

#     @staticmethod
#     def rescale(ori_shape, boxes, target_shape):
#         '''Rescale the output to the original image shape'''
#         ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
#         padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

#         boxes[:, [0, 2]] -= padding[0]
#         boxes[:, [1, 3]] -= padding[1]
#         boxes[:, :4] /= ratio

#         boxes[:, 0].clamp_(0, target_shape[1])  # x1
#         boxes[:, 1].clamp_(0, target_shape[0])  # y1
#         boxes[:, 2].clamp_(0, target_shape[1])  # x2
#         boxes[:, 3].clamp_(0, target_shape[0])  # y2

#         return boxes

#     @staticmethod
#     def make_divisible(x, divisor):
#         # Upward revision the value x to make it evenly divisible by the divisor.
#         return math.ceil(x / divisor) * divisor

#     def check_img_size(self, img_size, s=32, floor=0):
#         """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
#         if isinstance(img_size, int):  # integer i.e. img_size=640
#             new_size = max(self.make_divisible(img_size, int(s)), floor)
#         elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
#             new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
#         else:
#             raise Exception(f"Unsupported type of img_size: {type(img_size)}")

#         if new_size != img_size:
#             print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
#         return new_size if isinstance(img_size,list) else [new_size]*2

#     def model_switch(self, model, img_size):
#         ''' Model switch to deploy status '''
#         from yolov6.layers.common import RepVGGBlock
#         for layer in model.modules():
#             if isinstance(layer, RepVGGBlock):
#                 layer.switch_to_deploy()

#     def precess_image(self,img_src, img_size, stride, half):
#         '''Process image before image inference.'''
#         image = letterbox(img_src, img_size, stride=stride)[0]
#         # Convert
#         image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         image = torch.from_numpy(np.ascontiguousarray(image))
#         image = image.half() if half else image.float()  # uint8 to fp16/32
#         image /= 255  # 0 - 255 to 0.0 - 1.0

#         return image, img_src

#     # def infer(self, source, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
#     #     img, img_src = self.precess_image(source, self.img_size, self.stride, self.half)
#     #     img = img.to(self.device)

#     #     if len(img.shape) == 3:
#     #         img = img[None]
#     #         # expand for batch dim

#     #     pred_results = self.model(img)
#     #     det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

#     #     if len(det):
#     #         det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
#     #         for *xyxy, conf, cls in reversed(det):
#     #             class_num = int(cls)  # integer class
#     #             label = f'{self.class_names[class_num]} {conf:.2f}'
#     #             self.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=(255,0,0))

#     #         img_src = np.asarray(img_src)

#     #     return img_src, len(det)

#     def infer(self, source, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
#         img, img_src = self.precess_image(source, self.img_size, self.stride, self.half)
#         img = img.to(self.device)

#         if len(img.shape) == 3:
#             img = img[None]  # expand for batch dim

#         pred_results = self.model(img)
#         det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
#         labels_vn = ['Phình động mạch chủ', 'Xẹp phổi', 'Vôi hóa', 'Tim to', 'Đông đặc phổi', 'Phổi kẽ', 'Thâm nhiễm', 'Mờ phổi', 'Khối u', 'Tổn thương khác', 'Tràn dịch màn phổi', 'Màng phổi dày', 'Tràn khí phế mạc', 'Xơ phổi']
#         label_mapping = {i: label for i, label in enumerate(labels_vn)}
#         labels = []  # List to store predicted labels

#         if len(det):
#             det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
#             for *xyxy, conf, cls in reversed(det):
#                 class_num = int(cls)  # integer class
#                 label = f'{self.class_names[class_num]} {conf:.2f}'
#                 label_name = f'{label_mapping.get(class_num)} {conf:.2f}'
#                 print(label.encode('utf-8').decode('utf-8'))
#                 self.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=(255,0,0))
#                 labels.append(label_name)  # Append label to the list

#         img_src = np.asarray(img_src)

#         return img_src, len(det), labels

