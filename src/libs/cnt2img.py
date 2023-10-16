#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
from logging import getLogger
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transform import BaseTransform
from PIL import Image, ImageDraw


logger = getLogger(__name__)

class cnt2img(object):
    def __init__(self):
        self.rad_cnt_path = "../../data/rad_cnt/result"
        for i in range(280):
            self.i = i
            self.create_rad_cnt_img()
            self.cut_img()
            self.create_correct_img()
    
    def create_rad_cnt_img(self):
        rad_result_path = self.rad_cnt_path + str(self.i) + ".csv"
        img_path = "../../data/img/result" + str(self.i) + ".jpg"
        df = pd.read_csv(rad_result_path, header=None)
        df.columns = ["x", "y", "z", "value", "type"]
        sensor_data = df = df[df['type'] != 'source_data']
        min_value = sensor_data['value'].min()
        max_value = sensor_data['value'].max()
        sensor_data.loc[:,'value_normalized'] = (sensor_data['value'] - min_value) / (max_value - min_value)
        source_heatmap = sensor_data.pivot('y', 'x', 'value_normalized')
        sns.heatmap(source_heatmap, cmap='coolwarm', cbar=False, xticklabels=False, yticklabels=False)
        plt.gca().invert_yaxis() #invert y axis
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig(img_path, format="jpg", bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def create_correct_img(self):
        rad_result_path = self.rad_cnt_path + str(self.i) + ".csv"
        img_path = "../../data/cor_img/cor" + str(self.i) + ".jpg"

        df = pd.read_csv(rad_result_path, header=None)
        df.columns = ["x", "y", "z", "value", "type"]
        source = df[df['type'] == 'source_data']
        max_value_source = source['value'].max()
        source = source.copy()  # Create a copy of the subset to avoid the warning
        source['value_normalized'] = source['value'] / max_value_source

        # 2Dグリッドの作成
        x_range = np.arange(0, 14, 0.1)  # x座標の範囲
        y_range = np.arange(0, 10, 0.1)  # y座標の範囲
        X, Y = np.meshgrid(x_range, y_range)

        # 3D正規分布の計算と重ね合わせ
        pdf = np.zeros_like(X)
        for i in range(len(source)):
            xi, yi, value = source['x'][i], source['y'][i], source['value_normalized'][i]
            sigma = 0.2  # 標準偏差（適切な値に調整）
            pdf += (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2)) * value

        # 2Dヒートマップの表示
        plt.imshow(pdf, cmap='coolwarm',origin = "lower")
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(img_path, format="jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

    def cut_img(self):
        img_path = "../../data/img/result" + str(self.i) + ".jpg"
        input_image = Image.open(img_path) #(496, 369) 183024

        # 切り取りたい領域の座標（左上と右下）を指定
        left = random.randint(0, 446)  # 左上の x 座標
        top = random.randint(0, 319)  # 左上の y 座標
        right = left + random.randint(20, 50) # 右下の x 座標
        bottom = top + random.randint(20, 50) # 右下の y 座標

        # 切り取る領域を白で埋める
        draw = ImageDraw.Draw(input_image)
        draw.rectangle([left, top, right, bottom], fill=(255, 255, 255))
        
        # 切り取った画像を保存
        input_image.save(img_path, format="JPEG")



if __name__ == "__main__":
    c2i = cnt2img()