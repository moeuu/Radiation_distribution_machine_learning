#!/usr/bin/env python3

import numpy as np
import pandas as pd
from logging import getLogger
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


logger = getLogger(__name__)

class cnt2img(object):
    def __init__(self):
        self.rad_cnt_path = "../../data/rad_cnt/result"
        for i in range(5):
            self.i = i
            self.create_rad_cnt_img()
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
            sigma = 0.4  # 標準偏差（適切な値に調整）
            pdf += (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2)) * value

        # 2Dヒートマップの表示
        plt.imshow(pdf, cmap='coolwarm',origin = "lower")
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(img_path, format="jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    c2i = cnt2img()