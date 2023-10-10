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
        self.rad_cnt_path = "../data/rad_cnt/result"
        for i in range(1000):
            self.i = i
            self.create_rad_cnt_img()
            self.create_correct_img()
    
    def create_rad_cnt_img(self):
        result_path = self.rad_cnt_path + self.i
        df = pd.read_csv(result_path, header=None)
        df.columns = ["x", "y", "z", "value", "type"]
        sensor_data = df = df[df['type'] != 'source_data']
        min_value = sensor_data['value'].min()
        max_value = sensor_data['value'].max()
        sensor_data['value_normalized'] = (sensor_data['value'] - min_value) / (max_value - min_value)
        source_heatmap = sensor_data.pivot('y', 'x', 'value_normalized')
        sns.heatmap(source_heatmap, cmap='coolwarm', cbar=False, xticklabels=False, yticklabels=False)
        plt.gca().invert_yaxis() #invert y axis
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('../data/img/result2.jpg', format="jpg", bbox_inches='tight', pad_inches=0)
    
    def create_correct_img(self):
        result_path = self.rad_cnt_path + self.i
        df = pd.read_csv(result_path, header=None)
        df.columns = ["x", "y", "z", "value", "type"]
        source = df[df['type'] == 'source_data']
        max_value_source = source['value'].max()
        source['value_normalized'] = source['value'] / max_value_source
        source_heatmap = source.pivot(index = 'y', columns = 'x', values= 'value_normalized')
        source_heatmap = source_heatmap.reindex(range(10), columns=range(14))
        sns.heatmap(source_heatmap, cmap='coolwarm', cbar=False, xticklabels=False, yticklabels=False)
        plt.gca().invert_yaxis() #invert y axis
        plt.xlabel('')
        plt.ylabel('')
        plt.show()

if __name__ == "__main__":
    c2i = cnt2img()