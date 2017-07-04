import csv
import os

import PIL.Image
import numpy as np
import pandas as pd
import tensorflow as tf

from config import cfg


def csv_to_dict(csv_path):
    with open(csv_path, 'r') as fp:
        csv_fp = csv.reader(fp)
        next(csv_fp)
        d = dict(filter(None, csv_fp))
        return d


def image2matrix(data_dir, prefixes, data_dictionary, query_file=False):

    x_raw_img_data = []
    y_labels = []

    for prefix in prefixes:
        raw_image = PIL.Image.open(os.path.join(data_dir, "{}.png".format(prefix)))
        if cfg['enabled_down_scaling']:
            raw_image.thumbnail((cfg['down_scale_img_h'], cfg['down_scale_img_w']), PIL.Image.ANTIALIAS)

        img_arr = np.array(raw_image.getdata()).reshape((raw_image.size[0], raw_image.size[1], 1)).astype(np.uint8)
        x_raw_img_data.append(img_arr)

        if not query_file:
            y_label = data_dictionary.get(prefix)
            y_labels.append(np.array(y_label).reshape(1))

    if query_file:
        return np.array(x_raw_img_data)
    else:
        return np.array(x_raw_img_data), np.array(y_labels)


def export_result_to_csv(score_predictions, query_image_file, model_name, epoch_number):
    query_df = pd.read_csv(query_image_file)
    columns = query_df.columns.values

    output_df = pd.DataFrame(columns=columns)
    output_df['Id'] = query_df['Id']
    output_df['Predicted'] = [round(p[0], 10) for prediction in score_predictions for pred in prediction for p in pred]

    model_name = os.path.join(cfg['path']['output_dir'], model_name)
    output_df.to_csv(model_name + "/output-epoch-" + str(epoch_number), index=False)


def print_and_log(string_to_log, logger):
    print(string_to_log)
    logger.info(string_to_log)
