from statistics import mean
import matplotlib

# PyCharm默认后端是 Qt5Agg. 切换到Agg，使其不画图
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import logging
import numpy as np

SCORES_CSV_PATH = "./scores/scores.csv"
SCORES_PNG_PATH = "./scores/scores.png"
GOAL_LINE = 200
CONSECUTIVE_RUNS_TO_SOLVE = 100


class ScoreLogger:

    def __init__(self, env_name):
        self.sheets_all = []
        self.env_name = env_name

        if os.path.exists(SCORES_PNG_PATH):
            os.remove(SCORES_PNG_PATH)

        if os.path.exists(SCORES_CSV_PATH):
            os.remove(SCORES_CSV_PATH)

    def add_score(self, score, run):
        item = (run, score)
        self.sheets_all.append(item)

        if run >= 500 and run % 100 == 0:
            self.save_all()

    def save_all(self):
        self._save_csv(SCORES_CSV_PATH)
        self._save_png(output_path=SCORES_PNG_PATH,
                       x_label="runs",
                       y_label="scores",
                       average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
                       show_goal=False,
                       show_trend=True,
                       show_legend=True)

        scores_as_list = [x[1] for x in self.sheets_all]
        mean_score = mean(scores_as_list)
        logging.info("Scores: (min: " + "{0:0.2f}".format(min(scores_as_list)) + ", avg: " + "{0:0.2f}".format(mean_score) + ", max: " + "{0:0.2f}".format(max(scores_as_list)) + ")\n")

    def _save_png(self, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []

        for idx, item in enumerate(self.sheets_all):
            x.append(int(idx))
            y.append(int(item[1]))

        # 画：分值线
        plt.subplots()
        plt.plot(x, y, label="score per run")

        # 画：最近连续值的分值线
        average_range = len(x) if average_of_n_last is None else min(average_of_n_last, len(x))
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * average_range, linestyle="--", label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [GOAL_LINE] * len(x), linestyle=":", label=str(GOAL_LINE) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path):
        with open(path, "w", newline='') as scores_file:
            writer = csv.writer(scores_file)
            rows = [x for x in self.sheets_all]
            writer.writerows(rows)
