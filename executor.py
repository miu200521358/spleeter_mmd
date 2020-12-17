# -*- coding: utf-8 -*-
import argparse
import time

import mmd.lip
from mmd.utils.MLogger import MLogger

logger = MLogger(__name__)


def show_worked_time(elapsed_time):
    # 経過秒数を時分秒に変換
    td_m, td_s = divmod(elapsed_time, 60)
    td_h, td_m = divmod(td_m, 60)

    if td_m == 0:
        worked_time = "{0:02d}秒".format(int(td_s))
    elif td_h == 0:
        worked_time = "{0:02d}分{1:02d}秒".format(int(td_m), int(td_s))
    else:
        worked_time = "{0:02d}時間{1:02d}分{2:02d}秒".format(int(td_h), int(td_m), int(td_s))

    return worked_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-file', type=str, dest='wav_file', default='', help='wave file path')
    parser.add_argument('--parent-dir', type=str, dest='parent_dir', default='', help='Process parent dir path')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='Log level')
    parser.add_argument("--log-mode", type=int, dest='log_mode', default=0, help='Log output mode')

    args = parser.parse_args()
    MLogger.initialize(level=args.verbose, mode=args.log_mode)
    result = True

    start = time.time()

    logger.info("MMDリップトレース開始\n　処理対象映像ファイル: {0}", args.wav_file, decoration=MLogger.DECORATION_BOX)

    result, wav_dir = mmd.lip.execute(args)

    elapsed_time = time.time() - start

    logger.info("MMDリップトレース終了\n　処理対象映像ファイル: {0}\n　トレース結果: {1}\n　処理時間: {2}", \
                args.wav_file, wav_dir, show_worked_time(elapsed_time), decoration=MLogger.DECORATION_BOX)
    

