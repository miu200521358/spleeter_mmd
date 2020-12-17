# -*- coding: utf-8 -*-
import os
import argparse
import glob
import re
import json
import csv
import shutil
import sys
import pathlib
import numpy as np
from tqdm import tqdm

from spleeter.separator import Separator
from spleeter.audio.adapter import get_default_audio_adapter

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric
from mmd.mmd.VmdData import OneEuroFilter

import cis
import librosa
import sklearn
import numpy as np
from collections import defaultdict
import scipy.signal

logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    try:
        logger.info("音声準備開始", decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.wav_file):
            logger.error("指定されたファイルパスが存在しません。\n{0}", args.wav_file, decoration=MLogger.DECORATION_BOX)
            return False, None

        # 親パス(指定がなければ動画のある場所。Colabはローカルで作成するので指定あり想定)
        base_path = str(pathlib.Path(args.wav_file).parent) if not args.parent_dir else args.parent_dir

        audio_adapter = get_default_audio_adapter()
        sample_rate = 44100
        waveform, _ = audio_adapter.load(args.wav_file, sample_rate=sample_rate)

        # 音声と曲に分離
        separator = Separator('spleeter:2stems')

        # Perform the separation :
        prediction = separator.separate(waveform)

        # 音声データ
        vocals = prediction['vocals']

        audio_adapter.save(f"{base_path}/vocals.mp3", vocals, separator._sample_rate, "mp3", "128k")

        return True, base_path
    except Exception as e:
        logger.critical("音声処理で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False, None

def analyze():
    #音声区間の全部の平均特徴をベクトルとして利用
    mfcc_data = []
    boin_list = ["a","i","u","e","o"]
    nobashi_boin = ["a:","i:","u:","e:","o:"]
    remove_list = ["silB","silE","sp"]

    #データの読み込み, 音素毎にmfccを計算(使用データは500ファイル分)
    for i in range(1,500,1):
        data_list = []
        open_file = "wav/sound-"+str(i).zfill(3)+".lab"
        filename = "wav/sound-"+str(i).zfill(3)#サンプリング周波数は16kHz
        v, fs = cis.wavread(filename+".wav")
        with open(open_file,"r") as f:
            data = f.readline().split()
            while data:
                data_list.append(data)
                data = f.readline().split()
            for j in range(len(data_list)):
                label =  data_list[j][2]
                if label in boin_list:
                    start = int(fs * float(data_list[j][0]))
                    end = int(fs * float(data_list[j][1]))
                    voice_data = v[start:end]
                    #短すぎるとうまく分析できないので飛ばす．
                    if end - start <= 512:
                        continue
                    # ハミング窓をかける
                    hammingWindow = np.hamming(len(voice_data))
                    voice_data = voice_data * hammingWindow
                    p = 0.97
                    voice_data = preEmphasis(voice_data, p)
                    mfcc_data.append(mfcc(voice_data))

#高域強調
def preEmphasis(wave, p=0.97):
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, wave)
#mfccの計算
def mfcc(wave):
    mfccs = librosa.feature.mfcc(wave, sr = fs, n_fft = 512)
    mfccs = np.average(mfccs, axis = 1)
    #一次元配列の形にする
    mfccs = mfccs.flatten()
    mfccs = mfccs.tolist()
    #mfccの第1次元と14次元以降の特徴はいらないから消す
    mfccs.pop(0)
    mfccs = mfccs[:12]
    mfccs.insert(0,label)
    return mfccs


def delta():
    #音声区間の全部の特徴をベクトルとして利用
    phoneme = []
    feature_data = []
    delta_list = []
    delta_2_list = []
    nobashi_boin = ["a:","i:","u:","e:","o:"]
    remove_list = ["silB","silE","sp"]


    #データの読み込み, 音素毎にmfccを計算(使用データは500ファイル分)
    for i in range(1,500,1):
        data_list = []
        open_file = "wav/sound-"+str(i).zfill(3)+".lab"
        filename = "wav/sound-"+str(i).zfill(3)#サンプリング周波数は16kHz
        v, fs = cis.wavread(filename+".wav")
        with open(open_file,"r") as f:
            data = f.readline().split()
            while data:
                data_list.append(data)
                data = f.readline().split()
            for j in range(len(data_list)):
                label =  data_list[j][2]
                if label not in remove_list:
                    start = int(fs * float(data_list[j][0]))
                    end = int(fs * float(data_list[j][1]))
                    #伸ばし母音に関して
                    if label in nobashi_boin:
                        label = label[0]
                    voice_data = v[start:end]
                    # ハミング窓をかける
                    hammingWindow = np.hamming(len(voice_data))
                    voice_data = voice_data * hammingWindow
                    p = 0.97
                    voice_data = preEmphasis(voice_data, p)
                    mfccs = librosa.feature.mfcc(voice_data, sr = fs, n_fft = 512)
                    mfccs_T = mfccs.T
                    S = librosa.feature.melspectrogram(voice_data, sr = fs,n_fft = 512)
                    S = sum(S)
                    PS=librosa.power_to_db(S)
                    for i in range(len(PS)):
                        feature = mfccs_T[i][1:13].tolist()
                        feature.append(PS[i])
                        feature_data.append(feature)
                        phoneme.append(label)
            K = 3
            scale = make_scale(K)
            delta = make_delta(K,scale,feature_data[len(delta_list):len(feature_data)])
            delta_list.extend(delta)
            second_delta = make_delta(K,scale,delta)
            delta_2_list.extend(second_delta)


# ------------------------------------
# https://qiita.com/k-maru/items/1596830285b0235fb1d4
#高域強調
def preEmphasis(wave, p=0.97):
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, wave)

#mfccの計算
def mfcc(wave):
    mfccs = librosa.feature.mfcc(wave, sr = fs, n_fft = 512)
    mfccs = np.average(mfccs, axis = 1)
    #一次元配列の形にする
    mfccs = mfccs.flatten()
    mfccs = mfccs.tolist()
    #mfccの第1次元と14次元以降の特徴はいらないから消す
    mfccs.pop(0)
    mfccs = mfccs[:12]
    return mfccs

#対数パワースペクトルの計算
def cal_logpower(wave):
    S = librosa.feature.melspectrogram(wave, sr = fs,n_fft = 512)
    S = sum(S)
    PS=librosa.power_to_db(S)
    PS = np.average(PS)
    return PS

#前後何フレームを見るか(通常2~5)
#フレームに対する重み部分
def make_scale(K):
    scale = []
    div = sum(2*(i**2) for i in range(1,K+1))
    for i in range(-K,K+1):
        scale.append(i/div)
    return np.array(scale)

#差分特徴量の抽出
def make_delta(K, scale, feature):
    #自身の位置からK個前までのデータ参照
    before = [feature[0]]*K
    #自身の位置以降のK個のデータ参照
    after = []
    #差分特徴量の保管リスト
    delta = []
    for i in range(K+1):
        after.append(feature[i])      
    for j in range(len(feature)):
        if j == 0:
            match =  np.array(before + after)
            dif_cal =  np.dot(scale, match)
            delta.append(dif_cal)
            after.append(feature[j+K+1])
            after.pop(0)
        #後ろからK+1までは差分としてみる部分がある
        elif j < (len(feature) - K - 1):
            match = np.array(before + after)
            dif_cal = np.dot(scale, match)
            delta.append(dif_cal)
            before.append(feature[j])
            before.pop(0)                
            after.append(feature[j+K+1])
            after.pop(0)
        #データ量-K以降はafterにデータを追加できないため
        else:
            match = np.array(before + after)
            dif_cal = np.dot(scale, match)
            delta.append(dif_cal)
            before.append(feature[j])
            before.pop(0)
            after.append(feature[len(feature)-1])
            after.pop(0)
    return delta
