### ++++++++++++++++++++++++++++++++++++++FFT CODE+++++++++++++++++++++++++++++++++++++

# t = np.linspace(0, 0.5, 500)
# s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)
# # t = np.array([0,1,2,3,4,5,6,7])
# # s = np.array([0,1,2,3,4,5,6,7])

# plt.ylabel("Amplitude")
# plt.xlabel("Time [s]")
# # plt.plot(t, s)
# # plt.show()

# fft = np.fft.fft(s)
# print(fft)
# # for i in range(2):
# #     print("Value at index {}:\t{}".format(i, yf[i + 1]), "\nValue at index {}:\t{}".format(yf.size -1 - i, yf[-1 - i]))

# print(t[1], t[0])
# T = t[1] - t[0]  # sample rate
# print(1/T)
# print(s.size)
# N = s.size

# # 1/T = frequency
# f = np.linspace(0, 1 / T, N)
# print(f)

# freq = np.fft.fftfreq(N, d = T)# * 1/T
# print(freq)


# plt.ylabel("Amplitude")
# plt.xlabel("Frequency [Hz]")
# plt.bar(freq[1:N // 2], np.abs(fft)[1:N // 2] * 2 / N, width=1.5)  # 1 / N is a normalization factor
# plt.show()




import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd

# x = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
# y = np.exp((-x)/3.0)

df = pd.read_csv("Data/EatFood1/accelerometer-1533862083.csv")

# df = df.iloc[:4096, :]


def interpolateFunc(df):
    df=df
    print(df["timestamp"].max())
    print(df["timestamp"].min())
    print(df["timestamp"][0])

    # df2 = pd.read_csv("Data/Cooking1/accelerometer-1533863975.csv")
    df["timestamp"] = (df["timestamp"] - df["timestamp"].min())/ (df["timestamp"].max() - df["timestamp"].min())
    # print(df["timestamp"])
    # df = df.iloc[:512, :]
    trans_f = interpolate.interp1d(df["timestamp"], df["x"], kind = 'cubic')
    # fcubic = interpolate.interp1d(x, y, kind='cubic')

    # Friday, August 10, 2018 12:48:03.435
    # Friday, August 10, 2018 12:48:03.457 AM
    # Friday, August 10, 2018 12:48:03.481
    # Friday, August 10, 2018 12:48:03.497
    # Friday, August 10, 2018 12:48:03.519

    # print(df["timestamp"].max())
    # print(df["timestamp"].min())
    # print(df["x"])

    xnew = np.arange(df["timestamp"].min(), df["timestamp"].max(), 0.0001)
    ynew = trans_f(xnew)


    print(len(ynew))
    print(len(df["x"]))

    # print(xnew)
    # print(ynew)

    plt.figure(0)
    plt.plot(df["timestamp"], df["x"], 'b')
    plt.figure(1)
    plt.plot(xnew, ynew, 'g')#, xnew, ycubic, 'o')



    fft = np.fft.fft(ynew)

    t=np.fft.fft(np.sin(ynew))

    freq = np.fft.fftfreq(t.shape[-1])
    plt.plot(freq)

    plt.show()

    return df

