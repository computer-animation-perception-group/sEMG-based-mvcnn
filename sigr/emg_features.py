# emg features
import numpy as np
import pywt

# from nitime import algorithms as alg
# from scipy import stats
# from numpy import linalg as LA
# from scipy import signal as sig
# from scipy import linalg
# import operator
# from math import pi
# from sampen import sampen2

def emg_iemg(signal):
    signal_abs = [abs(s) for s in signal]
    signal_abs = np.array(signal_abs)
    return np.sum(signal_abs)


def emg_mav(signal):
    signal_abs = [abs(s) for s in signal]
    signal_abs = np.array(signal_abs)
    if len(signal_abs) == 0:
        return 0
    else:
        return np.mean(signal_abs)

def emg_mav1(signal):
    signal_abs = [abs(s) for s in signal]
    signal_abs = np.array(signal_abs)
    N = len(signal)
    w = []
    for i in range(1,N+1,1):
        if i >= 0.25*N and i <= 0.75*N:
            w.append(1)
        else:
            w.append(0.5)
    w = np.array(w)
    return np.mean(signal_abs*w)

def emg_mav2(signal):
    signal_abs = [abs(s) for s in signal]
    signal_abs = np.array(signal_abs)
    N = len(signal)
    w = []
    for i in range(1,N+1,1):
        if i >= 0.25*N and i <= 0.75*N:
            w.append(1)
        elif i < 0.25*N:
            w.append(4.0*i/N)
        else:
            w.append(4.0*(i-N)/N)
    w = np.array(w)
    return np.mean(signal_abs*w)


def emg_mavslpphinyomark(signal):
    frame_length = len(signal)
    sub_window_length = int(np.floor(frame_length / 3))
    sub_window_overlap = 0    
    return emg_mavslp(signal, sub_window_length, sub_window_overlap)
    


def emg_mavslp(signal, sub_window_length, sub_window_overlap):
     frame_length = len(signal)
     if sub_window_length > frame_length:
         sub_window_length = frame_length
    
     window_step = int(sub_window_length*float(1-sub_window_overlap))
    
     mavs = []
     start = 0
     flag = 0
     for i in range(0, frame_length, sub_window_length):
         if (start+sub_window_length) >= frame_length:
             end = frame_length
             flag = 1
         else:
             end = start + sub_window_length

         each_frame = signal[start:end]

         start = start + window_step

         each_mav = emg_mav(each_frame)
         mavs.append(each_mav)
         if flag == 1:
             break

     if len(mavs) == 1:
         return mavs
     else:
         newmavs = []
         for i in range(1, len(mavs), 1):
             newmavs.append(mavs[i] - mavs[i - 1])
     return newmavs


def emg_ssi(signal):
     signal_squ = [s * s for s in signal]
     signal_squ = np.array(signal_squ)
     return np.sum(signal_squ)


# def emg_var(signal):
#     signal = np.array(signal)
#     ssi = emg_ssi(signal)
#     length = signal.shape[0]
#     if length <= 1:
#         return 0
#     return float(ssi) / (length - 1)


def emg_rms(signal):
     signal = np.array(signal)
     ssi = emg_ssi(signal)
     length = signal.shape[0]
     if length <= 0:
         return 0
     return np.sqrt(float(ssi) / length)


# def emg_mavtm(signal, order):
#     signal = np.array(signal)
#     signal_order = [s ** order for s in signal]
#     return abs(np.mean(signal_order))


def emg_vorder(signal):
    signal = np.array(signal)
    signal_order = [s ** 2 for s in signal]
    value = np.mean(signal_order)
    return value ** (float(1) / 2)


def emg_log(signal):
    signal = np.array(signal)
    signal_log = []
    for s in signal:
        if abs(s) == 0:
            signal_log.append(1e-6)
        else:
            signal_log.append(np.log(abs(s)))
    value = np.mean(signal_log)
    return np.exp(value)


def emg_wl(signal):
     signal = np.array(signal)
     length = signal.shape[0]
     wl = [abs(signal[i + 1] - signal[i]) for i in range(length - 1)]
     return np.sum(wl)


# def emg_aac(signal):
#     signal = np.array(signal)
#     length = signal.shape[0]
#     wl = [abs(signal[i + 1] - signal[i]) for i in range(length - 1)]
#     return np.mean(wl)


# def emg_zc(signal, zc_threshold):
#     sign = [[signal[i] * signal[i - 1], abs(signal[i] - signal[i - 1])] for i in range(1, len(signal), 1)]

#     sign = np.array(sign)
#     sign = sign[sign[:, 0] < 0]
#     if sign.shape[0] == 0:
#         return 0
#     sign = sign[sign[:, 1] >= zc_threshold]
#     return sign.shape[0]


# def emg_wl_dasdv(signal):
#     signal = np.array(signal)
#     length = signal.shape[0]
#     wl = [(signal[i + 1] - signal[i]) ** 2 for i in range(length - 1)]
#     sum_squ = np.sum(wl)
#     if length <= 1:
#         return 0
#     return np.sqrt(sum_squ / (length - 1))


# def emg_afb(signal, hamming_window_length):
#     hamming_window = np.hamming(hamming_window_length)
#     signal_length = len(signal)
#     signal_after_filter = []
#     end_flag = 0

#     for i in range(signal_length):
#         start = i
#         end = i + hamming_window_length
#         if end >= signal_length:
#             end = signal_length
#             end_flag = 1

#         signal_seg = signal[start:end]
#         signal_seg = np.array(signal_seg)
#         signal_after_filter.append(np.sum(signal_seg * signal_seg * hamming_window) / np.sum(hamming_window))

#         if end_flag == 1:
#             end_flag = 0
#             break
#     signal_after_filter = np.array(signal_after_filter)

#     a_value = signal_after_filter[0]

#     for i in range(1, len(signal_after_filter) - 1, 1):
#         if signal_after_filter[i] > signal_after_filter[i - 1] and signal_after_filter[i] > signal_after_filter[i + 1]:
#             a_value = signal_after_filter[i]
#             break
#     return a_value


# def emg_myop(signal, threshold):
#     signal = np.array(signal)
#     length = signal.shape[0]

#     signal = signal[signal >= threshold]
#     count = signal.shape[0]
#     if length <= 0:
#         return 0
#     return float(count) / length

def emg_sscbestninapro1(signal):
    return emg_ssc(signal, 1e-5)

def emg_ssc(signal, threshold):
     signal = np.array(signal)
     temp = [(signal[i] - signal[i - 1]) * (signal[i] - signal[i + 1]) for i in range(1, signal.shape[0] - 1, 1)]
     temp = np.array(temp)

     temp = temp[temp >= threshold]
     return temp.shape[0]


# def emg_wamp(signal, threshold):
#     signal = np.array(signal)
#     temp = [abs(signal[i] - signal[i - 1]) for i in range(1, signal.shape[0], 1)]
#     temp = np.array(temp)

#     temp = temp[temp >= threshold]
#     return temp.shape[0]


def emg_hemg(signal, bins):
     signal = np.array(signal)
     hist, bin_edge = np.histogram(signal, bins)
     return hist
     
def emg_hemg20(signal):
     signal = np.array(signal)
     hist, bin_edge = np.histogram(signal, 20)
     return hist     


# def emg_mhw_energy(signal):
#     signal = np.array(signal)
#     window_length = signal.shape[0]
#     sub_window_len = int(window_length/2.4)
#     mhwe = []
#     start = 0
#     for i in range(3):
#         end = min(start+sub_window_len, window_length)

#         sub_signal = signal[start:end]
#         start = int(start + sub_window_len*0.7)
#         hamming_window = np.hamming(len(sub_signal))
#         sub_signal = sub_signal * hamming_window
#         sub_signal = sub_signal **2
#         mhwe.append(np.sum(sub_signal))

#     mhwe = np.array(mhwe)
#     return mhwe


# def emg_mtw_energy(signal, first_percent, second_percent):
#     signal = np.array(signal)
#     window_length = signal.shape[0]
#     sub_window_len = int(window_length/2.4)
#     mtwe = []
#     start = 0
#     for i in range(3):
#         end = min(start+sub_window_len, window_length)
#         sub_signal = signal[start:end]
#         start = int(start + sub_window_len*0.7)

#         t_window = []
#         k1 = 1 / float(len(sub_signal) * first_percent)
#         k2 = 1 / float(len(sub_signal) * (second_percent - 1))
#         b2 = 1 / float(1 - second_percent)
#         first_point = int(len(sub_signal) * first_percent)
#         second_point = int(len(sub_signal) * second_percent)
#         for i in range(len(sub_signal)):
#             if i >= 0 and i < first_point:
#                 y = k1 * i
#             elif i >= second_point and i <= len(sub_signal):
#                 y = k2 * i + b2
#             else:
#                 y = 1
#             t_window.append(y)
#         t_window = np.array(t_window)
#         sub_signal = sub_signal * t_window
#         sub_signal = sub_signal ** 2
#         mtwe.append(np.sum(sub_signal))
#     mtwe = np.array(mtwe)

#     return mtwe


# def emg_arc(signal, order):
#     if order >= len(signal):
#         rd = len(signal)-1
#     else:
#         rd = order
#     arc, ars = alg.AR_est_YW(signal, rd)
#     arc = np.array(arc)
#     return arc


# def emg_cc(signal, order):
#     arc = emg_arc(signal, order)
#     cc = []
#     cc.append(-arc[0])
#     cc = np.array(cc)
#     for i in range(1, arc.shape[0], 1):
#         cp = cc[0:i]
#         cp = cp[::-1]
#         num = range(1, i + 1, 1)
#         num = np.array(num)
#         num = -num / float(i + 1) + 1
#         cp = cp * num
#         cp = np.sum(cp)
#         cc = np.append(cc, -arc[i] * (1 + cp))
#     return cc
def emg_fftdb1(signal):
    [cc, freqs] =  emg_fft(signal,100)    
    return cc

def emg_fft(signal, fs):
    fft_size = signal.shape[0]

    freqs = np.linspace(0, fs/2, fft_size/2+1)

    xf = np.fft.rfft(signal)/fft_size
    cc = np.clip(np.abs(xf), 1e-20, 1e100)
    # pl.scatter(freqs, cc)
    # pl.show()
    return cc, freqs

def emg_fft_power(signal, fs=1000):
    fft_size = signal.shape[0]
    cc, freq = emg_fft(signal, fs)
    cc = cc * cc
    cc = cc / float(fft_size)

    cc = np.array(cc)
    # if cc.all() == 0:
    #     cc[cc == 0] = 0
    #     cc[cc != 0] = 10 * np.log10(cc[cc != 0])
    #     cc = 0
    # else:
    #     cc = 10 * np.log10(cc)
    return cc, freq


# def emg_mdf(signal, fs, mtype):
#     if mtype == 'MEDIAN_POWER':
#         cc, freq = emg_fft_power(signal, fs)
#     else:
#         cc, freq = emg_fft(signal, fs)
#     csum = 0
#     pre_csum = 0
#     index = 0
#     ccsum = np.sum(cc)

#     for i in range(cc.shape[0]):
#         pre_csum = csum
#         csum = csum + cc[i]
#         if csum >= ccsum / 2:
#             if (ccsum / 2 - pre_csum) < (csum - ccsum / 2):
#                 index = i - 1
#             else:
#                 index = i
#             break
#     return freq[index]


# def emg_mnf(signal, fs, type):
#     if type == 'MEDIAN_POWER':
#         cc, freq = emg_fft_power(signal, fs)
#     else:
#         cc, freq = emg_fft(signal, fs)

#     ccsum = np.sum(cc)
#     fp = cc * freq
#     if np.sum(cc) == 0:
#         return 0

#     return np.sum(fp) / ccsum


# def emg_pkf(signal, fs):
#     cc, freq = emg_fft_power(signal, fs)

#     max_index, max_power = max(enumerate(cc), key=operator.itemgetter(1))
#     return freq[max_index]


def emg_mnp(signal, fs=1000):
    cc, freq = emg_fft_power(signal, fs)
    return np.mean(cc)


# def emg_ttp(signal, fs):
#     cc, freq = emg_fft_power(signal, fs)
#     return np.sum(cc)


# def emg_smn(signal, fs, order):
#     cc, freq = emg_fft_power(signal, fs)
#     freq = freq ** order
#     cc = cc * freq
#     return np.sum(cc)


# def emg_fr(signal, fs, low_down, low_up, high_down, high_up):
#     cc, freq = emg_fft_power(signal, fs)

#     maxfre = np.max(freq)
#     minfre = np.min(freq)

#     ld = minfre + (maxfre - minfre) * low_down / 100
#     lu = minfre + (maxfre - minfre) * low_up / 100
#     hd = minfre + (maxfre - minfre) * high_down / 100
#     hu = minfre + (maxfre - minfre) * high_up / 100

#     low = cc[(freq >= ld) & (freq <= lu)]

#     high = cc[(freq >= hd) & (freq <= hu)]

#     if len(high) == 0 | len(low) == 0:
#         return 0

#     if np.sum(high) == 0:
#         return 0

#     return np.sum(low) / np.sum(high)


# def emg_psr(signal, fs, prange):
#     cc, freq = emg_fft_power(signal, fs)
#     max_index, max_power = max(enumerate(cc), key=operator.itemgetter(1))
#     if max_index-prange < 0:
#         start = 0
#     else:
#         start = max_index - prange
#     if max_index+prange >len(signal):
#         end = len(signal)
#     else:
#         end = max_index + prange
#     range_value = cc[start:end]
#     range_value = np.sum(range_value)
#     sum_value = np.sum(cc)
#     if sum_value == 0:
#         return 0
#     return range_value / sum_value


# def emg_vcf(signal, fs):
#     sm2 = emg_smn(signal, fs, 2)
#     sm1 = emg_smn(signal, fs, 1)
#     sm0 = emg_smn(signal, fs, 0)
#     if sm0 == 0:
#         return 0

#     return sm2 / sm0 - (sm1 / sm0) ** 2


# def emg_hos2(signal, fs, t1):
#     cc, freq = emg_fft_power(signal, fs)
#     cc = np.array(cc)

#     signalt = np.zeros(cc.shape[0])
#     length = cc.shape[0]
#     if t1 >= 0:
#         signalt[0:(length - t1)] = cc[t1:]
#     else:
#         signalt[-t1:] = cc[0:(length + t1)]

#     signalt = cc * signalt
#     return np.mean(signalt)


# def emg_hos3(signal, fs, t1, t2):
#     cc, freq = emg_fft_power(signal, fs)

#     cc = np.array(cc)
#     length = cc.shape[0]
#     signalt1 = np.zeros(length)
#     signalt2 = np.zeros(length)
#     signalt1[0:(length - t1)] = cc[t1:]
#     signalt2[0:(length - t2)] = cc[t2:]
#     signalt = cc * signalt1 * signalt2
#     return np.mean(signalt)


# def emg_hos4(signal, fs, t1, t2, t3):
#     cc, freq = emg_fft_power(signal, fs)

#     cc = np.array(cc)
#     length = cc.shape[0]
#     signalt1 = np.zeros(length)
#     signalt2 = np.zeros(length)
#     signalt3 = np.zeros(length)
#     signalt1[0:(length - t1)] = cc[t1:]
#     signalt2[0:(length - t2)] = cc[t2:]
#     signalt3[0:(length - t3)] = cc[t3:]

#     signalt = cc * signalt1 * signalt2 * signalt3
#     mean4 = np.mean(signalt)
#     result = mean4 - emg_hos2(signal, fs, t1) * emg_hos2(signal, fs, t2 - t3) - emg_hos2(signal, fs, t2) * emg_hos2(
#         signal, fs, t3 - t1) - emg_hos2(signal, fs, t3) * emg_hos2(signal, fs, t1 - t2)
#     return result


# def emg_hos(signal, fs):
#     hos = []
#     hos.append(emg_hos2(signal, fs, 0))
#     hos.append(emg_hos2(signal, fs, 1))
#     hos.append(emg_hos2(signal, fs, 2))
#     for i in [0, 1, 2]:
#         for j in range(i, 3):
#             hos.append(emg_hos3(signal, fs, i, j))
#     for i in [0, 1, 2]:
#         for j in range(i, 3):
#             for k in range(j, 3):
#                 hos.append(emg_hos4(signal, fs, i, j, k))
#     return hos


def emg_dwt(signal):
    wavelet_level = int(np.log2(len(signal)))
    coeffs = pywt.wavedec(signal, 'db1', level=wavelet_level)
    return np.hstack(coeffs)


# # return all the energy of dwt coeffs
# def emg_dwt_energy(signal, wavelet_name, wavelet_level):
#     coeffs = emg_dwt(signal, wavelet_name, wavelet_level)
#     energys = []
#     for c in coeffs:
#         c_squ = [cc ** 2 for cc in c]
#         c_squ = np.array(c_squ)
#         energys.append(np.sum(c_squ))
#     energys = np.array(energys)
#     return energys


def emg_dwpt(signal, wavelet_name='db1'):
    wavelet_level = int(np.log2(len(signal)))
    wp = pywt.WaveletPacket(signal, wavelet_name, mode='sym')
    coeffs = []
    level_coeff = wp.get_level(wavelet_level)
    for i in range(len(level_coeff)):
        coeffs.append(level_coeff[i].data)
    coeffs = np.array(coeffs)
    coeffs = coeffs.flatten()
    return coeffs


# def emg_dwpt_energy(signal, wavelet_name, wavelet_level):
#     coeffs = emg_dwpt(signal, wavelet_name, wavelet_level)
#     coeffs = coeffs ** 2
#     return np.sum(coeffs)


def emg_mdwt(signal, wavelet_name='db1'):
     coeffs = pywt.wavedec(signal, wavelet_name)
     mdwt = []
     for detail_coeff in range(1, len(coeffs)):
         coeff_abs = [abs(c) for c in coeffs[detail_coeff]]
         coeff_abs = np.array(coeff_abs)
         mdwt.append(np.sum(coeff_abs))
     mdwt = np.array(mdwt)
     return mdwt

def emg_mdwtdb7ninapro(signal):
     coeffs = pywt.wavedec(signal, wavelet='db7', mode='symmetric', level=3)
     mdwt = []
     for detail_coeff in range(1, len(coeffs)):
         coeff_abs = [abs(c) for c in coeffs[detail_coeff]]
         coeff_abs = np.array(coeff_abs)
         mdwt.append(np.sum(coeff_abs))
     mdwt = np.array(mdwt)
     return mdwt

def emg_mdwtdb1ninapro(signal):
     coeffs = pywt.wavedec(signal, wavelet='db1', mode='symmetric', level=3)
     mdwt = []
     for detail_coeff in range(1, len(coeffs)):
         coeff_abs = [abs(c) for c in coeffs[detail_coeff]]
         coeff_abs = np.array(coeff_abs)
         mdwt.append(np.sum(coeff_abs))
     mdwt = np.array(mdwt)
     return mdwt

# def emg_mrwa(signal, wavelet_name):
#     coeffs = pywt.wavedec(signal, wavelet_name)
#     coeffs = np.array(coeffs)
#     mrwa = []
#     mrwa.append(LA.norm(coeffs[0]))
#     for i in range(1, coeffs.shape[0], 1):
#         detail = coeffs[i]
#         detail_squ = [d * d for d in detail]
#         detail_squ = np.array(detail_squ)
#         mrwa.append(np.sum(detail_squ) / detail_squ.shape[0])
#     mrwa = np.array(mrwa)
#     return mrwa


def emg_dwpt_mean(signal):
     coeffs = emg_dwpt(signal)
     return np.mean(coeffs)


def emg_dwpt_sd(signal):
     coeffs = emg_dwpt(signal)
     return np.std(coeffs)


# def emg_dwpt_skewness(signal, wavelet_name, wavelet_level):
#     coeffs = emg_dwpt(signal, wavelet_name, wavelet_level)
#     coeffs = np.array(coeffs)
#     skew = stats.skew(coeffs)
#     return skew


# def emg_dwpt_kurtosis(signal, wavelet_name, wavelet_level):
#     coeffs = emg_dwpt(signal, wavelet_name, wavelet_level)
#     coeffs = np.array(coeffs)
#     kurtosis = stats.kurtosis(coeffs)
#     return kurtosis


# def emg_dwpt_m(signal, wavelet_name, wavelet_level, order):
#     coeffs = emg_dwpt(signal, wavelet_name, wavelet_level)
#     coeffs = np.array(coeffs)
#     length = coeffs.shape[0]
#     a = range(1, length + 1, 1)
#     a = np.array(a)
#     a = (a / float(length)) ** order
#     coeffs = coeffs * a
#     return np.sum(coeffs)


# def emg_apen(signal, sub_length, threshold):
#     return fai(signal, sub_length, threshold) - fai(signal, sub_length + 1, threshold)


# def fai(signal, sub_length, threshold):
#     dist = []
#     signal = np.array(signal)
#     N = signal.shape[0]

#     if (N - sub_length + 1) == 0:
#         return 0

#     for i in range(0, N - sub_length + 1, 1):
#         sub1 = signal[i:(i + sub_length)]
#         row_dist = []
#         for j in range(0, N - sub_length + 1, 1):
#             sub2 = signal[j:(j + sub_length)]
#             dist_value = abs(sub1 - sub2)
#             dist_value = np.max(dist_value)
#             row_dist.append(dist_value)
#         row_dist = np.array(row_dist)
#         dist.append(row_dist)
#     dist = np.array(dist)
#     cmr = [d[d <= threshold].shape[0] for d in dist]
#     cmr = np.array(cmr)
#     cmr = cmr / float(N - sub_length + 1)
#     cmr = np.log(cmr)
#     cmr = np.sum(cmr)

#     cmr = cmr / float(N - sub_length + 1)
#     return cmr


# def emg_wte(signal, width):
#     wavelet = sig.ricker
#     widths = np.arange(1, width + 1)
#     cwt = sig.cwt(signal, wavelet, widths)
#     wte = []
#     for i in range(cwt.shape[1]):
#         col = cwt[:, i]
#         col = col * col
#         col_energy = np.sum(col)
#         col = col / float(col_energy)
#         col = -(col * np.log(col))
#         wte.append(np.sum(col))
#     wte = np.array(wte)
#     return wte


# def emg_wfe(signal, width):
#     wavelet = sig.ricker
#     widths = np.arange(1, width + 1)
#     cwt = sig.cwt(signal, wavelet, widths)
#     wfe = []
#     for i in range(cwt.shape[0]):
#         row = cwt[i, :]
#         row = row * row
#         row_energy = np.sum(row)
#         row = row / float(row_energy)
#         row = -(row * np.log(row))
#         wfe.append(np.sum(row))
#     wfe = np.array(wfe)
#     return wfe










