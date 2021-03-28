from warnings import simplefilter
simplefilter('ignore')
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import linregress

##############################################################
def my_metric(y_true, y_pred):
    ### Calcular a precisão baseado na distância entre as retas e a diferença angular
    gt_batch = y_true
    pred_batch = y_pred

    x = np.linspace(0, 127, 128, dtype=np.int)

    P = 0

    for z in range(len(gt_batch)):

        gt = np.squeeze(gt_batch[z])
        pred = np.squeeze(pred_batch[z])

        D = 0
        mh = 1 # Valor padrão para caso não exista uma hough transform

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(pred, theta=tested_angles)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1)):
            (xh0, yh0) = dist * np.array([np.cos(angle), np.sin(angle)])
            mh = np.tan(angle + np.pi/2)

        maximus = []
        top = 0.985*np.max(gt)
        for i in range(128):
            for j in range(128):
                if gt[i][j] > top:
                    maximus.append([j, i])
                    if len(maximus) > 20:
                        break
            if len(maximus) > 20:
                break

        line = linregress(maximus)

        if np.isnan(line.slope):
            mc = 999999
            central = [[maximus[0][0], i] for i in x]
        else:
            mc = line.slope
            bc = line.intercept
            central = [[i, int(mc*i + bc)] for i in x]
        hough = [[i, int(mh*(i - xh0) + yh0)] for i in x]

        mp = -1/mh
        perps = []

        for par in hough:
            xp0 = par[0]
            yp0 = par[1]
            perps.append([[i, int(mp*(i - xp0) + yp0)] for i in x])

        xc = []
        yc = []
        xh = []
        yh = []
        for linha_perp in perps:
            for par in linha_perp:
                if par in central and (0 <= par[1] <= 127):
                    for par2 in linha_perp:
                        if par2 in hough and (0 <= par2[1] <= 127):
                            xc.append(par[0])
                            yc.append(par[1])
                            xh.append(par2[0])
                            yh.append(par2[1])
        
        theta_c = np.arctan(mc)
        theta_h = np.arctan(mh)

        if len(xh) > 0:
            for i in range(len(xh)):
                D += np.sqrt((xc[i] - xh[i])**2 + (yc[i] - yh[i])**2)

            D /= (127*np.sqrt(2)*(i + 1))
            p = (1 - D)*np.absolute(np.cos(theta_c - theta_h))
            P += p

    return (P/len(gt_batch))

##############################################################
def w1_metric(y_true, y_pred):
    ### Calcular o coeficiente angular da reta ground truth e da reta predita e subtraí-los. Quanto menor esse valor, melhor é a predição.
    gt_batch = y_true
    pred_batch = y_pred

    x = np.linspace(0, 127, 128, dtype=np.int)

    P = 0

    for z in range(len(gt_batch)):

        gt = np.squeeze(gt_batch[z])
        pred = np.squeeze(pred_batch[z])

        mh = 1 # Valor padrão para caso não exista uma hough transform

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(pred, theta=tested_angles)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1)):
            (xh0, yh0) = dist * np.array([np.cos(angle), np.sin(angle)])
            mh = np.tan(angle + np.pi/2)

        maximus = []
        top = 0.985*np.max(gt)
        for i in range(128):
            for j in range(128):
                if gt[i][j] > top:
                    maximus.append([j, i])
                    if len(maximus) > 20:
                        break
            if len(maximus) > 20:
                break
        
        line = linregress(maximus)
        
        if np.isnan(line.slope): 
            mc = 999999
            bc = 0
        else:
            mc = line.slope
            bc = line.intercept
        
        theta_c = np.arctan(mc)
        theta_h = np.arctan(mh)

        D = np.abs(theta_c - theta_h)
        p = 1 - (D / (np.pi/2))
        P += p
    return (P/len(gt_batch))

##############################################################
def w2_metric(y_true, y_pred, mode='f1'):
    ### Cálculo dos VP, FP e FN
    gt_batch = y_true
    pred_batch = y_pred
    result = 0

    x = np.linspace(0, 127, 128, dtype=np.int)

    for z in range(len(y_pred)):

        gt = np.squeeze(gt_batch[z])
        pred = np.squeeze(pred_batch[z])

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(pred, theta=tested_angles)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1)):
            (xh0, yh0) = dist * np.array([np.cos(angle), np.sin(angle)])
            mh = np.tan(angle + np.pi/2)

        T = 25
        VP = 0
        FP = 0
        FN = 0

        maximus = []
        top = 0.985*np.max(gt)
        for i in range(128):
            for j in range(128):
                if gt[i][j] > top:
                    maximus.append([j, i])
                    if len(maximus) > 20:
                        break
            if len(maximus) > 20:
                break

        interval_central = [] ### Índices da reta central que possui pontos dentro da imagem
        interval_hough = [] ### Índices da reta hough que possui pontos dentro da imagem
        
        line = linregress(maximus)
        if np.isnan(line.slope): 
            central = [[int(maximus[0][0]), i] for i in x]
        else:
            mc = line.slope
            bc = line.intercept
            central = [[i, int(mc*i + bc)] for i in x]

        hough = [[i, int(mh*(i - xh0) + yh0)] for i in x]

        for i in range(len(central)):
            if 0 <= central[i][1] <= 127:
                interval_central.append(i)

        for i in range(len(hough)):
            if 0 <= hough[i][1] <= 127:
                interval_hough.append(i)

        if not len(interval_hough):
            ### y = mh*(x - xh0) + yh0 (com y=0)
            ### -yh0/mh + xh0 = x
            ### -yh0/mh ~ 0 -> x ~ xh0
            hough = [[int(xh0), i] for i in x]
            for i in range(len(hough)):
                if 0 <= hough[i][1] <= 127:
                    interval_hough.append(i)

        for i in interval_hough:
            in_range = False
            for j in interval_central:
                if np.sqrt((hough[i][0] - central[j][0])**2 + (hough[i][1] - central[j][1])**2) <= T:
                    in_range = True
                    break
            if in_range:
                VP += 1

        for i in interval_hough:
            in_range = False
            for j in interval_central:
                if np.sqrt((hough[i][0] - central[j][0])**2 + (hough[i][1] - central[j][1])**2) <= T:
                    in_range = True
                    break
            if not in_range:
                FP += 1

        for i in interval_central:
            in_range = False
            for j in interval_hough:
                if np.sqrt((central[i][0] - hough[j][0])**2 + (central[i][1] - hough[j][1])**2) <= T:
                    in_range = True
                    break
            if not in_range:
                FN += 1

        if mode == 'f1':
            result += (2*VP/(2*VP + FP + FN))
        elif mode == 'precision':
            result += (VP/(VP + FP))
        elif mode == 'recall':
            result += (VP/(VP + FN))
        
    return result/len(gt_batch)
