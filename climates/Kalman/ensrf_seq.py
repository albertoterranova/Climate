import tqdm
import numpy as np


def ensrf_sequential_fixed_cov(xbm, Xbp, y, H, error, ):

    for i in tqdm.tqdm(range(len(y))):
        if np.isnan(y[i]) == False:
            Hxensp = H[i]@Xbp

            Hxens_m = H[i]@xbm

            PHT = (Xbp@(H[i]@Xbp).T)/29

            HPHTR = ((H[i]@Xbp) @ (H[i]@Xbp).T)/29 + error

            KG = PHT / HPHTR

            KG = KG[:, None]

            innov = y[i] - Hxens_m

            Hxens_p = Hxensp[np.newaxis]

            kg = KG

            xam = xbm[:, np.newaxis] + (kg * innov)

            beta = 1 / (1 + np.sqrt(0.9 / (HPHTR)))

            kg *= beta

            Xap = Xbp - np.dot(kg, Hxens_p)

        else:
            xam = xbm

            Xap = Xbp

            Xbp, xbm = Xap, xam

    return xam, Xap


def ensrf_sequential_fixed_cov(xbm, Xbp, y, H, error, localization, x):

    for i in tqdm.tqdm(range(len(y))):

        if localization is not None:

            if np.isnan(y[i]) == False:

                Hxensp = H[i]@Xbp

                Hxens_m = H[i]@xbm

                numerator = np.multiply(H[i]@localization, (x@(H[i]@x).T)/29)

                denominator = np.multiply(
                    H[i].T@localization@H[i], (H[i]@x @ (H[i]@x).T)/29) + error

                KG = numerator / denominator

                KG = KG[:, None]

                innov = y[i] - Hxens_m

                Hxens_p = Hxensp[np.newaxis]

                kg = KG

                xam = xbm + kg * innov

                beta = 1 / (1 + np.sqrt(0.9 / (denominator)))

                kg *= beta

                Xap = Xbp - np.dot(kg, Hxens_p)

                Xbp, xbm = Xap, xam

            else:

        else:

            if np.isnan(y[i]) == False:

                Hxensp = H[i]@Xbp

                Hxens_m = H[i]@xbm

                numerator = (x@(H[i]@x).T)/29

                denominator = (H[i]@x @ (H[i]@x).T)/29 + error

                KG = numerator / denominator

                KG = KG[:, None]

                innov = y[i] - Hxens_m

                Hxens_p = Hxensp[np.newaxis]

                kg = KG

                xam = xbm + kg * innov

                beta = 1 / (1 + np.sqrt(0.9 / (denominator)))

                kg *= beta

                Xap = Xbp - np.dot(kg, Hxens_p)

                Xbp, xbm = Xap, xam

            else:

                xam = xbm

                Xap = Xbp

                Xbp, xbm = Xap, xam

    return xam, Xap


def ensrf_sequential_fast_1(xbm, Xbp, y, H, error):

    for i in tqdm.tqdm(range(len(y))):

        if np.isnan(y[i]) == False:

            Hxensp = H[i]@Xbp

            Hxens_m = H[i]@xbm

            Hx = H[i]@Xbp

            new_var = Xbp@Hx.T
            numerator = new_var / 29

            denominator = (H[i]@Xbp @ (H[i]@Xbp).T)/29 + error

            KG = numerator / denominator

            KG = KG[:, None]

            innov = y[i] - Hxens_m

            Hxens_p = Hxensp[np.newaxis]

            kg = KG

            xam = xbm[:np.newaxis] + kg * innov

            beta = 1 / (1 + np.sqrt(0.9 / (denominator)))

            kg *= beta

            Xap = Xbp - np.dot(kg, Hxens_p)

            Xbp, xbm = Xap, xam

        else:

            xam = xbm

            Xap = Xbp

            Xbp, xbm = Xap, xam

    return xam, Xap


def ensrf_client(xbm, xbp, y, H, error, client):
    Hxp = np.matmul(H, xbp)
    Hxm = np.matmul(H, xbm)
    PHT = np.matmul(xbp, Hxp.T)/29
    HPHT = np.matmul(Hxp, Hxp.T)/29
    R = np.identity(10359)*0.9
    HPHTR = HPHT+R
    PHT = da.from_array(PHT)
    PHT = client.persist(PHT)
    PHT = client.scatter(PHT, broadcast=True).result()
    HPHTR = da.from_array(HPHTR)
    HPHTR = client.persist(HPHTR)
    HPHTR = client.scatter(HPHTR, broadcast=True).result()
    HPHTR_inv = da.linalg.inv(HPHTR.rechunk((10359, 10359)))
    HPHTR_inv = client.persist(HPHTR_inv)
    K = da.matmul(PHT, HPHTR_inv)
    K = K.persist()
    y = y[~np.isnan(y)].values
    innov = y - Hxm
    innov = client.persist(da.from_array(innov))
    innov = client.scatter(innov, broadcast=True).result()
    add = da.matmul(K, innov)
    add = client.persist(add)
    xam = xbm + add.compute()
    return xam
