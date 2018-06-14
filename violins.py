import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from scipy import stats
import numpy as np


def violin(y=None, split=None, data=None, **kwargs):

    f, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 10), gridspec_kw={"height_ratios": [5, 1]}
    )

    ax1 = sns.violinplot(x=split, y=y, data=data, ax=ax1, **kwargs)

    groups = data.groupby(split)[y]
    q1, q2 = set(tips[split])
    a = data.query(f'{split} == "{q1}"')[y].values
    b = data.query(f'{split} == "{q2}"')[y].values

    n1, n2 = groups.size()
    o1, o2 = groups.std()
    m1, m2 = groups.mean()

    s1, s2 = o1 / np.sqrt(n1), o2 / np.sqrt(n2)
    s1n, s2n = s1 ** 2 / n1, s2 ** 2 / n2
    df = (s1n + s2n) ** 2 / (s1n ** 2 / (n1 - 1) + s2n ** 2 / (n2 - 1))

    t, p = stats.ttest_ind(a, b)

    t_min = t_dist.ppf(0.0001, df)
    t_max = t_dist.ppf(0.9999, df)

    x = np.linspace(t_min, t_max, 100)
    t_pdf = t_dist.pdf(x, df)

    ax2.plot(x, t_pdf, "k", lw=1)
    ax2.set_xlim([t_min, t_max])
    ax2.set_ylim([0, 1.1 * np.max(t_pdf)])

    x_sig = np.linspace(t_dist.ppf(0.025, df), t_dist.ppf(0.975, df), 100)
    t_sig = t_dist.pdf(x_sig, df)
    ax2.fill_between(x_sig, t_sig, facecolor="0.75", alpha=0.4)

    ax2.vlines(t, 0, 1.1 * np.max(t_pdf))

    plt.show()
    return t, p


if __name__ == "__main__":
    tips = sns.load_dataset("tips")
