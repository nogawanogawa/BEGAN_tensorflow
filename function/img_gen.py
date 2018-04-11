import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 縦横10枚ずつ画像を描画
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        # 描画時に0 ~ 1の間に収める
        mask_0 = (sample <= 0)
        sample[mask_0] = 0
        mask_1 = (sample  > 1)
        sample[mask_1] = 1

        plt.imshow(sample.reshape(3, 32, 32).transpose(1, 2, 0))

    return fig
