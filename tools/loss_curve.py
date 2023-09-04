import argparse
import matplotlib.pyplot as plt

def get_epoch_loss(log_file):
    log_ = []
    # read log
    with open(log_file, 'r') as log_f:
        contents = log_f.readlines()
        for line in contents:
            if line.endswith('* \n'):
                continue
            info = line.strip().split('||')
            epoch = [i for i in info if 'Epoch' in i][0]
            loss = [i for i in info if 'Loss' in i][0]
            log_.append([int(epoch.split('Epoch:')[-1].strip()), float(loss.split('Loss:')[-1].strip())])
    del(info, epoch, loss, contents)
    # process
    epochs = log_[-1][0]
    losses = [0] * epochs
    for i in range(1, epochs+1):
        l = [l[-1] for l in log_ if l[0] == i]
        losses[i-1] = sum(l) / len(l)
    return epochs, losses

def draw_loss_curve(epochs: int, losses: list, save_: str):
    epochs_range = list(range(1, epochs + 1))
    plt.plot(epochs_range, losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Classification Loss Over Training')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(save_)

if __name__=="__main__":
    parser = argparse.ArgumentParser('Classification inference code!!')
    parser.add_argument('--log', type=str, default='weights/test/log.log', help='log file path')
    parser.add_argument('--out', type=str, default='weights/test/loss_curve.jpg', help='out jpg path')
    opt = parser.parse_args()

    epochs, losses = get_epoch_loss(opt.log)
    draw_loss_curve(epochs, losses, save_= opt.out)


