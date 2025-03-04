import HandoverRL as h
from gnb import *


def main():
    h.init_module(16, 2, "target.pth", "target2.pth")
    np.random.seed(14012003)
    gnb = Gnb([3])

    for _ in range(5000):
        for rnti in range(len(gnb.ues)):
            if h.handover_decision(*gnb.get_metrics(rnti)):
                gnb.handover(rnti)

    gnb.plot_things()



if __name__ == '__main__':
    main()
