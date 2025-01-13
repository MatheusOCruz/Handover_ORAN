import HandoverRL as h
from gnb import *


def main():
    h.init_module(16, 2, "target.pth", "target.pth")
    np.random.seed(14012003)
    gnb = Gnb(1)

    for _ in range(5000):

        if h.handover_decision(*gnb.get_metrics()):
            gnb.handover()

    gnb.plot_things()



if __name__ == '__main__':
    main()
