import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import matplotlib

matplotlib.use('TkAgg')


class Ue:
    def __init__(self, maxlen):
        ida = range(maxlen)

        self.x = cycle(ida)
        self.save_x = 0  # pra pegar o valor de X sem usar next
        self.gnb = 0

    def next(self):
        self.save_x = next(self.x)
        return self.save_x


class Gnb:
    def __init__(self, cases, noise_std=5):
        x = np.arange(0, 4 * np.pi, 0.01)
        y = np.sin(x)
        self.ues = []
        self.torres = []
        self.onde_fez_handover =[]
        for case in cases:
            
            match case:
            # base
                case 0:
                    gnb_0 = list(map(lambda x: 48 * x - 92, y.tolist()))
                    gnb_1 = list(map(lambda x: (-x - 44), gnb_0))
                    gnb_0 = [x + 140 for x in gnb_0]
            # ruido
                case 1:
                    gnb_0 = list(map(lambda x: 48 * x - 92, y.tolist()))
                    gnb_0 = [x + np.random.normal(0, noise_std) for x in gnb_0]
                    gnb_1 = list(map(lambda x: (-x - 44), gnb_0))
                    gnb_0 = [x + 140 for x in gnb_0]
            # fase
                case 2:
                    gnb_0 = list(map(lambda x: 48 * x - 92 + 140, y.tolist()))
                    gnb_1 = gnb_0[75:] + gnb_0[:75]
            # fase + ruido
                case 3:
                    gnb_0 = list(map(lambda x: 48 * x - 92 + 140, y.tolist()))
                    gnb_0 = [x + np.random.normal(0, noise_std) for x in gnb_0]
                    gnb_1 = gnb_0[75:] + gnb_0[:75]



            self.ues.append(Ue(len(gnb_0)))
            self.torres.append([gnb_0.copy(), gnb_1.copy()])
            self.onde_fez_handover.append([])



        
        
    def handover(self,rnti):
        self.onde_fez_handover[rnti].append((self.ues[rnti].save_x, self.ues[rnti].gnb))  # pro plot
        self.ues[rnti].gnb ^= 1

    def get_metrics(self, rnti) -> tuple:
        ue = self.ues[rnti]
        index = ue.next()
        return self.torres[rnti][0][index], self.torres[rnti][1][index], ue.gnb, rnti

    def plot_things(self):
        x = np.arange(0, 4 * np.pi, 0.01)
        for rnti in range(len(self.ues)):
            plt.figure() 
            
            gnb_0 = self.torres[rnti][0]
            gnb_1 = self.torres[rnti][1]

            gnb_0_points = [gnb_0[i] for i, gnb_id in self.onde_fez_handover[rnti] if gnb_id == 0]
            gnb_1_points = [gnb_1[i] for i, gnb_id in self.onde_fez_handover[rnti] if gnb_id == 1]

            x2_0 = [x[i] for i, j in self.onde_fez_handover[rnti] if not j]
            x2_1 = [x[i] for i, j in self.onde_fez_handover[rnti] if j]

            plt.plot(x, gnb_0, color="blue")
            plt.plot(x, gnb_1, color="green")

            plt.scatter(x2_0, gnb_0_points, color='red', zorder=5)
            plt.scatter(x2_1, gnb_1_points, color='orange', zorder=5)
            
            plt.grid(True) 
        
        plt.show()
