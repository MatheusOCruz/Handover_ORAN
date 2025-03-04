```bash
pip install ./HandoverRL
python3 main.py
```
The package comes with 4 synthetic data scenarios for training and testing the model, as well as the option to train a model (from an existing one or from scratch) and to use the trained model and plot the results.

The plot created shows the RSRP metric over time and marks where the model decided to trigger the Handover


to choose the scenario being used, change the following line in `main.py`:

```
gnb = Gnb([1,3])
```
each entry in the vector is an UE, and the scenario represented by the int value can be found in the `gnb.py` file, and are: 
[](images/Figure_1.png)
[](images/Figure_2.png)
[](images/Figure_3.png)
[](images/Figure_4.png)

To train the model, change the line:

```
 if h.handover_decision(*gnb.get_metrics(rnti)):
```
to:
```
 if h.train_step(*gnb.get_metrics(rnti)):
```

and increase the number of iterations in the for loop, the model given used 100k steps
