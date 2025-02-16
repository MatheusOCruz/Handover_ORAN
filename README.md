```bash
pip install ./HandoverRL
python3 main.py
```
The package comes with 4 with synthetic data scenarios for training and testing the model, as well as the option to train a model (from an existing one of from scratch) and to use \
the trained model and plot the results.

The plot created shows the RSRP metric over time and marks where the model decided to trigger the Handover


to choose the scenario being used, just change in the `main.py` the line:

```
gnb = Gnb([1,3])
```
each entry in the vector is an UE, and the scenario represented by the int value can be found in the `gnb.py` file.

To train the model, change the line:

```
 if h.handover_decision(*gnb.get_metrics(rnti)):
```
to:
```
 if h.train_step(*gnb.get_metrics(rnti)):
```

and increase the number of iteration in the for loop, the model given used 100k steps
