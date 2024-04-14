import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from copy import deepcopy as copy

def get_mean_std(group_data, column="Final/test.0/IoU-New"):
    mean = group_data[column].mean()
    std = group_data[column].std()
    return mean, std
data_to_plot = ["loss/_target_", "Final/test.0/IoU-Old", "Final/test.0/IoU-New", "Final/test.0/mIoU"] 
data = pd.read_csv("wandb_seeds.csv", usecols=data_to_plot)
data["loss"] = data["loss/_target_"].apply(lambda x: x.replace("loss.", "").replace("Loss", "").replace("DER", "BACS"))
data_to_plot = data_to_plot[1:]
data1 = copy(data).drop(columns=data_to_plot[1:]).assign(Classes="0-15") # "0-15"
data1["mIoU"] = data1[data_to_plot[0]]

data2 = copy(data).drop(columns=[data_to_plot[i] for i in [0,2]]).assign(Classes="16-20") #"16-20"
data2["mIoU"] = data2[data_to_plot[1]]

data3 = copy(data).drop(columns=data_to_plot[:2]).assign(Classes="all") # "all"
data3["mIoU"] = data3[data_to_plot[2]]

newdata = pd.concat([data1, data2, data3])
fig, ax = plt.subplots() # define the axis object here
sns.set_style("darkgrid")
ax = sns.boxplot(x="Classes", y="mIoU", hue="loss", hue_order=["BACS", "MiB", "Plop"],palette=sns.color_palette("pastel"), data=newdata, ax=ax) 
[ax.axvline(x+.5,color='0.8') for x in ax.get_xticks()]
ax.yaxis.grid(True) # Hide the horizontal gridlines 
plt.show()