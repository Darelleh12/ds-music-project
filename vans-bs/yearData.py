import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./vans-listening-data_2021-2026.csv")

def getYearData(data: pd.DataFrame, year: int) -> pd.DataFrame:
    newData = data.copy()
    newData["ts"] = pd.to_datetime(data["ts"])
    newData = newData[newData["ts"].dt.year == year]
    newData.reset_index(inplace = True, drop = True)
    return newData

# functions below assumes a single years data as input

months = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

def makeHist(data1: pd.DataFrame, data2: pd.DataFrame):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    data1["month"] = data1["ts"].dt.month
    data2["month"] = data2["ts"].dt.month

    ax0.bar(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], data1["month"].value_counts().sort_index())
    ax0.set_xticklabels(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], rotation = 45)
    ax0.set_yticks(np.arange(0, 3600, step=500))
    ax1.bar(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], data2["month"].value_counts().sort_index())
    ax1.set_xticklabels(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], rotation = 45)
    ax1.set_yticks(np.arange(0, 3600, step=500))
    fig.tight_layout()
    plt.show()

def topArtists(data1: pd.DataFrame, data2: pd.DataFrame):
    art1 = data1["master_metadata_album_artist_name"].value_counts().sort_values(ascending=False)[0:10]
    art2 = data2["master_metadata_album_artist_name"].value_counts().sort_values(ascending=False)[0:10]

    album1 = data1["master_metadata_album_album_name"].value_counts().sort_values(ascending=False)[0:10]
    album2 = data2["master_metadata_album_album_name"].value_counts().sort_values(ascending=False)[0:10]
    
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows = 2, ncols = 2)

    ax0.bar(art1.index, art1.values)
    ax0.set_xticklabels(art1.index, rotation=90)

    ax1.bar(art2.index, art2.values)
    ax1.set_xticklabels(art2.index, rotation=90)

    ax2.bar(album1.index, album1.values)
    ax2.set_xticklabels(album1.index, rotation=90)

    ax3.bar(album2.index, album2.values)
    ax3.set_xticklabels(album2.index, rotation=90)

    fig.tight_layout()
    plt.show()

# testing

d24 = getYearData(data, 2024)
d25 = getYearData(data, 2025)

# months = pd.DataFrame(d24["ts"].dt.month.value_counts().sort_index().values, index=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

# print(d24["ts"].dt.month.value_counts().sort_index().values)
# print(months)

# print(d24["ts"].dt.day.value_counts())

# makeHist(d24, d25)

print(d24.info())

topArtists(d24, d25)