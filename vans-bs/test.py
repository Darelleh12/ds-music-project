import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataV = pd.read_csv("vans-listening-data_2021-2026.csv")
dataL = pd.read_csv("l-listening-data_2017-26.csv")

dataV["ts"] = pd.to_datetime(dataV["ts"])
dataL["ts"] = pd.to_datetime(dataL["ts"])

dataV = dataV[dataV["ts"].dt.year == 2025]
dataL = dataL[dataL["ts"].dt.year == 2025]
dataV = dataV[dataV["ms_played"] >= 10000]
dataL = dataL[dataL["ms_played"] >= 10000]

print(dataV)

# total_time = dataV["ms_played"].sum()

# print(f"Minutes Listened: {total_time/(1000*60)}, Hours: {total_time/(1000*60*60)}")
# print(f"Average mins per day: {total_time/(1000*60*365)}")
# print(f"Standard deviation of time per day: {data.groupby(data['ts'].dt.date)["ms_played"].sum().std()/(1000*60)}")
# print(f"Unique number of artists: {len(data["master_metadata_album_artist_name"].value_counts())}")
# print(f"Unique number of songs: {len(data["master_metadata_track_name"].value_counts())}")
# print(f"Skip rate: {len(data[data["skipped"] == True]) / len(data["skipped"])}")


# ! ------------ !
# top artists

# fig, (axVT, axLT) = plt.subplots(nrows = 1, ncols = 2)

# vTAT = (dataV.groupby("master_metadata_album_artist_name")["ms_played"].sum()/(1000*60*60)).sort_values(ascending=False)[0:10]
# lTAT = (dataL.groupby("master_metadata_album_artist_name")["ms_played"].sum()/(1000*60*60)).sort_values(ascending=False)[0:10]

# axVT.bar(vTAT.index, vTAT.values)
# axVT.set_xticklabels(vTAT.index, rotation=90)

# axLT.bar(lTAT.index, lTAT.values)
# axLT.set_xticklabels(lTAT.index, rotation=90)

# fig.suptitle("Top 10 Artists in Minutes Streamed")

# fig.tight_layout()
# plt.show()

# ! ------------ !
# listening based on time of day

# vTime = (dataV.groupby(dataV["ts"].dt.hour)["ms_played"].sum().sort_index()/(1000*60))/365
# lTime = (dataL.groupby(dataL["ts"].dt.hour)["ms_played"].sum().sort_index()/(1000*60))/365

# fig, ax = plt.subplots(nrows = 1, ncols = 1)

# ax.plot(vTime, color="red", label="User 1")
# ax.plot(lTime, color="blue", label="User 2")

# ax.set_xlabel("Time of Day")
# ax.set_ylabel("Average minutes listened")
# ax.legend()

# fig.suptitle("Average Minutes Listened per Hour")

# fig.tight_layout()
# plt.show()


# ! ------------------ !
# avg listening that is top artists

# vTopArtists = list(dataV.groupby("master_metadata_album_artist_name")["ms_played"].sum().sort_values(ascending=False)[0:10].index)
# lTopArtists = list(dataL.groupby("master_metadata_album_artist_name")["ms_played"].sum().sort_values(ascending=False)[0:10].index)

# vHourArtists = pd.DataFrame()
# for artist in vTopArtists:
#     vHourArtists = pd.concat([vHourArtists, dataV[dataV["master_metadata_album_artist_name"] == artist]])

# lHourArtists = pd.DataFrame()
# for artist in lTopArtists:
#     lHourArtists = pd.concat([lHourArtists, dataL[dataL["master_metadata_album_artist_name"] == artist]])

# vTimeArtists = (vHourArtists.groupby(vHourArtists["ts"].dt.hour)["ms_played"].sum().sort_index()/(1000*60))/365
# lTimeArtists = (lHourArtists.groupby(lHourArtists["ts"].dt.hour)["ms_played"].sum().sort_index()/(1000*60))/365

# fig, ax = plt.subplots(nrows = 1, ncols = 1)

# ax.plot(vTimeArtists, color = "red", label="User 1")
# ax.plot(lTimeArtists, color="blue", label="User 2")

# ax.set_xlabel("Time of Day")
# ax.set_ylabel("Average Minutes Listened to Top Artist")
# ax.legend()

# fig.suptitle("Average Minutes Listened to Top Artist per Hour")

# fig.tight_layout()
# plt.show()


# ! -------------- !
# avg listening a day during months

# vMonthDaily = dataV.groupby(dataV["ts"].dt.month)["ms_played"].sum().sort_index(ascending=True)/(1000*60)
# for i in range(1, 13):
#     if i in [1, 3, 5, 7, 8, 10, 12]:
#         vMonthDaily[i] /= 31
#     elif i == 2:
#         vMonthDaily[i] /= 28
#     else:
#         vMonthDaily[i] /= 30

# lMonthDaily = dataL.groupby(dataL["ts"].dt.month)["ms_played"].sum().sort_index(ascending=True)/(1000*60)
# for i in range(1, 13):
#     if i in [1, 3, 5, 7, 8, 10, 12]:
#         lMonthDaily[i] /= 31
#     elif i == 2:
#         lMonthDaily[i] /= 28
#     else:
#         lMonthDaily[i] /= 30

# fig, ax = plt.subplots(nrows = 1, ncols = 1)

# ax.plot(vMonthDaily, color = "red", label="User 1")
# ax.plot(lMonthDaily, color="blue", label="User 2")

# ax.set_xlabel("Month")
# ax.set_ylabel("Average Daily Minutes Listened")
# ax.legend()

# fig.suptitle("Average Daily Minutes Listened per Month")

# fig.tight_layout()
# plt.show()


# ! ---------- !
# shuffles and skips

# skipsV = dataV[dataV["skipped"] == True]
# skipsL = dataL[dataL["skipped"] == True]

# vSkipRate = len(skipsV[skipsV["shuffle"] == True])/len(skipsV)
# lSkipRate = len(skipsL[skipsL["shuffle"] == True])/len(skipsL)

# print(vSkipRate, lSkipRate)