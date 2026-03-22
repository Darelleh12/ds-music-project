import pandas as pd

dfs: pd.DataFrame = []
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2019-2021_0.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2021_1.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2021-2022_2.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2022_3.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2022-2023_4.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2023_5.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2023-2024_6.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2024-2025_7.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2025_8.json"))
dfs.append(pd.read_json("./Spotify Extended Streaming History/Streaming_History_Audio_2025-2026_9.json"))

combined = pd.concat(dfs)
combined.reset_index(inplace = True, drop = True)

combined.dropna(subset = "spotify_track_uri", inplace = True)
combined.drop(columns = ["ip_addr", "audiobook_title", "audiobook_uri", "audiobook_chapter_uri", "audiobook_chapter_title", "episode_name", "episode_show_name", "spotify_episode_uri"], inplace = True)

print(combined.info())

combined.to_csv("vans-listening-data_2021-2026.csv", index = False)