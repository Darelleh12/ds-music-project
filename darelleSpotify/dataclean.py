import pandas as pd

dfs: pd.DataFrame = []
dfs.append(pd.read_json("darelleSpotifyExtendedHistory/Streaming_History_Audio_2017-2022_0.json"))
dfs.append(pd.read_json("darelleSpotifyExtendedHistory/Streaming_History_Audio_2022-2023_1.json"))
dfs.append(pd.read_json("darelleSpotifyExtendedHistory/Streaming_History_Audio_2023-2025_2.json"))
dfs.append(pd.read_json("darelleSpotifyExtendedHistory/Streaming_History_Audio_2025-2026_3.json"))

combined = pd.concat(dfs)
combined.reset_index(inplace = True, drop = True)

combined.dropna(subset = "spotify_track_uri", inplace = True)
combined.drop(columns = ["ip_addr", "audiobook_title", "audiobook_uri", "audiobook_chapter_uri", "audiobook_chapter_title", "episode_name", "episode_show_name", "spotify_episode_uri"], inplace = True)

print(combined.info())

combined.to_csv("l-listening-data_2017-26.csv", index = False)
