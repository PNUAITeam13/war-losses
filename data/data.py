import pandas as pd

data = pd.read_csv("../origin_data/russia_losses_personnel.csv")

data = data[["date", "personnel"]]

data.rename(columns={"personnel": "Losses", "date": "Date"}, inplace=True)
data["Losses"] = data["Losses"].diff().fillna(data["Losses"])

data.to_csv("losses.csv", index=False)
