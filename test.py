import pandas as pd

cls = [
        "interlocking tiles",
        "asphalt", 
        "repair tiles",
        "interlocking tiles",
        "paint -fade-",
        "paint -good-",
        "repair tiles",
        "interlocking tiles",
       ]

conf = [
        0.9452,
        0.1236, 
        0.3388, 
        0.0678, 
        0.1634, 
        0.8654, 
        0.3627, 
        0.4573, 
       ]

data = {"class": cls, "conf": conf}

df = pd.DataFrame(data)
df.sort_values(by=['conf'], inplace=True, ascending=False)
df.sort_values(by=['class'], inplace=True, ascending=True)

print(list(df.loc[:, "class"]))