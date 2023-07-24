import pandas as pd

PATH = "/media/Libri2Mix/mixture_train-360_mix_clean.csv"
out = "/media/Libri2Mix/mixture_train-360_mix_clean_metadata.csv"


prefix = "/media/Libri2Mix/train-360/"

df = pd.read_csv(PATH)
for col in ["mixture_path", "source_1_path", "source_2_path"]:
    if col == "mixture_path":
        dir_name = "mix_clean"
    elif col == "source_1_path":
        dir_name = "s1"
    else : 
        dir_name = "s2"
    df[col] = df[col].apply(lambda x: prefix + dir_name + "/" + x.split("/")[-1])
    df.to_csv(out, index=False)