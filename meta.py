import pandas as pd

test_PATH = "/media/ssd1/users/hj/Libri2Mix/mixture_test_mix_clean.csv"
test_out = "/media/ssd1/users/hj/Libri2Mix/mixture_test_mix_clean_metadata_in_ssd.csv"

train_PATH = "/media/ssd1/users/hj/Libri2Mix/mixture_train-360_mix_clean.csv"
train_out = "/media/ssd1/users/hj/Libri2Mix/mixture_train-360_mix_clean_metadata_in_ssd.csv"


test_prefix = "/media/ssd1/users/hj/Libri2Mix/test/"
train_prefix = "/media/ssd1/users/hj/Libri2Mix/train-360/"



df = pd.read_csv(train_PATH)
for col in ["mixture_path", "source_1_path", "source_2_path"]:
    if col == "mixture_path":
        dir_name = "mix_clean"
    elif col == "source_1_path":
        dir_name = "s1"
    else : 
        dir_name = "s2"
    df[col] = df[col].apply(lambda x: train_prefix + dir_name + "/" + x.split("/")[-1])
    df.to_csv(train_out, index=False)

df = pd.read_csv(test_PATH)
for col in ["mixture_path", "source_1_path", "source_2_path"]:
    if col == "mixture_path":
        dir_name = "mix_clean"
    elif col == "source_1_path":
        dir_name = "s1"
    else : 
        dir_name = "s2"
    df[col] = df[col].apply(lambda x: test_prefix + dir_name + "/" + x.split("/")[-1])
    df.to_csv(test_out, index=False)