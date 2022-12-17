import pickle

PICKLE_DIR = "/home/solnik/floor_plans/Final Projects/pickles"


def save(obj, name, pickle_dir=PICKLE_DIR):
    print(f"SAVING {pickle_dir}/{name} ")
    with open(f"{pickle_dir}/{name}", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(name, pickle_dir=PICKLE_DIR):
    with open(f"{pickle_dir}/{name}", "rb") as handle:
        return pickle.load(handle)
