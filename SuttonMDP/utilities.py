import pickle
import os.path as path


def pickle_load(file_path: str):
    if path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def pickle_dump(file_path: str, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def pickle_load_or_execute(file_path: str, data_generator, data_generator_args, use_cached_file=True):
    if use_cached_file and path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        results = data_generator(*data_generator_args)
        # pickle_dump(file_path, results)
        return results
