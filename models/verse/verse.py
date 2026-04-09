from subprocess import CalledProcessError

from models.verse.convert import process
from models.verse.embedding import Embedding
import pandas as pd
import subprocess
import multiprocessing

from paths_globals import *

from tools.train_utils import prepare_node_classification_data, prepare_link_prediction_data


def train_model(
    edge_list_path: str,
    embedding_config: Dict,
    save_path: str,
    downstream_path: str,
    seed: int,
    n_jobs: int = -1,
) -> float:

    dimension = embedding_config[CONFIG_DIMENSION_KEY]

    bcsr_path = osp.join(osp.dirname(edge_list_path), BCSR_GRAPH_FILE_NAME)

    if not osp.isfile(bcsr_path):
        process(
            format="edgelist",
            matfile_variable_name="network",
            undirected=True,
            sep=" ",
            input=edge_list_path,
            output=bcsr_path,
        )

    bin_embedding_path = save_path.replace(".npy", ".bin")
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() + 1 + n_jobs
    if n_jobs == 0:
        raise RuntimeError("Number of threads can not be zero!")

    command = (
        f'{osp.join(VERSE_SRC_PATH,"verse")} -input "{bcsr_path}" -output {bin_embedding_path} -dim {dimension}'
        f' -alpha {embedding_config["alpha"]} -threads {n_jobs} -nsamples {embedding_config["n_neg_samples"]} '
        f'-lr {embedding_config["lr"]} -rng_seed {seed}'
    )
    try:
        completed_process = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True
        )
        print(completed_process.stdout)
        print(completed_process.stderr)
    except CalledProcessError as e:
        print("stdout")
        print(e.stdout)
        print("output")
        print(e.output)
        print("error")
        print(e.stderr)

    embedding = Embedding(bin_embedding_path, dimension).embeddings

    np.save(save_path, embedding)

    if osp.isfile(bin_embedding_path):
        os.remove(bin_embedding_path)

    downstream_df = pd.read_csv(downstream_path, index_col=0)
    if DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY in list(downstream_df):
        X_train, y_train, X_val, y_val = prepare_link_prediction_data(
            downstream_df=downstream_df,
            edge_list=edge_list_path,
            embedding=embedding,
            return_val_data=True,
        )
    else:
        X_train, y_train, X_val, y_val = prepare_node_classification_data(
            downstream_df=downstream_df,
            embedding=embedding,
            return_val_data=True,
        )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    return clf.score(X_val, y_val)
