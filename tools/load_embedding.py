import torch
from torch_geometric.nn import DeepGraphInfomax, GraphSAGE
from tools import train_utils
from models.pyg.dgi_inductive import Encoder as DGIEncoder
from models.pyg.dgi_inductive import corruption as corruption_dgi

from paths_globals import *

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Determine the device to use (GPU or CPU)
def determine_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def generate_random_embeddings(num_nodes, embedding_dim):
    embeddings = np.random.normal(0, 1, (num_nodes, embedding_dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize each vector
    return embeddings


# Create the model based on the algorithm used during training
def create_model(dimension, dataset, device, algorithm, embedding_config):
    if algorithm == DGI:
        # Recreate the DeepGraphInfomax model
        model = DeepGraphInfomax(
            hidden_channels=dimension,
            encoder=DGIEncoder(
                dataset.num_features,
                dimension,
                num_layers=embedding_config["num_layers"],
            ),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption_dgi,
        ).to(device)
    elif algorithm == GRAPHSAGE:
        model = GraphSAGE(
            dataset.num_node_features,
            hidden_channels=dimension,
            num_layers=embedding_config["num_layers"],
        ).to(device)
    else:
        raise ValueError("Check algorithm name")
    return model


# Get embedding vectors from the trained model
def get_embedding_vectors(model, data, algorithm, device):
    # Convert the sparse tensors to dense before passing them to the model
    data.x = data.x.to_dense()
    data.edge_index = data.edge_index.to_dense() if hasattr(data.edge_index, "to_dense") else data.edge_index

    if algorithm == DGI:
        with torch.no_grad():
            embedding, _, _ = model(data.x.to(device), data.edge_index.to(device), batch_size=data.num_nodes)
    elif algorithm == GRAPHSAGE:
        with torch.no_grad():
            embedding = model(data.x.to(device), data.edge_index.to(device))
    return embedding


# Load the embeddings based on the model and algorithm
def load_embedding(dimension, embedding_config=None, dataset_params=None, data=None, iteration=None, algorithm=None):

    load_path = CREATE_MODELS_PATH(dataset_params=dataset_params, embedding_name=algorithm, embedding_dim=dimension)
    if algorithm in [ASNE, VERSE, NODE2VEC]:
        embedding = np.load(osp.join(load_path, EMBEDDING_FILE_NAME(iteration)))
        return embedding

    parameter_dict = train_utils.get_best_parameter_dict(
        embedding_method=algorithm,
        dataset_params=dataset_params,
        dimensions=[dimension],
    )

    embedding_config[CONFIG_DIMENSION_KEY] = dimension

    params = parameter_dict[dimension]
    for k, v in params.items():
        embedding_config[k] = v

    device = determine_device()

    model = create_model(dimension, data, device, algorithm, embedding_config)

    model_path = osp.join(load_path, MODEL_FILE_NAME(iteration))
    # Load the saved state dictionary into the model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()

    # Get the embedding vectors from the model
    embedding = get_embedding_vectors(model, data, algorithm, device)

    # Return the embeddings as a NumPy array
    return embedding.detach().cpu().numpy()
