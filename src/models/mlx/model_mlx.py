import mlx.core as mx
import mlx.nn as nn

# --- NCF Model ---
class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=128, hidden_layers=[256, 128, 64],
                 dropout_rate=0.2, use_features=False, n_genres=0, n_languages=0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.use_features = use_features

        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        if use_features:
            self.genre_embedding = nn.Embedding(n_genres, 32)
            self.language_embedding = nn.Embedding(n_languages, 16)
            mlp_input_dim = 2 * embedding_dim + 32 + 16
        else:
            mlp_input_dim = 2 * embedding_dim

        mlp_layers = []
        input_dim = mlp_input_dim
        for hidden_dim in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        self.mlp_layers = nn.Sequential(*mlp_layers)

        self.prediction = nn.Linear(hidden_layers[-1] + embedding_dim, 1)

    def __call__(self, user_ids, item_ids, genre_ids=None, language_ids=None):
        user_embedding_gmf = self.user_embedding_gmf(user_ids)
        item_embedding_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_embedding_gmf * item_embedding_gmf

        user_embedding_mlp = self.user_embedding_mlp(user_ids)
        item_embedding_mlp = self.item_embedding_mlp(item_ids)
        if self.use_features and genre_ids is not None and language_ids is not None:
            genre_embedding = self.genre_embedding(genre_ids)
            language_embedding = self.language_embedding(language_ids)
            mlp_input = mx.concatenate([user_embedding_mlp, item_embedding_mlp, genre_embedding, language_embedding], axis=1)
        else:
            mlp_input = mx.concatenate([user_embedding_mlp, item_embedding_mlp], axis=1)
        mlp_output = self.mlp_layers(mlp_input)
        concat_output = mx.concatenate([gmf_output, mlp_output], axis=1)
        prediction = self.prediction(concat_output)
        return mx.sigmoid(prediction).squeeze()

# --- Matrix Factorization ---
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=128, use_bias=True):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.use_bias = use_bias
        if use_bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.global_bias = mx.zeros((1,))

    def __call__(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        prediction = (user_emb * item_emb).sum(axis=1)
        if self.use_bias:
            prediction += self.user_bias(user_ids).squeeze()
            prediction += self.item_bias(item_ids).squeeze()
            prediction += self.global_bias
        return mx.sigmoid(prediction)

# --- Hybrid Model ---
class HybridModel(nn.Module):
    def __init__(self, n_users, n_items, n_genres, n_languages, embedding_dim=128, content_dim=64, hidden_dims=[256,128,64]):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.genre_embedding = nn.Embedding(n_genres, content_dim // 2)
        self.language_embedding = nn.Embedding(n_languages, content_dim // 4)
        total_dim = 2*embedding_dim + content_dim // 2 + content_dim // 4
        layers = []
        input_dim = total_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def __call__(self, user_ids, item_ids, genre_ids, language_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        genre_emb = self.genre_embedding(genre_ids)
        lang_emb = self.language_embedding(language_ids)
        features = mx.concatenate([user_emb, item_emb, genre_emb, lang_emb], axis=1)
        output = self.mlp(features)
        return mx.sigmoid(output).squeeze()

# --- Model Factory ---
def get_model(model_type: str, **kwargs):
    if model_type == 'ncf':
        return NCF(**kwargs)
    elif model_type == 'mf':
        return MatrixFactorization(**kwargs)
    elif model_type == 'hybrid':
        return HybridModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")