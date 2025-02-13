import numpy as np
import pandas as pd
import streamlit as st

# from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Cache data loading
@st.cache_data
def load_data(filepath):
    from pathlib import Path

    loaders_by_format = {
        "csv": pd.read_csv,
        "xlsx": pd.read_excel,
        "db": pd.read_sql,
        "parquet": pd.read_parquet,
        "npy": lambda x: pd.DataFrame(list(np.load(x, allow_pickle=True))),
    }
    extension = Path(filepath).name.split(".")[1]
    df = loaders_by_format[extension](filepath)
    df["track"] = df["filepath"].apply(lambda x: Path(x).name)
    df["genre"] = df["style_genre_discogs400"].apply(lambda x: x.split("---")[0])
    df["style"] = df["style_genre_discogs400"].apply(lambda x: x.split("---")[-1])
    return df


df = load_data("../data/processed/20250212032022_audio_features.npy")

# Sidebar filters
st.sidebar.header("Track Filters")

# Music Style Filter
all_styles = df["style_genre_discogs400"].unique()
selected_styles = st.sidebar.multiselect(
    "Select music styles", options=all_styles, default=[]
)

# Tempo Filter
min_bpm, max_bpm = st.sidebar.slider(
    "Tempo Range (BPM)",
    min_value=int(df["tempocnn_bpm"].min()),
    max_value=int(np.ceil(df["tempocnn_bpm"].max())),
    value=(int(df["tempocnn_bpm"].min()), int(np.ceil(df["tempocnn_bpm"].max()))),
)

# Danceability Sorting
sort_by_danceability = st.sidebar.checkbox("Sort by danceability")

# Arousal & Valence Filters
arousal_range = st.sidebar.slider(
    "Arousal Range", min_value=0, max_value=10, value=(0, 10)
)
valence_range = st.sidebar.slider(
    "Valence Range", min_value=0, max_value=10, value=(0, 10)
)

# Key Filter
st.sidebar.subheader("Key Filter")
key_filter = st.sidebar.checkbox("Enable Key Filtering")
if key_filter:
    key_method = st.sidebar.selectbox(
        "Key Estimation Method", ["temperley", "krumhansl", "edma"]
    )
    selected_key = st.sidebar.selectbox(
        "Key", ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    )
    selected_mode = st.sidebar.selectbox("Mode", ["major", "minor"])

# Apply filters
df_filtered = df.copy()

# Style filter
if selected_styles:
    df_filtered = df_filtered[
        df_filtered["style_genre_discogs400"].isin(selected_styles)
    ]

# Tempo filter
df_filtered = df_filtered[
    (df_filtered["tempocnn_bpm"] >= min_bpm) & (df_filtered["tempocnn_bpm"] <= max_bpm)
]

# Arousal & Valence filter
df_filtered = df_filtered[
    (df_filtered["emomusic_arousal"].between(*arousal_range))
    & (df_filtered["emomusic_valence"].between(*valence_range))
]

# Key filter
if key_filter:
    target_key = f"{selected_key} {selected_mode}"
    key_col = f"key_{key_method}_predict"
    df_filtered = df_filtered[df_filtered[key_col] == target_key]

# Danceability sorting
if sort_by_danceability:
    df_filtered = df_filtered.sort_values("effnet_discogs_danceable", ascending=False)

# Show filtered tracks
st.header("Select Reference Track")
if not df_filtered.empty:
    st.write(f"Found {len(df_filtered)} tracks")
    st.dataframe(
        df_filtered[
            [
                "track",
                "duration",
                "tempocnn_bpm",
                "style_genre_discogs400",
                "emomusic_arousal",
                "emomusic_valence",
                "key_temperley_predict", "key_krumhansl_predict", "key_edma_predict"
            ]
        ].head(100)
    )

    # Track selection
    tracks = df_filtered["track"].unique()
    selected_track = st.selectbox("Choose a reference track:", tracks)
else:
    st.error("No tracks match the current filters!")
    st.stop()

# Similarity settings
st.sidebar.header("Similarity Settings")
similarity_method = st.sidebar.radio(
    "Similarity Method", ["Discogs", "Musicnn", "Hybrid"]
)

weights = {}
if similarity_method == "Hybrid":
    weights["discogs"] = st.sidebar.slider("Discogs Weight", 0.0, 1.0, 0.5)
    weights["musicnn"] = 1 - weights["discogs"]


# Compute similarities
@st.cache_data
def compute_similarities(_df, query_track, method, weights=None):
    # Get query embeddings
    query_idx = _df[_df["track"] == query_track].index[0]
    discogs_query = _df.loc[query_idx, "discogs_embeddings_mean"]
    musicnn_query = _df.loc[query_idx, "musicnn_embeddings_mean"]

    # Compute similarities
    discogs_matrix = np.stack(_df["discogs_embeddings_mean"].values)
    musicnn_matrix = np.stack(_df["musicnn_embeddings_mean"].values)

    discogs_sim = cosine_similarity([discogs_query], discogs_matrix)[0]
    musicnn_sim = cosine_similarity([musicnn_query], musicnn_matrix)[0]

    return discogs_sim, musicnn_sim


discogs_sim, musicnn_sim = compute_similarities(
    df, selected_track, similarity_method, weights
)

# Combine similarities
if similarity_method == "Discogs":
    scores = discogs_sim
elif similarity_method == "Musicnn":
    scores = musicnn_sim
else:
    scores = (weights["discogs"] * discogs_sim) + (weights["musicnn"] * musicnn_sim)

# Create results dataframe
results_df = df[["track", "duration", "tempocnn_bpm", "style_genre_discogs400"]].copy()
results_df["similarity_score"] = scores
results_df = results_df[results_df["track"] != selected_track]  # Remove query track
results_df = results_df.sort_values("similarity_score", ascending=False).head(20)

# Display results
st.header("Similar Tracks")
st.write(f"Top 20 tracks similar to: {selected_track}")
st.dataframe(results_df)

# Audio previews
st.subheader("Track Previews")
for track in results_df["track"].head(5):
    track_row = df[df["track"] == track].iloc[0]
    st.write(
        f"**{track}** (Similarity: {results_df[results_df['track'] == track]['similarity_score'].values[0]:.3f})"
    )
    st.audio(track_row["filepath"])  # Assumes valid audio file paths
