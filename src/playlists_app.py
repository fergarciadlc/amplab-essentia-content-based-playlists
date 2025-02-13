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
                "key_temperley_predict",
                "key_krumhansl_predict",
                "key_edma_predict",
            ]
        ].head(100)
    )

    # Track selection
    tracks = df_filtered["track"].unique()
    selected_track = st.selectbox("Choose a reference track:", tracks)
else:
    st.error("No tracks match the current filters!")
    st.stop()


# Compute similarities
@st.cache_data
def compute_similarities(_df, query_track):
    # Get query embeddings
    query_idx = _df[_df["track"] == query_track].index[0]
    discogs_query = _df.loc[query_idx, "discogs_embeddings_mean"]
    musicnn_query = _df.loc[query_idx, "musicnn_embeddings_mean"]

    # Compute similarities using custom function
    _df = _df.copy()
    _df["discogs_similarity"] = _df["discogs_embeddings_mean"].apply(
        lambda x: cosine_similarity(x, discogs_query)
    )
    _df["musicnn_similarity"] = _df["musicnn_embeddings_mean"].apply(
        lambda x: cosine_similarity(x, musicnn_query)
    )

    return _df


# Compute similarities
similarity_df = compute_similarities(df, selected_track)


# Create separate result dataframes
def prepare_results(df, similarity_col, query_track):
    return (
        df[df["track"] != query_track]
        .sort_values(similarity_col, ascending=False)
        .head(10)[
            [
                "track",
                "duration",
                "tempocnn_bpm",
                "style_genre_discogs400",
                similarity_col,
            ]
        ]
        .reset_index(drop=True)
    )


discogs_results = prepare_results(similarity_df, "discogs_similarity", selected_track)
musicnn_results = prepare_results(similarity_df, "musicnn_similarity", selected_track)

# Display results side by side
st.header("Similar Tracks Comparison")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Discogs Embedding Results")
    st.dataframe(discogs_results.style.format({"discogs_similarity": "{:.3f}"}))

with col2:
    st.subheader("Musicnn Embedding Results")
    st.dataframe(musicnn_results.style.format({"musicnn_similarity": "{:.3f}"}))

# # Audio previews
# st.subheader("Track Previews")
# for track in results_df["track"].head(5):
#     track_row = df[df["track"] == track].iloc[0]
#     st.write(
#         f"**{track}** (Similarity: {results_df[results_df['track'] == track]['similarity_score'].values[0]:.3f})"
#     )
#     st.audio(track_row["filepath"])  # Assumes valid audio file paths
