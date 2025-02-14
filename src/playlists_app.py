# from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class Columns:
    # columns after renaming
    filepath: str = "filepath"
    bpm: str = "bpm"
    duration: str = "duration"
    arousal: str = "arousal"
    valence: str = "valence"
    key_temperley: str = "key_temperley"
    key_krumhansl: str = "key_krumhansl"
    key_edma: str = "key_edma"
    discogs_embeddings: str = "discogs_embeddings_mean"
    musicnn_embeddings: str = "musicnn_embeddings_mean"
    danceability: str = "danceability"
    style_genre: str = "style_genre"
    # Only available after loading the data
    track: str = "track"
    genre: str = "genre"
    style: str = "style"
    instrumental: str = "instrumental"


def get_column_mapping():
    return {
        "extracted_time_utc": "extracted_time_utc",
        "filepath": "filepath",
        "duration": "duration",
        "sample_rate": "sample_rate",
        "num_channels": "num_channels",
        "md5": "md5",
        "bit_rate": "bit_rate",
        "codec": "codec",
        "tempocnn_bpm": "bpm",
        "key_temperley_predict": "key_temperley",
        "key_temperley_probability": "key_temperley_probability",
        "key_krumhansl_predict": "key_krumhansl",
        "key_krumhansl_probability": "key_krumhansl_probability",
        "key_edma_predict": "key_edma",
        "key_edma_probability": "key_edma_probability",
        "loudness_ebur128_integrated_lufs": "loudness_ebur128_integrated_lufs",
        "loudness_ebur128_range_lu": "loudness_ebur128_range_lu",
        "emomusic_valence": "valence",
        "emomusic_arousal": "arousal",
        "musicnn_embeddings_mean": "musicnn_embeddings_mean",
        "style_genre_discogs400": "style_genre",
        "style_genre_discogs400_probability": "style_genre_discogs400_probability",
        "vggish_voice": "vggish_voice",
        "vggish_instrumental": "vggish_instrumental",
        "effnet_discogs_danceable": "danceability",
        "effnet_discogs_not_danceable": "effnet_discogs_not_danceable",
        "discogs_embeddings_mean": "discogs_embeddings_mean",
        "track": "track",
        "genre": "genre",
        "style": "style",
        "instrumental": "instrumental",
    }


def rename_columns(df, column_mapping):
    return df.rename(columns=column_mapping)


# Create an instance to use throughout your code
columns = Columns()

st.set_page_config(layout="wide")


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
    df["instrumental"] = df["vggish_instrumental"].apply(
        lambda x: "instrumental" if x > 0.5 else "voice"
    )
    return df


df = load_data("../data/processed/20250212032022_audio_features.npy")
df = rename_columns(df, get_column_mapping())
columns = Columns()

# Sidebar filters
st.sidebar.header("Track Filters")

# Music Style Filter
all_styles = df[columns.style_genre].unique()
selected_styles = st.sidebar.multiselect(
    "Select music styles", options=all_styles, default=[]
)

# Tempo Filter
min_bpm, max_bpm = st.sidebar.slider(
    "Tempo Range (BPM)",
    min_value=int(df[columns.bpm].min()),
    max_value=int(np.ceil(df[columns.bpm].max())),
    value=(int(df[columns.bpm].min()), int(np.ceil(df[columns.bpm].max()))),
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
    df_filtered = df_filtered[df_filtered[columns.style_genre].isin(selected_styles)]

# Tempo filter
df_filtered = df_filtered[
    (df_filtered[columns.bpm] >= min_bpm) & (df_filtered[columns.bpm] <= max_bpm)
]

# Arousal & Valence filter
df_filtered = df_filtered[
    (df_filtered[columns.arousal].between(*arousal_range))
    & (df_filtered[columns.valence].between(*valence_range))
]

# Key filter
if key_filter:
    target_key = f"{selected_key} {selected_mode}"
    key_col = f"key_{key_method}"
    df_filtered = df_filtered[df_filtered[key_col] == target_key]

# Danceability sorting
if sort_by_danceability:
    df_filtered = df_filtered.sort_values(columns.danceability, ascending=False)

# Show filtered tracks
st.header("Select Reference Track")
if not df_filtered.empty:
    st.write(f"Found {len(df_filtered)} tracks")
    st.dataframe(
        df_filtered[
            [
                columns.track,
                # columns.duration,
                columns.bpm,
                columns.style_genre,
                columns.style,
                columns.genre,
                columns.arousal,
                columns.valence,
                columns.instrumental,
                columns.danceability,
                columns.key_temperley,
                columns.key_krumhansl,
                columns.key_edma,
            ]
        ].head(100)
    )

    # Track selection
    tracks = df_filtered[columns.track].unique()
    selected_track = st.selectbox("Choose a reference track:", tracks)
else:
    st.error("No tracks match the current filters!")
    st.stop()


# Compute similarities
@st.cache_data
def compute_similarities(_df, query_track):
    # Get query embeddings
    query_idx = _df[_df[columns.track] == query_track].index[0]
    discogs_query = _df.loc[query_idx, columns.discogs_embeddings]
    musicnn_query = _df.loc[query_idx, columns.musicnn_embeddings]

    # Compute similarities using custom function
    _df = _df.copy()
    _df["discogs_similarity"] = _df[columns.discogs_embeddings].apply(
        lambda x: cosine_similarity(x, discogs_query)
    )
    _df["musicnn_similarity"] = _df[columns.musicnn_embeddings].apply(
        lambda x: cosine_similarity(x, musicnn_query)
    )

    return _df


# Compute similarities
similarity_df = compute_similarities(df, selected_track)


# Create separate result dataframes
def prepare_results(df, similarity_col, query_track):
    return (
        df[df[columns.track] != query_track]
        .sort_values(similarity_col, ascending=False)
        .head(10)[
            [
                columns.track,
                columns.duration,
                columns.bpm,
                columns.style_genre,
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
