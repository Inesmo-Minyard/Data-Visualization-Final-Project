from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Beyond Popularity: Anime Success",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CONSTANTS AND COLOURS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CVD_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

HIGHLIGHT_COLOR = "#0072B2"
CONTEXT_COLOR = "#B8B8B8"

ATMOSPHERE_COLORS = {
    "Cozy / Quiet": "#009E73",
    "Melancholic / Reflective": "#0072B2",
    "Dark / Tense": "#D55E00",
    "Romantic / Emotional": "#CC79A7",
    "Energetic / Adventurous": "#E69F00",
    "Fantasy / Escapist": "#56B4E9",
    "Light / Comedic": "#F0E442",
    "Other": "#B8B8B8",
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def apply_chart_style(
    fig,
    title,
    subtitle=None,
    height=550,
):
    """
    Apply a consistent white-background editorial style
    to a Plotly figure.
    """

    full_title = title

    if subtitle:
        full_title = f"{title}<br><sup>{subtitle}</sup>"

    fig.update_layout(
        title={
            "text": full_title,
            "x": 0,
            "xanchor": "left",
        },
        height=height,
        font={
            "family": "Arial",
            "size": 14,
            "color": "#222222",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={
            "l": 70,
            "r": 40,
            "t": 105,
            "b": 70,
        },
        hoverlabel={
            "bgcolor": "white",
            "font_size": 13,
        },
        legend_title_text="",
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#D9D9D9",
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#D9D9D9",
    )

    return fig


def safe_median(dataframe, column):
    """
    Return a numeric median or NaN if the column is unavailable.
    """

    if column not in dataframe.columns:
        return np.nan

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    return values.median()


def format_integer(value):
    """
    Format a number with comma separators.
    """

    if pd.isna(value):
        return "N/A"

    return f"{value:,.0f}"


def format_decimal(value, digits=2):
    """
    Format a decimal number.
    """

    if pd.isna(value):
        return "N/A"

    return f"{value:.{digits}f}"


def read_optional_csv(file_name):
    """
    Read an optional CSV file.
    Return an empty DataFrame if the file does not exist.
    """

    file_path = BASE_DIR / file_name

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


def parse_pipe_list(value):
    """
    Convert a pipe-separated text field into a Python list.
    """

    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


# ============================================================
# 4. DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    anime_path = BASE_DIR / "anime_processed.csv"

    if not anime_path.exists():
        raise FileNotFoundError(
            "anime_processed.csv was not found in the same "
            "folder as app.py."
        )

    anime = pd.read_csv(anime_path)

    atmosphere = read_optional_csv(
        "atmosphere_overview.csv"
    )

    hidden_gems = read_optional_csv(
        "hidden_gems.csv"
    )

    studio_leaders = read_optional_csv(
        "studio_leaders.csv"
    )

    genre_theme = read_optional_csv(
        "genre_theme_leaders.csv"
    )

    findings = read_optional_csv(
        "project_findings.csv"
    )

    return (
        anime,
        atmosphere,
        hidden_gems,
        studio_leaders,
        genre_theme,
        findings,
    )


try:
    (
        anime_df,
        atmosphere_df,
        hidden_gems_df,
        studio_leaders_df,
        genre_theme_df,
        findings_df,
    ) = load_data()

except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Place anime_processed.csv in the same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error(
        "The dashboard could not load the data."
    )
    st.exception(error)
    st.stop()


# ============================================================
# 5. DATA-TYPE PREPARATION
# ============================================================

numeric_columns = [
    "score",
    "scored_by",
    "members",
    "favorites",
    "favorite_rate_pct",
    "release_year",
    "episodes",
    "log_members",
    "atmosphere_match_score",
    "genre_count",
    "theme_count",
]

for column in numeric_columns:
    if column in anime_df.columns:
        anime_df[column] = pd.to_numeric(
            anime_df[column],
            errors="coerce",
        )

if (
    "log_members" not in anime_df.columns
    and "members" in anime_df.columns
):
    anime_df["log_members"] = np.where(
        anime_df["members"] > 0,
        np.log10(anime_df["members"]),
        np.nan,
    )

if (
    "favorite_rate_pct" not in anime_df.columns
    and {"favorites", "members"}.issubset(anime_df.columns)
):
    anime_df["favorite_rate_pct"] = np.where(
        anime_df["members"] > 0,
        anime_df["favorites"]
        / anime_df["members"]
        * 100,
        np.nan,
    )

if "atmosphere" not in anime_df.columns:
    anime_df["atmosphere"] = "Other"

if "type" not in anime_df.columns:
    anime_df["type"] = "Unknown"

if "release_year" not in anime_df.columns:
    anime_df["release_year"] = np.nan


# ============================================================
# 6. HEADER
# ============================================================

st.title(
    "Beyond Popularity: What Makes Anime Successful?"
)

st.markdown(
    """
    This interactive dashboard explores how anime format,
    atmosphere, historical period, studio, and content
    characteristics are associated with audience reach,
    user ratings, and viewer loyalty.
    """
)


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Atmosphere Analysis",
        "Hidden Gems",
        "Studios",
        "Genre–Theme Combinations",
        "Methodology",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Global filters")


# ============================================================
# 8. GLOBAL FILTERS
# ============================================================

available_atmospheres = sorted(
    anime_df["atmosphere"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

default_atmospheres = [
    atmosphere
    for atmosphere in available_atmospheres
    if atmosphere != "Other"
]

if not default_atmospheres:
    default_atmospheres = available_atmospheres

selected_atmospheres = st.sidebar.multiselect(
    "Atmosphere",
    options=available_atmospheres,
    default=default_atmospheres,
)

available_types = sorted(
    anime_df["type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_types = st.sidebar.multiselect(
    "Anime format",
    options=available_types,
    default=available_types,
)

valid_years = anime_df[
    "release_year"
].dropna()

if valid_years.empty:
    minimum_year = 1900
    maximum_year = 2025
else:
    minimum_year = int(valid_years.min())
    maximum_year = int(
        min(valid_years.max(), 2025)
    )

default_start_year = max(
    1970,
    minimum_year,
)

if default_start_year > maximum_year:
    default_start_year = minimum_year

selected_year_range = st.sidebar.slider(
    "Release year",
    min_value=minimum_year,
    max_value=maximum_year,
    value=(
        default_start_year,
        maximum_year,
    ),
)

minimum_ratings = st.sidebar.select_slider(
    "Minimum number of user ratings",
    options=[
        0,
        100,
        500,
        1000,
        5000,
        10000,
    ],
    value=1000,
)


# ============================================================
# 9. APPLY GLOBAL FILTERS
# ============================================================

filter_mask = pd.Series(
    True,
    index=anime_df.index,
)

if selected_atmospheres:
    filter_mask &= anime_df[
        "atmosphere"
    ].isin(selected_atmospheres)
else:
    filter_mask &= False

if selected_types:
    filter_mask &= anime_df[
        "type"
    ].isin(selected_types)
else:
    filter_mask &= False

filter_mask &= anime_df[
    "release_year"
].between(
    selected_year_range[0],
    selected_year_range[1],
)

if "scored_by" in anime_df.columns:
    filter_mask &= (
        anime_df["scored_by"].fillna(0)
        >= minimum_ratings
    )

filtered_df = anime_df[
    filter_mask
].copy()

if filtered_df.empty:
    st.warning(
        "No anime match the selected filters. "
        "Please broaden the filter selection."
    )
    st.stop()


# ============================================================
# 10. OVERVIEW PAGE
# ============================================================

if page == "Overview":
    st.header("Dataset overview")

    st.markdown(
        """
        The overview presents the scale and coverage of the
        filtered dataset. All charts respond to the global
        sidebar filters.
        """
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Anime titles",
        f"{filtered_df['mal_id'].nunique():,}"
        if "mal_id" in filtered_df.columns
        else f"{len(filtered_df):,}",
    )

    metric_2.metric(
        "Rating median score",
        format_decimal(
            safe_median(
                filtered_df,
                "score",
            )
        ),
    )

    metric_3.metric(
        "Median audience size",
        format_integer(
            safe_median(
                filtered_df,
                "members",
            )
        ),
    )

    metric_4.metric(
        "Atmospheres",
        f"{filtered_df['atmosphere'].nunique():,}",
    )

    st.divider()

    yearly_releases = (
        filtered_df
        .dropna(
            subset=["release_year"]
        )
        .groupby("release_year")
        .size()
        .reset_index(
            name="anime_count"
        )
    )

    fig_years = px.line(
        yearly_releases,
        x="release_year",
        y="anime_count",
        color_discrete_sequence=[
            HIGHLIGHT_COLOR
        ],
        labels={
            "release_year": "Release year",
            "anime_count": "Anime titles",
        },
    )

    fig_years.update_traces(
        line_width=3,
        hovertemplate=(
            "<b>%{x:.0f}</b><br>"
            "Anime releases: %{y:,.0f}"
            "<extra></extra>"
        ),
    )

    fig_years = apply_chart_style(
        fig_years,
        title="Anime releases expanded strongly over time",
        subtitle=(
            "Annual release count under the selected filters"
        ),
        height=520,
    )

    st.plotly_chart(
        fig_years,
        use_container_width=True,
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        type_counts = (
            filtered_df["type"]
            .value_counts()
            .rename_axis("type")
            .reset_index(
                name="anime_count"
            )
        )

        fig_types = px.bar(
            type_counts,
            x="anime_count",
            y="type",
            orientation="h",
            color_discrete_sequence=[
                HIGHLIGHT_COLOR
            ],
            labels={
                "anime_count": "Anime titles",
                "type": "",
            },
        )

        fig_types.update_yaxes(
            categoryorder="total ascending"
        )

        fig_types.update_traces(
            texttemplate="%{x:,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Anime titles: %{x:,.0f}"
                "<extra></extra>"
            ),
        )

        fig_types = apply_chart_style(
            fig_types,
            title="Release formats",
            height=500,
        )

        st.plotly_chart(
            fig_types,
            use_container_width=True,
        )

    with chart_right:
        atmosphere_counts = (
            filtered_df["atmosphere"]
            .value_counts()
            .rename_axis("atmosphere")
            .reset_index(
                name="anime_count"
            )
        )

        fig_atmospheres = px.bar(
            atmosphere_counts,
            x="anime_count",
            y="atmosphere",
            orientation="h",
            color="atmosphere",
            color_discrete_map=ATMOSPHERE_COLORS,
            labels={
                "anime_count": "Anime titles",
                "atmosphere": "",
            },
        )

        fig_atmospheres.update_yaxes(
            categoryorder="total ascending"
        )

        fig_atmospheres.update_traces(
            texttemplate="%{x:,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Anime titles: %{x:,.0f}"
                "<extra></extra>"
            ),
        )

        fig_atmospheres = apply_chart_style(
            fig_atmospheres,
            title="Atmosphere distribution",
            height=500,
        )

        fig_atmospheres.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_atmospheres,
            use_container_width=True,
        )

    if not findings_df.empty:
        st.subheader("Research findings")

        st.dataframe(
            findings_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 11. ATMOSPHERE ANALYSIS PAGE
# ============================================================

elif page == "Atmosphere Analysis":
    st.header("Atmosphere analysis")

    st.markdown(
        """
        This section compares anime atmospheres across
        audience size, ratings, loyalty, and historical
        production patterns.
        """
    )

    atmosphere_analysis_df = filtered_df[
        filtered_df["atmosphere"] != "Other"
    ].copy()

    if atmosphere_analysis_df.empty:
        st.warning(
            "No classified atmosphere data remain "
            "after filtering."
        )
        st.stop()

    atmosphere_summary = (
        atmosphere_analysis_df
        .groupby("atmosphere")
        .agg(
            anime_count=(
                "mal_id",
                "nunique",
            )
            if "mal_id" in atmosphere_analysis_df.columns
            else (
                "title",
                "count",
            ),
            median_members=(
                "members",
                "median",
            ),
            median_score=(
                "score",
                "median",
            ),
            median_loyalty=(
                "favorite_rate_pct",
                "median",
            ),
        )
        .reset_index()
    )

    atmosphere_summary = atmosphere_summary.dropna(
        subset=[
            "median_members",
            "median_score",
        ]
    )

    if atmosphere_summary.empty:
        st.warning(
            "There is not enough atmosphere-level data "
            "for the selected filters."
        )
        st.stop()

    atmosphere_summary[
        "log_median_members"
    ] = np.log10(
        atmosphere_summary[
            "median_members"
        ].clip(lower=1)
    )

    fig_atmosphere_success = px.scatter(
        atmosphere_summary,
        x="log_median_members",
        y="median_score",
        size="anime_count",
        color="atmosphere",
        text="atmosphere",
        size_max=55,
        color_discrete_map=ATMOSPHERE_COLORS,
        hover_name="atmosphere",
        hover_data={
            "log_median_members": False,
            "median_members": ":,.0f",
            "median_score": ":.2f",
            "median_loyalty": ":.2f",
            "anime_count": ":,",
        },
        labels={
            "log_median_members": (
                "Typical audience size, log10(median members)"
            ),
            "median_score": "Median user score",
            "median_loyalty": "Median loyalty, %",
            "anime_count": "Anime titles",
            "atmosphere": "Atmosphere",
        },
    )

    fig_atmosphere_success.update_traces(
        marker={
            "opacity": 0.82,
            "line": {
                "width": 1.5,
                "color": "white",
            },
        },
        textposition="top center",
    )

    fig_atmosphere_success = apply_chart_style(
        fig_atmosphere_success,
        title=(
            "Large audiences do not always coincide "
            "with the strongest ratings"
        ),
        subtitle=(
            "Bubble size represents the number of titles"
        ),
        height=650,
    )

    fig_atmosphere_success.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_atmosphere_success,
        use_container_width=True,
    )

    left_chart, right_chart = st.columns(2)

    with left_chart:
        rating_order = (
            atmosphere_summary
            .sort_values(
                "median_score",
                ascending=True,
            )
        )

        fig_rating = px.bar(
            rating_order,
            x="median_score",
            y="atmosphere",
            orientation="h",
            color="atmosphere",
            color_discrete_map=ATMOSPHERE_COLORS,
            text="median_score",
            labels={
                "median_score": "Median user score",
                "atmosphere": "",
            },
        )

        fig_rating.update_traces(
            texttemplate="%{x:.2f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median score: %{x:.2f}"
                "<extra></extra>"
            ),
        )

        fig_rating = apply_chart_style(
            fig_rating,
            title="Typical rating by atmosphere",
            height=520,
        )

        fig_rating.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_rating,
            use_container_width=True,
        )

    with right_chart:
        loyalty_order = (
            atmosphere_summary
            .dropna(
                subset=["median_loyalty"]
            )
            .sort_values(
                "median_loyalty",
                ascending=True,
            )
        )

        fig_loyalty = px.bar(
            loyalty_order,
            x="median_loyalty",
            y="atmosphere",
            orientation="h",
            color="atmosphere",
            color_discrete_map=ATMOSPHERE_COLORS,
            text="median_loyalty",
            labels={
                "median_loyalty": (
                    "Median favorites-to-members rate, %"
                ),
                "atmosphere": "",
            },
        )

        fig_loyalty.update_traces(
            texttemplate="%{x:.2f}%",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median loyalty: %{x:.2f}%"
                "<extra></extra>"
            ),
        )

        fig_loyalty = apply_chart_style(
            fig_loyalty,
            title="Audience loyalty by atmosphere",
            height=520,
        )

        fig_loyalty.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_loyalty,
            use_container_width=True,
        )

    st.subheader("Atmosphere share across decades")

    decade_df = atmosphere_analysis_df[
        atmosphere_analysis_df[
            "release_year"
        ].between(
            1970,
            2019,
        )
    ].copy()

    if not decade_df.empty:
        decade_df["decade"] = (
            (
                decade_df[
                    "release_year"
                ].astype(int)
                // 10
            )
            * 10
        ).astype(str) + "s"

        decade_summary = (
            decade_df
            .groupby(
                [
                    "decade",
                    "atmosphere",
                ]
            )
            .size()
            .reset_index(
                name="anime_count"
            )
        )

        decade_summary[
            "decade_total"
        ] = (
            decade_summary
            .groupby("decade")[
                "anime_count"
            ]
            .transform("sum")
        )

        decade_summary[
            "share_pct"
        ] = (
            decade_summary[
                "anime_count"
            ]
            / decade_summary[
                "decade_total"
            ]
            * 100
        )

        decade_order = [
            "1970s",
            "1980s",
            "1990s",
            "2000s",
            "2010s",
        ]

        fig_decades = px.line(
            decade_summary,
            x="decade",
            y="share_pct",
            color="atmosphere",
            markers=True,
            category_orders={
                "decade": decade_order,
            },
            color_discrete_map=ATMOSPHERE_COLORS,
            hover_data={
                "anime_count": ":,",
                "decade_total": ":,",
                "share_pct": ":.1f",
            },
            labels={
                "decade": "Release decade",
                "share_pct": "Share of releases, %",
                "atmosphere": "Atmosphere",
                "anime_count": "Anime titles",
                "decade_total": "All titles in decade",
            },
        )

        fig_decades.update_traces(
            line={
                "width": 3,
            },
            marker={
                "size": 9,
                "line": {
                    "width": 1,
                    "color": "white",
                },
            },
        )

        fig_decades = apply_chart_style(
            fig_decades,
            title=(
                "The atmospheric composition of anime "
                "changed across decades"
            ),
            subtitle=(
                "Each completed decade totals 100%"
            ),
            height=650,
        )

        fig_decades.update_yaxes(
            ticksuffix="%",
            rangemode="tozero",
        )

        st.plotly_chart(
            fig_decades,
            use_container_width=True,
        )


# ============================================================
# 12. HIDDEN GEMS PAGE
# ============================================================

elif page == "Hidden Gems":
    st.header("Hidden gems")

    st.markdown(
        """
        Hidden gems combine strong user ratings with
        comparatively limited audience reach.
        """
    )

    candidate_df = filtered_df.dropna(
        subset=[
            "score",
            "members",
        ]
    ).copy()

    if candidate_df.empty:
        st.warning(
            "No titles are available for hidden-gem analysis."
        )
        st.stop()

    score_percentile = st.slider(
        "Minimum score percentile",
        min_value=50,
        max_value=95,
        value=75,
        step=5,
    )

    audience_percentile_limit = st.slider(
        "Maximum audience percentile",
        min_value=5,
        max_value=75,
        value=50,
        step=5,
    )

    candidate_df[
        "score_percentile"
    ] = (
        candidate_df["score"]
        .rank(
            pct=True,
            method="average",
        )
        * 100
    )

    candidate_df[
        "audience_percentile"
    ] = (
        candidate_df["members"]
        .rank(
            pct=True,
            method="average",
        )
        * 100
    )

    candidate_df[
        "obscurity_percentile"
    ] = (
        100
        - candidate_df[
            "audience_percentile"
        ]
    )

    candidate_df[
        "hidden_gem_score"
    ] = (
        candidate_df[
            "score_percentile"
        ]
        + candidate_df[
            "obscurity_percentile"
        ]
    ) / 2

    hidden_filtered = candidate_df[
        (
            candidate_df[
                "score_percentile"
            ]
            >= score_percentile
        )
        & (
            candidate_df[
                "audience_percentile"
            ]
            <= audience_percentile_limit
        )
    ].copy()

    hidden_filtered = hidden_filtered.sort_values(
        "hidden_gem_score",
        ascending=False,
    )

    metric_left, metric_middle, metric_right = st.columns(3)

    metric_left.metric(
        "Hidden gems found",
        f"{len(hidden_filtered):,}",
    )

    metric_middle.metric(
        "Median hidden-gem score",
        format_decimal(
            safe_median(
                hidden_filtered,
                "hidden_gem_score",
            ),
            1,
        ),
    )

    metric_right.metric(
        "Median audience",
        format_integer(
            safe_median(
                hidden_filtered,
                "members",
            )
        ),
    )

    if hidden_filtered.empty:
        st.warning(
            "No titles meet the current hidden-gem criteria."
        )
        st.stop()

    top_n = st.slider(
        "Number of titles to display",
        min_value=5,
        max_value=min(
            30,
            len(hidden_filtered),
        ),
        value=min(
            15,
            len(hidden_filtered),
        ),
    )

    top_hidden = (
        hidden_filtered
        .head(top_n)
        .sort_values(
            "hidden_gem_score",
            ascending=True,
        )
    )

    fig_hidden = px.bar(
        top_hidden,
        x="hidden_gem_score",
        y="title",
        orientation="h",
        color="atmosphere",
        color_discrete_map=ATMOSPHERE_COLORS,
        text="hidden_gem_score",
        hover_data={
            "score": ":.2f",
            "members": ":,",
            "scored_by": ":,",
            "score_percentile": ":.1f",
            "obscurity_percentile": ":.1f",
            "release_year": ":.0f",
        },
        labels={
            "hidden_gem_score": "Hidden gem score",
            "title": "",
            "atmosphere": "Atmosphere",
            "score": "User score",
            "members": "Members",
            "scored_by": "Users who rated",
            "release_year": "Release year",
        },
    )

    fig_hidden.update_traces(
        texttemplate="%{x:.1f}",
        textposition="outside",
        cliponaxis=False,
    )

    fig_hidden = apply_chart_style(
        fig_hidden,
        title=(
            "Highly rated titles can remain "
            "comparatively undiscovered"
        ),
        subtitle=(
            "Hidden-gem score combines rating strength "
            "and audience obscurity"
        ),
        height=max(
            550,
            top_n * 35 + 180,
        ),
    )

    st.plotly_chart(
        fig_hidden,
        use_container_width=True,
    )

    display_columns = [
        column
        for column in [
            "title",
            "title_english",
            "atmosphere",
            "type",
            "release_year",
            "score",
            "members",
            "scored_by",
            "hidden_gem_score",
        ]
        if column in hidden_filtered.columns
    ]

    st.subheader("Hidden-gem table")

    st.dataframe(
        hidden_filtered[
            display_columns
        ].head(100),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 13. STUDIOS PAGE
# ============================================================

elif page == "Studios":
    st.header("Studio performance")

    st.markdown(
        """
        This section compares studio performance across
        atmosphere categories using rating strength,
        production volume, and audience size.
        """
    )

    if not studio_leaders_df.empty:
        studio_plot_df = studio_leaders_df.copy()

        numeric_studio_columns = [
            "anime_count",
            "median_score",
            "median_members",
            "studio_performance_index",
        ]

        for column in numeric_studio_columns:
            if column in studio_plot_df.columns:
                studio_plot_df[column] = pd.to_numeric(
                    studio_plot_df[column],
                    errors="coerce",
                )

        if "atmosphere" in studio_plot_df.columns:
            studio_plot_df = studio_plot_df[
                studio_plot_df[
                    "atmosphere"
                ].isin(selected_atmospheres)
            ]

        required_studio_columns = {
            "studio",
            "atmosphere",
            "anime_count",
            "median_score",
            "median_members",
        }

        if (
            not studio_plot_df.empty
            and required_studio_columns.issubset(
                studio_plot_df.columns
            )
        ):
            fig_studios = px.scatter(
                studio_plot_df,
                x="anime_count",
                y="median_score",
                size="median_members",
                color="atmosphere",
                text="studio",
                size_max=60,
                color_discrete_map=ATMOSPHERE_COLORS,
                hover_name="studio",
                hover_data={
                    "anime_count": ":,",
                    "median_score": ":.2f",
                    "median_members": ":,.0f",
                    "studio_performance_index": ":.1f"
                    if "studio_performance_index"
                    in studio_plot_df.columns
                    else False,
                },
                labels={
                    "anime_count": (
                        "Number of titles analyzed for each studio"
                    ),
                    "median_score": (
                        "Median user score"
                    ),
                    "median_members": (
                        "Median members"
                    ),
                    "atmosphere": "Atmosphere",
                },
            )

            fig_studios.update_traces(
                marker={
                    "opacity": 0.82,
                    "line": {
                        "width": 1.5,
                        "color": "white",
                    },
                },
                textposition="top center",
            )

            fig_studios = apply_chart_style(
                fig_studios,
                title=(
                    "Studio leadership differs "
                    "across anime atmospheres"
                ),
                subtitle=(
                    "Bubble size represents median audience size"
                ),
                height=680,
            )

            fig_studios.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig_studios,
                use_container_width=True,
            )

            st.dataframe(
                studio_plot_df.sort_values(
                    "median_score",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.warning(
                "studio_leaders.csv does not contain "
                "the expected columns."
            )

    else:
        st.info(
            "studio_leaders.csv was not found. "
            "Export it from the notebook and place it "
            "next to app.py."
        )


# ============================================================
# 14. GENRE–THEME PAGE
# ============================================================

elif page == "Genre–Theme Combinations":
    st.header("Genre–theme combinations")

    st.markdown(
        """
        This section identifies combinations that balance
        high user ratings, broad audience reach, and
        sufficient production volume.
        """
    )

    if not genre_theme_df.empty:
        genre_plot_df = genre_theme_df.copy()

        numeric_genre_columns = [
            "anime_count",
            "median_score",
            "median_members",
            "success_index",
            "score_percentile",
            "audience_percentile",
        ]

        for column in numeric_genre_columns:
            if column in genre_plot_df.columns:
                genre_plot_df[column] = pd.to_numeric(
                    genre_plot_df[column],
                    errors="coerce",
                )

        required_genre_columns = {
            "genre_theme",
            "anime_count",
            "median_score",
            "median_members",
        }

        if required_genre_columns.issubset(
            genre_plot_df.columns
        ):
            genre_plot_df = genre_plot_df.dropna(
                subset=[
                    "median_score",
                    "median_members",
                ]
            )

            genre_plot_df[
                "log_median_members"
            ] = np.log10(
                genre_plot_df[
                    "median_members"
                ].clip(lower=1)
            )

            if "success_index" in genre_plot_df.columns:
                genre_plot_df = genre_plot_df.sort_values(
                    "success_index",
                    ascending=False,
                )

            top_genre_count = st.slider(
                "Number of combinations",
                min_value=5,
                max_value=min(
                    30,
                    len(genre_plot_df),
                ),
                value=min(
                    15,
                    len(genre_plot_df),
                ),
            )

            genre_display_df = genre_plot_df.head(
                top_genre_count
            )

            fig_genre_theme = px.scatter(
                genre_display_df,
                x="log_median_members",
                y="median_score",
                size="anime_count",
                color="genre_theme",
                text="genre_theme",
                size_max=55,
                color_discrete_sequence=CVD_COLORS,
                hover_name="genre_theme",
                hover_data={
                    "log_median_members": False,
                    "median_members": ":,.0f",
                    "median_score": ":.2f",
                    "anime_count": ":,",
                    "success_index": ":.1f"
                    if "success_index"
                    in genre_display_df.columns
                    else False,
                },
                labels={
                    "log_median_members": (
                        "Typical audience size, "
                        "log10(median members)"
                    ),
                    "median_score": "Median user score",
                    "anime_count": "Anime titles",
                    "genre_theme": (
                        "Genre–theme combination"
                    ),
                },
            )

            fig_genre_theme.update_traces(
                marker={
                    "opacity": 0.80,
                    "line": {
                        "width": 1.5,
                        "color": "white",
                    },
                },
                textposition="top center",
                textfont={
                    "size": 10,
                },
            )

            fig_genre_theme = apply_chart_style(
                fig_genre_theme,
                title=(
                    "Only some genre–theme combinations "
                    "combine broad reach with strong ratings"
                ),
                subtitle=(
                    "Bubble size represents the number of titles"
                ),
                height=720,
            )

            fig_genre_theme.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig_genre_theme,
                use_container_width=True,
            )

            st.dataframe(
                genre_display_df,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.warning(
                "genre_theme_leaders.csv does not contain "
                "the expected columns."
            )

    else:
        st.info(
            "genre_theme_leaders.csv was not found. "
            "Export it from the notebook and place it "
            "next to app.py."
        )


# ============================================================
# 15. METHODOLOGY PAGE
# ============================================================

elif page == "Methodology":
    st.header("Methodology")

    st.markdown(
        """
        ### Main research question

        **How are anime characteristics associated with
        popularity, user ratings, and audience loyalty,
        and how have these relationships changed over time?**

        ### Core measures

        - **Audience reach:** number of MyAnimeList members.
        - **User evaluation:** average MyAnimeList score.
        - **Audience loyalty:** favorites divided by members.
        - **Relative popularity:** audience percentile within
          the same release period.
        - **Atmosphere:** rule-based classification derived
          from genre and theme metadata.

        ### Reliability filters

        Several analyses require at least 1,000 user ratings.
        This reduces the influence of unstable scores based
        on very small samples.

        ### Use of medians

        Audience size is strongly right-skewed. Medians are
        used frequently to reduce the influence of a small
        number of blockbuster titles.

        ### Custom analytical indices

        The hidden-gem, studio-performance, and genre–theme
        success indices are project-specific measures.
        They combine percentile-based components with equal
        weighting and should be interpreted as ranking aids,
        not universal measures of quality.

        ### Main limitations

        - MyAnimeList users may not represent the entire
          global anime audience.
        - Members do not necessarily represent completed views.
        - Ratings and favourites are voluntary user actions.
        - Older titles have had more time to accumulate members.
        - Atmosphere classification involves subjective rules.
        - Associations do not demonstrate causality.
        """
    )

    st.subheader("Filtered dataset preview")

    preview_columns = [
        column
        for column in [
            "title",
            "title_english",
            "type",
            "release_year",
            "atmosphere",
            "score",
            "scored_by",
            "members",
            "favorites",
            "favorite_rate_pct",
        ]
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[
            preview_columns
        ].head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Current filter summary")

    st.write(
        {
            "Selected atmospheres": selected_atmospheres,
            "Selected formats": selected_types,
            "Release years": selected_year_range,
            "Minimum user ratings": minimum_ratings,
            "Matching titles": len(filtered_df),
        }
    )


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.caption(
    "Data source: MyAnimeList-derived anime dataset. "
    "Dashboard created with Python, Pandas, Plotly, "
    "and Streamlit."
)