from constants import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from  matplotlib.colors import LinearSegmentedColormap

CMAP_RG=LinearSegmentedColormap.from_list('rg',["r", "y", "g"], N=256) 
CMAP_RG_INVERTED=LinearSegmentedColormap.from_list('rg',["g", "y", "r"], N=256) 

EVALUATIONS_PATH = "evaluations"
MODELS_PERFORMANCES_PATH = "models_performances"
MAE = "MAE"
PERCENTAGE = "percentage"
GRAPHS_PATH = "graphs"
os.makedirs(GRAPHS_PATH, exist_ok=True)
MAE_PATH =f"{GRAPHS_PATH}/{MAE}/"
os.makedirs(MAE_PATH, exist_ok=True)
PERCENTAGE_PATH = f"{GRAPHS_PATH}/{PERCENTAGE}/" 
os.makedirs(PERCENTAGE_PATH, exist_ok=True)
for s in SCENARIOS:
    os.makedirs(f"{GRAPHS_PATH}/{MODELS_PERFORMANCES_PATH}/{s}", exist_ok=True)

#Generate back translation dataframe and heatmap
def back_translation():
    df = pd.read_csv(f"data/back_translation.csv", sep=";", index_col="model")

    # Remove avg_score if present
    if "avg_score" in df.columns:
        df = df.drop(columns=["avg_score"])

    # Convert to numeric (in case some values are read as strings)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    plt.figure(figsize=(16, 6))
    sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap=CMAP_RG,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        annot_kws={"fontsize":8},
        fmt=".1f"
    )

    plt.xlabel("Language")
    plt.ylabel("Model")
    #plt.title("Models Performance in Back Translation Test")
    plt.xticks(rotation=40, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_PATH}/back_translation.png", bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {f"{GRAPHS_PATH}/back_translation.png"}")

def generate_heatmap(
    df,
    xlabel,
    ylabel,
    savefig,
    annot=False,
    cmap=CMAP_RG,
    transpose = False
):
    # Transpose
    if transpose:
        df = df.T
        xlabel, ylabel = ylabel, xlabel
    
    df = df.fillna(0)

    n_rows, n_cols = df.shape

    # Smaller cells
    cell_width = 0.25
    cell_height = 0.25

    fig_width = max(8, n_cols * cell_width)
    fig_height = max(5, n_rows * cell_height)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap=cmap,
        linewidths=0.2,
        linecolor="white",
        annot=annot,
        annot_kws={"fontsize": 7},
        fmt=".1f",

        # Thinner color bar
        cbar_kws={
            "shrink": 0.8,
            "aspect": 40,      # larger → thinner
            "fraction": 0.03,  # width of colorbar
            "pad": 0.02
        },

        ax=ax
    )

    if transpose:
        rotation = 30
        fontsize = 14
    rotation = 50
    fontsize = 12
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    
    # Bigger labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=rotation,
        fontsize=fontsize,
        ha="right",
    )
    
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=30,
        fontsize=fontsize,
        ha="right",      # horizontal alignment
        va="center",     # vertical alignment
        rotation_mode="anchor"
    )
    
    ax.tick_params(axis='both', length=0)

    plt.tight_layout()

    plt.savefig(
        savefig,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03
    )

    plt.close()

    print(f"Saved: {savefig}")
    
#Weight coherence by validity scores
def model_performances():

    metrics = {
        "fact_coh": "Fact Coherence",
        "fact_val": "Fact Validity",
        "fact_coh_val": "Fact Weight coherence by validity",
        "stance_coh": "Stance Coherence",
        "stance_val": "Stance Validity",
        "stance_coh_val": "Stance Weight coherence by validity",
    }

    for scenario in SCENARIOS:

        for m, title in metrics.items():

            file_path = (
                f"{EVALUATIONS_PATH}/"
                f"{MODELS_PERFORMANCES_PATH}/"
                f"{scenario}/{m}.csv"
            )

            if not os.path.exists(file_path):
                continue

            # IMPORTANT: first column is the model name index
            df_metric = pd.read_csv(file_path, sep=",", index_col=0)

            # Convert only data columns to numeric
            df_metric = df_metric.apply(pd.to_numeric, errors="coerce")
            
            avg_col = f"{title.split(' ')[0]} Average"

            if avg_col in df_metric.columns:
                df_metric = df_metric.drop(columns=[avg_col])
                
            # Keep original model order from CSV
            # Sort only languages if desired
            df_metric = df_metric.sort_index(axis=1)

            xlabel = "Language" if scenario == SCENARIO_LANGUAGE else "Country" if scenario == SCENARIO_COUNTRY else "Lanaguage - Country" 
            
            generate_heatmap(
                df=df_metric,
                xlabel=xlabel,
                ylabel="Model",
                savefig=f"{GRAPHS_PATH}/{MODELS_PERFORMANCES_PATH}/{scenario}/{m}.png",
                #transpose=True,
                annot=True
            )
            print(f"Saved: {f"{GRAPHS_PATH}/{MODELS_PERFORMANCES_PATH}/{scenario}/{m}.png"}")

#Generate the Fact and Stance heatmaps of the MAEs errors of all the models           
def mae_models():
    for test in [FACT, STANCE]:
        csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

        if not os.path.exists(csv_path):
            print(f"Missing file: {csv_path}")
            return

        df_scenario = pd.read_csv(csv_path, sep=";")

        # Iterate over scenarios
        for scenario in SCENARIOS:

            df = df_scenario[df_scenario[SCENARIO] == scenario].copy()

            # Define x-axis labels depending on scenario
            if scenario == SCENARIO_LANGUAGE:
                df["x_label"] = df["languages"].astype(str)

            elif scenario == SCENARIO_COUNTRY:
                df["x_label"] = df["Country"].astype(str)

            elif scenario == SCENARIO_LAN_NAT:
                df["x_label"] = (df["languages"].astype(str)+ " | "+ df["Country"].astype(str))

            # Aggregate mean MAE
            df = (df.groupby(["Model", "x_label"], as_index=False)[f"{test} MAE"].mean())

            # Rename models
            df["Model"] = df["Model"].astype(str).map(MODEL_LABEL)

            ordered_labels = [
                MODEL_LABEL.get(model, model)
                for model in MODEL_LIST
            ]

            df["Model"] = pd.Categorical(
                df["Model"],
                categories=ordered_labels,
                ordered=True,
            )

            df = df.sort_values(["Model", "x_label"])

            # Pivot table
            pivot = df.pivot_table(
                index="Model",
                columns="x_label",
                values=f"{test} MAE",
                observed=False
            )

            plt.figure(figsize=(16, 6))

            sns.heatmap(
                pivot,
                vmin=0,
                vmax=1,
                cmap=CMAP_RG_INVERTED,
                linewidths=0.5,
                linecolor="white",
                fmt=".2f"
            )

            plt.xlabel(
                "Language"
                if scenario == SCENARIO_LANGUAGE
                else "Country"
                if scenario == SCENARIO_COUNTRY
                else "Language | Country"
            )

            plt.ylabel("Model")
            #plt.title(f"{test} MAE Heatmap")
            plt.xticks(rotation=40, ha="right", fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            output_path = (f"{GRAPHS_PATH}/{MAE}/{test}_models.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")
            
#Create the line graphs per each model of the average percentage score accross countries              
# def line_graphs_percentage_plain():
#     csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

#     if not os.path.exists(csv_path):
#         print(f"Missing file: {csv_path}")
#         return

#     df_scenario = pd.read_csv(csv_path, sep=";")

#     # Iterate over scenarios contained in the dataframe
#     for scenario in SCENARIOS:
#         df = df_scenario[df_scenario[SCENARIO] == scenario]

#         if scenario == SCENARIO_LANGUAGE:
#             df["x_label"] = df["languages"]
#         elif scenario == SCENARIO_COUNTRY:
#             df["x_label"] = df["Country"]
#         elif scenario == SCENARIO_LAN_NAT:
#             df["x_label"] = (df["languages"].astype(str) + " | "+ df["Country"].astype(str))

#         df = df.sort_values("x_label")
#         models = df["Model"].unique()

#         for model in models:
#             df_model = (df[df["Model"] == model].copy().reset_index(drop=True))

#             x = range(len(df_model))
#             plt.figure(figsize=(26, 10))

#             # Rainbow Map
#             plt.plot(
#                 x,
#                 df_model[RAINBOW_MAP],
#                 linestyle="--",
#                 linewidth=2,
#                 label=RAINBOW_MAP
#             )
            
#             for test in [FACT, STANCE]:
#                 plt.plot(
#                     x,
#                     df_model[test],
#                     marker="o",
#                     label="Fact Prediction"
#                 )

#                 # Emphasize distance
#                 plt.fill_between(
#                     x,
#                     df_model[test],
#                     df_model[RAINBOW_MAP],
#                     alpha=0.15
#                 )

#             plt.xticks(
#                 x,
#                 df_model["x_label"],
#                 rotation=90
#             )

#             plt.ylabel("Rainbow Meter Score (%)")
#             plt.xlabel("Language / Country")

#             plt.title(f"{model} | {SCENARIO_LABELS[scenario]} | Scores in %")

#             plt.legend()
#             plt.tight_layout()

#             save_path = (f"{GRAPHS_PATH}/percentage/{scenario}/plain_{model}.png")
#             os.makedirs(os.path.dirname(save_path),exist_ok=True)
#             plt.savefig(save_path)
#             print(f"Saved: {save_path}")
#             plt.close()

#Generate the Fact and Stance heatmaps of the MAEs errors of all languages and countries
# def heatmap_language_nat_mae():
#     csv_path = f"{EVALUATIONS_PATH}/lang_country_mae_summary.csv"

#     if not os.path.exists(csv_path):
#         print(f"Missing file: {csv_path}")
#         return

#     df = pd.read_csv(csv_path, sep=";")

#     for test in [FACT, STANCE]:
#         language_col = f"{SCENARIO_LANGUAGE}_{test}_MAE"
#         country_col = f"{SCENARIO_COUNTRY}_{test}_MAE"
#         lang_nat_col = f"{SCENARIO_LAN_NAT}_{test}_MAE"

#         language_rows = (
#             df[df[language_col].notna()][["Key", language_col]]
#             .copy()
#             .rename(columns={
#                 "Key": "Language",
#                 language_col: "Value"
#             })
#         )

#         country_rows = (
#             df[df[country_col].notna()][["Key", country_col]]
#             .copy()
#             .rename(columns={
#                 "Key": "Country",
#                 country_col: "Value"
#             })
#         )

#         lang_country_rows = (
#             df[df[lang_nat_col].notna()][["Key", lang_nat_col]]
#             .copy()
#         )

#         lang_country_rows[["Language", "Country"]] = (
#             lang_country_rows["Key"]
#             .str.split(r"\s*\|\s*", expand=True)
#         )

#         lang_country_rows.rename(columns={
#             lang_nat_col: "Value"
#         }, inplace=True)

#         languages = sorted(
#             set(language_rows["Language"].dropna())
#             | set(lang_country_rows["Language"].dropna())
#         )

#         countries = sorted(
#             set(country_rows["Country"].dropna())
#             | set(lang_country_rows["Country"].dropna())
#         )

#         full_rows =  languages + [SCENARIO_LABELS[SCENARIO_COUNTRY]]
#         full_columns = [SCENARIO_LABELS[SCENARIO_LANGUAGE]] + countries

#         matrix = pd.DataFrame(
#             np.nan,
#             index=full_rows,
#             columns=full_columns,
#         )

#         for _, row in language_rows.iterrows():

#             matrix.loc[
#                 row["Language"],
#                 SCENARIO_LABELS[SCENARIO_LANGUAGE]
#             ] = row["Value"]

#         for _, row in country_rows.iterrows():

#             matrix.loc[
#                 SCENARIO_LABELS[SCENARIO_COUNTRY],
#                 row["Country"]
#             ] = row["Value"]

#         for _, row in lang_country_rows.iterrows():

#             matrix.loc[
#                 row["Language"],
#                 row["Country"]
#             ] = row["Value"]

#         plt.figure(
#             figsize=(
#                 max(14, len(matrix.columns) * 0.45),
#                 max(12, len(matrix.index) * 0.35),
#             )
#         )

#         sns.heatmap(
#             matrix,
#             vmin=0,
#             vmax=1,
#             cmap=CMAP_RG_INVERTED,
#             annot=True,
#             fmt=".2f",
#             linewidths=0.5,
#             linecolor="grey",
#             square=False,
#             cbar_kws={"label": "MAE"},
#         )

#         #plt.title(f"{test} MAE Heatmap")
#         plt.xlabel("Countries", fontsize=15)
#         plt.ylabel("Languages", fontsize=15)
#         plt.xticks(rotation=40, ha='right', fontsize=15)
#         plt.yticks(rotation=0, fontsize=15)
#         plt.tight_layout()
#         output_png = (f"{GRAPHS_PATH}/MAE/language_country_{test}_mae.png")
#         plt.savefig(output_png, dpi=300, bbox_inches="tight")
#         plt.close()
#         print(f"Saved: {output_png}")

# Generate the Fact and Stance heatmaps of the MAEs errors
# of all languages and countries
def heatmap_language_country_mae():

    csv_path = f"{EVALUATIONS_PATH}/{MAE}/lang_country_mae_summary.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")

    for test in [FACT, STANCE]:

        language_col = (
            f"{SCENARIO_LANGUAGE}_{test}_MAE"
        )

        country_col = (
            f"{SCENARIO_COUNTRY}_{test}_MAE"
        )

        lang_nat_col = (
            f"{SCENARIO_LAN_NAT}_{test}_MAE"
        )

        # -----------------------------
        # Language scenario rows
        # -----------------------------
        language_rows = (
            df[
                df[language_col].notna()
            ][
                ["language", language_col]
            ]
            .copy()
            .rename(columns={
                "language": "Language",
                language_col: "Value"
            })
        )

        # -----------------------------
        # Country scenario rows
        # -----------------------------
        country_rows = (
            df[
                df[country_col].notna()
            ][
                ["country", country_col]
            ]
            .copy()
            .rename(columns={
                "country": "Country",
                country_col: "Value"
            })
        )

        # -----------------------------
        # Language-country rows
        # -----------------------------
        lang_country_rows = (
            df[
                df[lang_nat_col].notna()
            ][
                [
                    "language",
                    "country",
                    lang_nat_col
                ]
            ]
            .copy()
            .rename(columns={
                "language": "Language",
                "country": "Country",
                lang_nat_col: "Value"
            })
        )

        # -----------------------------
        # Unique labels
        # -----------------------------
        languages = sorted(
            set(
                language_rows["Language"]
                .dropna()
            )
            |
            set(
                lang_country_rows["Language"]
                .dropna()
            )
        )

        countries = sorted(
            set(
                country_rows["Country"]
                .dropna()
            )
            |
            set(
                lang_country_rows["Country"]
                .dropna()
            )
        )

        aggregate_row = (
            SCENARIO_LABELS[
                SCENARIO_COUNTRY
            ]
        )

        aggregate_col = (
            SCENARIO_LABELS[
                SCENARIO_LANGUAGE
            ]
        )

        full_rows = (
            languages + [aggregate_row]
        )

        full_columns = (
            [aggregate_col] + countries
        )

        matrix = pd.DataFrame(
            np.nan,
            index=full_rows,
            columns=full_columns
        )

        # -----------------------------
        # Fill language averages
        # -----------------------------
        for _, row in language_rows.iterrows():

            matrix.loc[
                row["Language"],
                aggregate_col
            ] = row["Value"]

        # -----------------------------
        # Fill country averages
        # -----------------------------
        for _, row in country_rows.iterrows():

            matrix.loc[
                aggregate_row,
                row["Country"]
            ] = row["Value"]

        # -----------------------------
        # Fill language-country values
        # -----------------------------
        for _, row in lang_country_rows.iterrows():

            matrix.loc[
                row["Language"],
                row["Country"]
            ] = row["Value"]

        # -----------------------------
        # Row ordering
        # -----------------------------
        row_scores = []

        for r in matrix.index:

            row = (
                matrix.loc[r]
                .dropna()
            )

            score = row.mean()

            row_scores.append(
                (r, score)
            )

        row_order = [
            r for r, _
            in sorted(
                row_scores,
                key=lambda x: x[1]
            )
        ]

        # -----------------------------
        # Column ordering
        # -----------------------------
        col_scores = []

        for c in matrix.columns:

            col = (
                matrix[c]
                .dropna()
            )

            score = col.mean()

            col_scores.append(
                (c, score)
            )

        col_order = [
            c for c, _
            in sorted(
                col_scores,
                key=lambda x: x[1]
            )
        ]

        matrix = matrix.loc[
            row_order,
            col_order
        ]

        figsize = (
            max(
                12,
                len(matrix.columns) * 0.55
            ),
            max(
                10,
                len(matrix.index) * 0.42
            ),
        )

        plt.figure(figsize=figsize)

        ax = sns.heatmap(
            matrix,
            mask=matrix.isna(),
            vmin=0,
            vmax=1,
            cmap=CMAP_RG_INVERTED,
            annot=True,
            fmt=".2f",
            annot_kws={
                "fontsize":10
            },
            linewidths=0.3,
            linecolor="grey",
            cbar_kws={
                "label":"MAE",
                "shrink":0.8,
                "aspect":40,
                "fraction":0.03,
                "pad":0.02
            }
        )

        aggregate_row_idx = (
            list(matrix.index)
            .index(aggregate_row)
        )

        aggregate_col_idx = (
            list(matrix.columns)
            .index(aggregate_col)
        )

        ax.hlines(
            aggregate_row_idx,
            *ax.get_xlim(),
            colors="black",
            linewidth=2
        )

        ax.vlines(
            aggregate_col_idx + 1,
            *ax.get_ylim(),
            colors="black",
            linewidth=2
        )

        fontsize = 20

        plt.xlabel(
            "Countries",
            fontsize=fontsize
        )

        plt.ylabel(
            "Languages",
            fontsize=fontsize
        )

        plt.xticks(
            rotation=45,
            ha="right",
            fontsize=fontsize
        )

        plt.yticks(
            rotation=0,
            fontsize=fontsize
        )

        plt.tight_layout()

        output_png = (
            f"{GRAPHS_PATH}/MAE/"
            f"language_country_{test}_mae.png"
        )

        plt.savefig(
            output_png,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_png}")

# Generate lineplots of MAE scores across scenarios and countries
# Generate lineplots of MAE scores across scenarios and countries
def lineplot_scenario_comparison_mae():

    csv_path = (
        f"{EVALUATIONS_PATH}/{MAE}/"
        "lang_country_mae_summary.csv"
    )

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")

    for test in [FACT, STANCE]:

        language_col = (
            f"language_scenario_{test}_MAE"
        )

        country_col = (
            f"country_scenario_{test}_MAE"
        )

        lang_country_col = (
            f"language_country_scenario_{test}_MAE"
        )

        # -------------------------
        # LANGUAGE → aggregate to country axis
        # -------------------------
        language_df = (
            df[df[language_col].notna()]
            [["country", language_col]]
            .copy()
            .groupby("country", as_index=False)
            .mean()
            .rename(columns={
                "country": "Country",
                language_col: "language_scenario_mae"
            })
        )

        # -------------------------
        # COUNTRY scenario (already country-level)
        # -------------------------
        country_df = (
            df[df[country_col].notna()]
            [["country", country_col]]
            .drop_duplicates()
            .rename(columns={
                "country": "Country",
                country_col: "country_scenario_mae"
            })
        )

        # -------------------------
        # LANGUAGE-COUNTRY scenario
        # -------------------------
        lang_country_df = (
            df[df[lang_country_col].notna()]
            [["country", lang_country_col]]
            .rename(columns={
                "country": "Country",
                lang_country_col: "language_country_scenario_mae"
            })
        )

        # -------------------------
        # Merge on country only
        # -------------------------
        merged = country_df.merge(
            language_df,
            on="Country",
            how="outer"
        )

        merged = merged.merge(
            lang_country_df,
            on="Country",
            how="outer"
        )

        # -------------------------
        # Sort by country scenario
        # -------------------------
        merged = merged.sort_values(
            "Country"
        )

        # -------------------------
        # Plot
        # -------------------------
        plt.figure(figsize=(16, 6))

        sns.lineplot(
            data=merged,
            x="Country",
            y="language_scenario_mae",
            marker="o",
            label=SCENARIO_LABELS[SCENARIO_LANGUAGE],
        )

        sns.lineplot(
            data=merged,
            x="Country",
            y="country_scenario_mae",
            marker="o",
            label=SCENARIO_LABELS[SCENARIO_COUNTRY],
        )

        sns.lineplot(
            data=merged,
            x="Country",
            y="language_country_scenario_mae",
            marker="o",
            label=SCENARIO_LABELS[SCENARIO_LAN_NAT],
        )

        plt.xlabel("Country")
        plt.ylabel("MAE")

        plt.xticks(
            rotation=45,
            ha="right",
            fontsize=10
        )

        plt.legend()
        plt.tight_layout()

        output_path = f"{GRAPHS_PATH}/MAE/country_scenario_{test}_mae.png"

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_path}")


def plot_fact_shift_alignment():
    
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")

    for test in [FACT, STANCE]:

        # Aggregate by country (important if multiple rows per country)

        agg = df.groupby("Country").agg({
            "Rainbow Map": "mean",
            f"{test} MAE": "mean"
        }).reset_index()

        agg = agg.sort_values(f"{test} MAE")
        countries = agg["Country"]
        rainbow = agg["Rainbow Map"]
        mae = agg[f"{test} MAE"]
        
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Rainbow Map bars
        ax1.bar(countries, rainbow, alpha=0.6, label="Rainbow Map")
        ax1.set_ylabel("Rainbow Map score")

        ax1.tick_params(axis="x", rotation=40)
        ax1.set_xticks(range(len(countries)))
        ax1.set_xticklabels(countries, rotation=40, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(countries, mae, color="red", marker="o", linewidth=2, label=f"{test} MAE")
        ax2.set_ylabel(f"{test} MAE")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        plt.tight_layout()
        output_path = f"{GRAPHS_PATH}/MAE/rainbow_map_comparison_{test}.png"
        plt.savefig(output_path, dpi=300)

    
# Generate lineplots of the percentage scores across scenarios and countries
def lineplot_scenario_comparison_percentage():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df_all = pd.read_csv(csv_path, sep=";")

    for test in [FACT, STANCE]:
        per_col = test

        df_language = df_all[
            df_all[SCENARIO] == SCENARIO_LANGUAGE
        ].copy()

        df_country = df_all[
            df_all[SCENARIO] == SCENARIO_COUNTRY
        ].copy()

        df_lang_nat = df_all[
            df_all[SCENARIO] == SCENARIO_LAN_NAT
        ].copy()

        # Aggregate scenario scores
        language_agg = (
            df_language.groupby(
                ["Country", "languages"]
            )[per_col]
            .mean()
            .reset_index()
            .rename(columns={
                per_col: "language_scenario_per"
            })
        )

        country_agg = (
            df_country.groupby(
                ["Country"]
            )[per_col]
            .mean()
            .reset_index()
            .rename(columns={
                per_col: "country_scenario_per"
            })
        )

        lang_nat_agg = (
            df_lang_nat.groupby(
                ["Country", "languages"]
            )[per_col]
            .mean()
            .reset_index()
            .rename(columns={
                per_col: "language_country_scenario_per"
            })
        )

        # Aggregate Rainbow Map values
        rainbow_agg = (
            df_all.groupby(
                ["Country", "languages"]
            )["Rainbow Map"]
            .mean()
            .reset_index()
            .rename(columns={
                "Rainbow Map": "rainbow_map_score"
            })
        )

        # Merge everything together
        merged = language_agg.merge(
            country_agg,
            on="Country",
            how="left",
        )

        merged = merged.merge(
            lang_nat_agg,
            on=["Country", "languages"],
            how="left",
        )

        merged = merged.merge(
            rainbow_agg,
            on=["Country", "languages"],
            how="left",
        )

        merged = merged.sort_values("rainbow_map_score")

        # Plot
        plt.figure(figsize=(16, 6))

        sns.lineplot(
            data=merged,
            x="Country",
            y="language_scenario_per",
            marker="o",
            label=SCENARIO_LABELS[SCENARIO_LANGUAGE],
        )

        sns.lineplot(
            data=merged,
            x="Country",
            y="country_scenario_per",
            marker="o",
            label=SCENARIO_LABELS[SCENARIO_COUNTRY],
        )

        sns.lineplot(
            data=merged,
            x="Country",
            y="language_country_scenario_per",
            marker="o",
            label=SCENARIO_LABELS[SCENARIO_LAN_NAT],
        )

        # Rainbow Map reference line
        sns.lineplot(
            data=merged,
            x="Country",
            y="rainbow_map_score",
            marker="o",
            linestyle="--",
            label="Rainbow Map",
        )

        #plt.title(f"{test} Rainbow Meter Score (%) Across Scenarios")
        plt.xlabel("Country")
        plt.ylabel("Rainbow Meter Score (%)")

        plt.xticks(rotation=45, ha="right", fontsize=10)

        plt.legend()
        plt.tight_layout()

        output_path = (
            f"{GRAPHS_PATH}/percentage/country_scenario_{test}_percentage.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {output_path}")
        

def heatmap_country_models_percentage_distance():

    for test in [FACT, STANCE]:

        csv_path = f"{EVALUATIONS_PATH}/percentage/country_model_{test}_distance.csv"

        if not os.path.exists(csv_path):
            print(f"Missing file: {csv_path}")
            continue

        df = pd.read_csv(csv_path, sep=";", index_col=0)

        #df = df[df.index != "tot_distance"]
        df = df.drop('avg_distance', axis=1)


        fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 0.25), max(5, df.shape[0] * 0.25)))

        sns.heatmap(

            df,
            vmin=0,
            vmax=100,
            cmap=CMAP_RG_INVERTED,
            linewidths=0.2,
            linecolor="white",
            annot=True,
            annot_kws={"fontsize":7},
            fmt='.0f',
            cbar_kws={
                "shrink": 0.8,
                "aspect": 40,      # larger → thinner
                "fraction": 0.03,  # width of colorbar
                "pad": 0.02
            },
        ax=ax
        )
        
        rotation = 50
        fontsize = 14
        ylabel, xlabel = "Countries", "Models"
        
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

        
        # Bigger labels
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=rotation,
            fontsize=fontsize,
            ha="right",
        )
        
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=rotation,
            fontsize=fontsize,
            ha="right",      # horizontal alignment
            va="center",     # vertical alignment
            rotation_mode="anchor"
        )
        
        ax.tick_params(axis='both', length=0)
        plt.tight_layout()

        savefig = f"{GRAPHS_PATH}/percentage/country_models_{test}_percentage_distance.png"
        plt.savefig(
            savefig,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03
        )
        plt.close()
        print(f"Saved: {savefig}")
    

#Create the line graphs per each model of the pvalue score accross countries
def line_graphs_pvalue():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df_scenario = pd.read_csv(csv_path, sep=";")

    # Iterate over scenarios contained in the dataframe
    for scenario in SCENARIOS:
        df = df_scenario[df_scenario[SCENARIO] == scenario]

        if scenario == SCENARIO_LANGUAGE:
            df["x_label"] = df["languages"]
        elif scenario == SCENARIO_COUNTRY:
            df["x_label"] = df["Country"]
        elif scenario == SCENARIO_LAN_NAT:
            df["x_label"] = (df["languages"].astype(str) + " | "+ df["Country"].astype(str))

        df = df.sort_values("x_label")
        models = df["Model"].unique()

        for model in models:
            df_model = (df[df["Model"] == model].copy().reset_index(drop=True))

            x = range(len(df_model))
            plt.figure(figsize=(26, 10))

            for test in [FACT, STANCE]:
                plt.plot(
                    x,
                    df_model[f"{test} Pvalue"],
                    marker="o",
                    label=f"{test} P-Value"
                )

            # significance threshold
            plt.axhline(
                y=0.05,
                linestyle="--",
                label="p = 0.05"
            )

            plt.xticks(x, df_model["x_label"], rotation=90)
            plt.ylabel("P-Value")
            plt.xlabel("Language / Country")
            plt.title(f"{model} | {scenario} | P-Value Comparison")
            plt.legend()
            plt.tight_layout()

            save_path = (f"{GRAPHS_PATH}/pvalues/{scenario}/{model}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            plt.close()


#Back translation Heatmap
#back_translation()

#Weight coherence by validity scores
#model_performances()

#Generate the Fact and Stance heatmaps of the MAEs errors of all the models   
#mae_models()

#Generate the Fact and Stance heatmaps of the MAEs errors of all languages and countries
heatmap_language_country_mae()
plot_fact_shift_alignment()

#lineplot_scenario_comparison_mae()

#Generate a lineplots of the percentage errors accross scenarios and countries
lineplot_scenario_comparison_percentage()
heatmap_country_models_percentage_distance()


print(f"✅ Graphs Generated")