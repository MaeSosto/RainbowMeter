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
MODEL_PERFORMANCES_PATH = "model_performances"
MAE = "MAE"
PERCENTAGE = "percentage"
GRAPHS_PATH = "graphs"
os.makedirs(GRAPHS_PATH, exist_ok=True)
MAE_PATH =f"{GRAPHS_PATH}/{MAE}/"
os.makedirs(MAE_PATH, exist_ok=True)
PERCENTAGE_PATH = f"{GRAPHS_PATH}/{PERCENTAGE}/" 
os.makedirs(PERCENTAGE_PATH, exist_ok=True)
for s in SCENARIOS:
    os.makedirs(f"{GRAPHS_PATH}/{MODEL_PERFORMANCES_PATH}/{s}", exist_ok=True)

#Generate back translation dataframe and heatmap
def back_translation():
    df = pd.read_csv(f"data/back_translation.csv", sep=";", index_col="model")

    # Remove avg_score if present
    if "avg_score" in df.columns:
        df = df.drop(columns=["avg_score"])

    # Convert to numeric (in case some values are read as strings)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    figsize = (16, 4)
    fontsize = 13
    rotation = 35
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap=CMAP_RG,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        annot_kws={"fontsize":7},
        fmt=".1f",
        cbar_kws={
                #"shrink": 0.8,
                "aspect": 40,      
                "fraction": 0.03,  
                "pad": 0.01
            },
    )

    # plt.xlabel("Language")
    # plt.ylabel("Model")
    plt.ylabel(None)
    #plt.title("Models Performance in Back Translation Test")
    plt.yticks(rotation=rotation, fontsize=fontsize, ha='right', va='center', rotation_mode='anchor')
    plt.xticks(rotation=rotation, ha='right', fontsize=fontsize)
    plt.tight_layout()
    path_file = f"{GRAPHS_PATH}/back_translation.png"
    plt.savefig(path_file, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {path_file}")

def generate_heatmap(
    df,
    savefig,
    annot=False,
    transpose = False
):
    # Transpose
    if transpose:
        df = df.T
        xlabel, ylabel = ylabel, xlabel
    
    df = df.fillna(0)

    n_rows, n_cols = df.shape

    # Smaller cells
    cell_width = 0.4
    cell_height = 0.5

    fig_width = max(16, n_cols * cell_width)
    fig_height = fig_width / 3.3 #max(5, n_rows * cell_height)
    figsize=(fig_width, fig_height)
    if transpose:
        rotation = 30
        fontsize = 14
    rotation = 35
    fontsize = 15
    
    #print(figsize)
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap=CMAP_RG,
        linewidths=0.2,
        linecolor="white",
        annot=annot,
        annot_kws={"fontsize": 10},
        fmt=".1f",

        # Thinner color bar
        cbar_kws={
            #"shrink": 0.8,
            "aspect": 40,      # larger → thinner
            "fraction": 0.03,  # width of colorbar
            "pad": 0.01
        },

        ax=ax
    )
    
    #ax.set_xlabel(xlabel, fontsize=fontsize)
    #ax.set_ylabel(ylabel, fontsize=fontsize)

    
    # Bigger labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=rotation,
        fontsize=fontsize,
        ha="right",
        rotation_mode="anchor"
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
    plt.savefig(savefig, dpi=300, bbox_inches="tight", pad_inches=0.01)
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

            file_path = (f"{EVALUATIONS_PATH}/{MODEL_PERFORMANCES_PATH}/{scenario}/{m}.csv")
            if not os.path.exists(file_path):
                continue

            # IMPORTANT: first column is the model name index
            df_metric = pd.read_csv(file_path, sep=",", index_col=0)

            # Convert only data columns to numeric
            df_metric = df_metric.apply(pd.to_numeric, errors="coerce")
            avg_col = f"{title.split(' ')[0]} Average"
            if avg_col in df_metric.columns:
                df_metric = df_metric.drop(columns=[avg_col])
                
            # Sort only languages
            df_metric = df_metric.sort_index(axis=1)

            #xlabel = "Language" if scenario == SCENARIO_LANGUAGE else "Country" if scenario == SCENARIO_COUNTRY else "Lanaguage - Country" 
            
            generate_heatmap(
                df=df_metric,
                savefig=f"{GRAPHS_PATH}/{MODEL_PERFORMANCES_PATH}/{scenario}/{m}.png",
                #transpose=True,
                annot=True
            )
            

#Generate the Fact and Stance heatmaps of the MAEs errors of all the models           
def mae_model_country():
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
            output_path = (f"{GRAPHS_PATH}/{MAE}/model_country_{test}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")
            
# Generate the Fact and Stance heatmaps of the MAEs errors of all languages and countries
def mae_country_language():

    csv_path = f"{EVALUATIONS_PATH}/{MAE}/lang_country_mae_summary.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")

    for test in [FACT, STANCE]:
        language_col = (f"{SCENARIO_LANGUAGE}_{test}_MAE")
        country_col = (f"{SCENARIO_COUNTRY}_{test}_MAE")
        lang_nat_col = (f"{SCENARIO_LAN_NAT}_{test}_MAE")

        # Language scenario rows
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

        # Country scenario rows
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

        # Language-country rows
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

        # Unique labels
        languages = sorted(set(language_rows["Language"].dropna())|set(lang_country_rows["Language"].dropna()))
        countries = sorted(set(country_rows["Country"].dropna())|set(lang_country_rows["Country"].dropna()))
        aggregate_row = ("No Country")#SCENARIO_LABELS[SCENARIO_COUNTRY])
        aggregate_col = ("English") #SCENARIO_LABELS[SCENARIO_LANGUAGE])
        full_rows = (languages + [aggregate_row])
        full_columns = ([aggregate_col] + countries)

        matrix = pd.DataFrame(
            np.nan,
            index=full_rows,
            columns=full_columns
        )

        # Fill language averages
        for _, row in language_rows.iterrows():

            matrix.loc[
                row["Language"],
                aggregate_col
            ] = row["Value"]

        # Fill country averages
        for _, row in country_rows.iterrows():

            matrix.loc[
                aggregate_row,
                row["Country"]
            ] = row["Value"]

        # Fill language-country values
        for _, row in lang_country_rows.iterrows():

            matrix.loc[
                row["Language"],
                row["Country"]
            ] = row["Value"]

        # Row ordering
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

        # Column ordering
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

        # Multiply by 100
        matrix = matrix * 100
        
        figsize = (16,9)
        fontsize = 13
        rotation = 35
        
        plt.figure(figsize=figsize)

        ax = sns.heatmap(
            matrix,
            mask=matrix.isna(),
            #vmin=0,
            #vmax=100,
            cmap=CMAP_RG_INVERTED,
            annot=True,
            #fmt=".2f",
            annot_kws={
                "fontsize":7
            },
            linewidths=0.3,
            linecolor="grey",
            cbar_kws={
                "aspect": 40,      
                "fraction": 0.03, 
                "pad": 0.01
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
            colors="grey",
            linewidth=2
        )

        ax.vlines(
            aggregate_col_idx + 1,
            *ax.get_ylim(),
            colors="grey",
            linewidth=2
        )

        ax.set_xlim(0, len(matrix.columns))
        # plt.xlabel("Countries", fontsize=fontsize)
        # plt.ylabel("Languages", fontsize=fontsize)

        plt.yticks(rotation=rotation, fontsize=fontsize, ha='right', va='center', rotation_mode='anchor')
        plt.xticks(rotation=rotation,ha="right",fontsize=fontsize)
        plt.tight_layout()

        output_png = (f"{GRAPHS_PATH}/MAE/country_language_{test}.png")
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_png}")

# Generate lineplots of MAE scores across scenarios and countries
def mae_scenario_country():
    csv_path = (f"{EVALUATIONS_PATH}/{MAE}/lang_country_mae_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return
    df = pd.read_csv(csv_path, sep=";")

    for test in [FACT, STANCE]:
        language_col = (f"language_scenario_{test}_MAE")
        country_col = (f"country_scenario_{test}_MAE")
        lang_country_col = (f"language_country_scenario_{test}_MAE")

        # LANGUAGE → aggregate to country axis
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

        # COUNTRY scenario (already country-level)
        country_df = (
            df[df[country_col].notna()]
            [["country", country_col]]
            .drop_duplicates()
            .rename(columns={
                "country": "Country",
                country_col: "country_scenario_mae"
            })
        )

        # LANGUAGE-COUNTRY scenario
        lang_country_df = (
            df[df[lang_country_col].notna()]
            [["country", lang_country_col]]
            .rename(columns={
                "country": "Country",
                lang_country_col: "language_country_scenario_mae"
            })
        )

        # Merge on country only
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

        # Sort by country scenario
        merged = merged.sort_values("Country")

        # Plot
        figsize = (16, 4)
        plt.figure(figsize=figsize)
        fontsize = 13
        
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

        #plt.xlabel("Country")
        plt.margins(x=0.01) # Remove left/right spacing on x-axis
        #plt.ylim(0, 1)
        plt.xlabel(None)
        plt.ylabel("MAE")
        plt.xticks(rotation=35, ha="right", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5)
        #plt.legend(fontsize = fontsize, bbox_to_anchor=[0.5, 1.2], loc='center', ncol = 3)
        plt.legend(fontsize = fontsize)
        plt.tight_layout()
        output_path = f"{GRAPHS_PATH}/MAE/scenario_country_{test}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

#Shows a line plot of the RM scores and the MAE scores per each country
def mae_rm_country():
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
        
        figsize = (16, 4)
        fontsize = 13
        rotation = 35
        
        fig, ax1 = plt.subplots(figsize=figsize)

        # Rainbow Map bars
        ax1.bar(countries, rainbow, alpha=0.6, label="Rainbow Map")
        ax1.set_ylabel("Rainbow Meter Score (%)")
        ax1.tick_params(axis="x", rotation=rotation)
        ax1.set_xticks(range(len(countries)))
        ax1.set_xticklabels(countries, rotation=rotation, ha="right", fontsize=fontsize)
        ax2 = ax1.twinx()
        ax2.plot(countries, mae, color="red", marker="o", linewidth=2, label=f"{test} MAE")
        ax2.set_ylabel(f"{test} MAE")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=fontsize)
        plt.margins(x=0.01) # Remove left/right spacing on x-axis
        plt.yticks(fontsize=fontsize)
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        output_path = f"{GRAPHS_PATH}/MAE/mae_rm_country_{test}.png"
        plt.savefig(output_path, dpi=300)

    
# Generate lineplots of the percentage scores across scenarios and countries
def percentage_scenario_country():
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

        merged.to_csv(f"{EVALUATIONS_PATH}/percentage/scenario_country_{test}.csv")
        
        # Plot
        figsize = (16, 4)
        fontsize = 13
        plt.figure(figsize=figsize)

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
        #plt.xlabel("Country", fontsize= fontsize)
        plt.xlabel(None)
        plt.margins(x=0.01) # Remove left/right spacing on x-axis
        plt.ylabel("Rainbow Meter Score (%)", fontsize= fontsize)
        plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=30, ha="right", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()

        output_path = (f"{GRAPHS_PATH}/percentage/scenario_country_{test}.png")

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {output_path}")
        
def percentage_model_country_distance():
    for test in [FACT, STANCE]:
        csv_path = f"{EVALUATIONS_PATH}/percentage/country_model_{test}_distance.csv"
        if not os.path.exists(csv_path):
            print(f"Missing file: {csv_path}")
            continue

        df = pd.read_csv(csv_path, sep=";", index_col=0)

        #df = df[df.index != "tot_distance"]
        df = df.drop('avg_distance', axis=1)

        figsize = (16, 4)
        rotation = 35
        fontsize = 13
        
        #print(figsize)
        fig, ax = plt.subplots(figsize=figsize)

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
                #"shrink": 0.8,
                "aspect": 40,      # larger → thinner
                "fraction": 0.03,  # width of colorbar
                "pad": 0.01
            },
        ax=ax
        )
        
        # ylabel, xlabel = "Models", "Countries"
        
        # ax.set_xlabel(xlabel, fontsize=fontsize)
        # ax.set_ylabel(ylabel, fontsize=fontsize)

        
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

        savefig = f"{GRAPHS_PATH}/percentage/model_country_distance_{test}.png"
        plt.savefig(
            savefig,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.01
        )
        plt.close()
        print(f"Saved: {savefig}")


#Back translation Heatmap
back_translation()

#Weight coherence by validity scores
model_performances()

#Generate the Fact and Stance heatmaps of the MAEs errors of all the models   
#mae_model_country()
mae_country_language()
mae_rm_country()
mae_scenario_country()

#Generate a lineplots of the percentage errors accross scenarios and countries
percentage_scenario_country()
percentage_model_country_distance()


print(f"✅ Graphs Generated")