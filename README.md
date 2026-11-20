# Classification and Analysis of Collision and Contact Sports

**A data-driven test of whether sports really divide into "collision" and "contact," using survey data on concussion history, sleep and attention.**

[Read the paper (PDF)](TranfagliaEtAl_2025_SportClassification.pdf)

This was a DSCI 490 capstone project at Salisbury University (spring 2025). It was presented at the Salisbury University Student Research Conference (SUSRC) in April 2025.

## Background

Sports are usually labeled "collision" (football, ice hockey) or "contact" (basketball, soccer) based on subjective descriptions of physicality. Those labels shape safety rules and youth-participation policy. But they ignore repeated head impacts (RHIs), which research increasingly links to lasting problems with attention, memory and sleep.

This project asked whether cognitive and behavioral data can support, or challenge, the traditional split.

## Research questions

- Can an objective, data-driven method separate collision sports from contact sports?
- Does a two-cluster solution fit the data naturally?
- Which cognitive-behavioral features best separate people with a concussion history from people without one?
- Where do data-driven groupings disagree with the traditional labels?

## Data

The **Sports Cog 1** dataset was collected through a paid online survey organized by Salisbury University's Psychology Department. The survey covered:
- sports participation and concussion history;
- sleep, attention, pain and motivation measures;
- demographics.

After screening for consent, age (18–25), attention checks and completeness, **332 participants** remained. Soccer (24%), basketball (11%) and football (10%) were the most common sports, and over 80% of participants reported no concussions.

## Method

1. **Cleaning and standardization (pandas, regex):**
   - Removed personally identifying fields.
   - Converted sleep schedules to 24-hour time and calculated sleep duration.
   - Encoded categorical answers as binary, ordinal or one-hot.
   - **Parsed free-text answers with an LLM** (OpenAI `gpt-4o-mini` through LangChain, `ParseSports.py` and `ParseConcussions.py`). Answers such as *"Football, running back, all 4 years. Track, 3 years"* became structured `{Sport, Years, Position}` records, and concussion descriptions became counts and sports.
2. **Exploratory analysis:**
   - Sport and concussion distributions. Football makes up 9.6% of athletes but 21.4% of concussions.
   - Sleep quality and concentration problems, compared by concussion status.
3. **Feature selection (R):** two-sample proportion tests at p < 0.05 and p < 0.01 cut more than 150 features to **16 significant ones** (9 about sleep, 7 about attention).
4. **Clustering (scikit-learn):**
   - PCA for dimensionality reduction.
   - K-means for k = 2 to 5, plus hierarchical clustering and DBSCAN.
   - Evaluated with silhouette, Calinski-Harabasz and Davies-Bouldin scores.
   - Run on everyone, on athletes only, and on sleep-only, attention-only and combined feature sets.
5. **Comparison:** each sport's spread across the clusters, checked against its traditional label and against average concussion counts.

## Findings

- **Two clusters fit best.** All three metrics agreed on k = 2:
  - silhouette 0.35;
  - Calinski-Harabasz 147;
  - Davies-Bouldin 1.12.

  This supports a binary split. With only the 16 significant features, PCA showed clean separation that random or full feature sets did not.
- **Concussion history is tied to worse sleep and attention.**
  - Concussed athletes reported "fairly" or "very bad" sleep 45% of the time, against 18% of non-concussed athletes.
  - 21% of concussed athletes "always" had trouble concentrating, against 2%.
- **Attention features formed the tightest clusters** (silhouette up to about 0.81) and best separated people by concussion history across the full sample.
- **Sleep features lined up best with traditional labels,** especially in placing football as a collision sport when only athletes were analyzed.
- **The labels are leaky.** Football, soccer, basketball and volleyball players all appeared in both clusters, and results moved with the significance threshold. Position-level differences, such as a kicker compared with a lineman, probably matter more than the sport as a whole.

Future work: position-specific data, EEG measures, and a larger sample with more collision sports.

## Tech stack

- **Python 3.9:** pandas, NumPy, scikit-learn (PCA, KMeans, AgglomerativeClustering, DBSCAN, t-SNE, RobustScaler, StandardScaler), Matplotlib, Seaborn
- **R** (R Markdown) for the proportion tests and p-value tables
- **LangChain + OpenAI API** for parsing free text

## Repository structure

| Path | Contents |
|---|---|
| `DataFiltering.py`, `CompleteDataClean.py`, `CleanData.py`, `TheDataFixer.py`, `ColumnRenamer.py`, `MergeData.py`, `StandardizedTime.py`, `MilitaryConversion.py` | Filtering, cleaning, merging and time standardization |
| `ParseSports.py`, `ParseConcussions.py`, `MergeSports.py`, `SportClassification.py` | LLM-assisted parsing and sport labeling |
| `Clusters.py`, `ImprovedClustering*.py`, `MoreClustering.py`, `SignificantCustering*.py`, `MostSignificantClustering.py`, `sihoette_graph.py` | PCA, clustering experiments and cluster evaluation |
| `Visualizer.py`, `PlotGen.py`, `SportPieChart.py`, `ConcussionPieChart.py`, `AgeVsFreqeuencyAnswer.py` | Exploratory and results plots |
| `P-values/` | R Markdown proportion tests for sleep and attention features, p-value charts and write-ups |
| `Plots/` | Generated figures: distributions, PCA scatters, cluster profiles, silhouette scores |
| `Data/` | Intermediate and clustered datasets, plus clustering logs |

## Running the analysis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
# Only needed for the LLM parsing step:
pip install langchain langchain-openai
export OPEN_AI_KEY=your-key
```

- The scripts read from and write to `Data/` and `Plots/` using relative paths, so run them from the repository root.
- The R Markdown files in `P-values/` run in RStudio.

## Team

| Name | |
|---|---|
| [Kyle Tranfaglia](https://github.com/ktranfaglia1) | Team lead and data scientist |
| [Dustin O'Brien](https://github.com/Omniladder) | Data scientist |
| [Dylan Johnson](https://github.com/djohnson4525) | Data scientist |

Advised by Dr. Kyle Teller, Department of Mathematics, Salisbury University.
