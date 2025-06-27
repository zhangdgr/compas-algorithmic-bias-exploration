import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

df = pd.read_csv("compas-scores-two-years_cleaned2.csv")

# create age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 100], labels=["18–25", "26–45", "46+"])

# Sidebar w/ filters
st.sidebar.title("Filters")
selected_race = st.sidebar.multiselect("Race", options=df["race"].unique(), default=df["race"].unique())
selected_gender = st.sidebar.radio("Gender", options=["Male", "Female", "All"], index=2)
selected_age_groups = st.sidebar.multiselect("Age Group", options=df["age_group"].unique().tolist(), default=df["age_group"].unique().tolist())


# filters
filtered_df = df[df["race"].isin(selected_race)]
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["sex"] == selected_gender]
filtered_df = filtered_df[filtered_df["age_group"].isin(selected_age_groups)]

filtered_df = filtered_df.copy()
rng = np.random.default_rng(seed=42)  # fixed seed for consistent jitter

filtered_df['priors_count_jitter'] = filtered_df['priors_count'] + rng.uniform(-0.15, 0.15, len(filtered_df))
filtered_df['decile_score_jitter'] = filtered_df['decile_score'] + rng.uniform(-0.15, 0.15, len(filtered_df))
filtered_df['juv_other_count_jitter'] = filtered_df['juv_other_count'] + rng.uniform(-0.15, 0.15, len(filtered_df))
filtered_df['age_jitter'] = filtered_df['age'] + rng.uniform(-0.15, 0.15, len(filtered_df))


# Introduction!
st.title("Exploring Algorithmic Bias in Criminal Justice: A Case Study on COMPAS")

st.markdown("""
### About the Dataset
This dashboard explores data from the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) tool, which is a risk assessment algorithm used by courts to predict the likelihood of recidivism. The dataset includes information on defendants such as demographics, prior convictions, juvenile charges, COMPAS-assigned risk scores, and actual recidivism outcomes over a two-year period.
            
### Tips on Exploration
The goal of this dashboard is to help explore the COMPAS dataset, and use it to examine whether COMPAS scores are fair and predictive across different demographic groups, particularly focusing on racial disparities. Through interaction and filtering our visualizations, we can try to uncover potential bias in how scores are distributed, and how accurately they reflect real-world outcomes.
""")

with st.expander("Key Definitions", expanded=True):
    st.markdown("""
**Recidivism**: When a defendant commits a new offense within a specified time period  
(e.g., within two years of release).

**COMPAS / Decile Score**: A COMPAS-assigned risk score ranging from **1 to 10**,  
where **10 = highest predicted risk** of recidivism and **1 = lowest**.
""")


# Demographic section!!
st.markdown("---")
st.markdown("## Explore Dataset Demographics")
st.markdown("Before diving into risk scores or outcomes, please use the charts below to understand the **racial**, **age**, and **gender** breakdown of the dataset.")
st.markdown("This will help you decide which demographic filters to apply using the **sidebar** on the left.")

# By Race
with st.expander("By Race", expanded=False):
    st.markdown("### COMPAS Score Distribution by Race")
    st.altair_chart(alt.Chart(filtered_df).mark_boxplot().encode(
        x=alt.X('race:N', title='Race'),
        y=alt.Y('decile_score:Q', title='COMPAS Score'),
        tooltip=['race', 'decile_score']
    ).properties(width=400, height=300), use_container_width=True)

    st.markdown("### Count of Defendants by Race")
    st.altair_chart(alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('race:N', title='Race'),
        y=alt.Y('count():Q', title='Number of Defendants'),
        tooltip=['race', 'count()']
    ).properties(width=400, height=300), use_container_width=True)

    st.markdown("### Recidivism by Race")
    st.altair_chart(alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('race:N', title='Race'),
        y=alt.Y('count():Q', stack='normalize', title='Proportion'),
        color=alt.Color(
    'two_year_recid:N',
    title='Recidivated',
    legend=alt.Legend(
        labelExpr="datum.label == 0 ? 'No' : 'Yes'"
    )
),
        tooltip=['race', 'two_year_recid', 'count()']
    ).properties(width=400, height=300), use_container_width=True)

# By Age Group
with st.expander("By Age Group", expanded=False):
    st.markdown("### COMPAS Score Distribution by Age Group")
    st.altair_chart(alt.Chart(filtered_df).mark_boxplot().encode(
        x=alt.X('age_group:N', title='Age Group'),
        y=alt.Y('decile_score:Q', title='COMPAS Score'),
        tooltip=['age_group', 'decile_score']
    ).properties(width=400, height=300), use_container_width=True)

    st.markdown("### Count of Defendants by Age Group")
    st.altair_chart(alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('age_group:N', title='Age Group'),
        y=alt.Y('count():Q', title='Number of Defendants'),
        tooltip=['age_group', 'count()']
    ).properties(width=400, height=300), use_container_width=True)

    st.markdown("### Recidivism by Age Group")
    st.altair_chart(alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('age_group:N', title='Age Group'),
        y=alt.Y('count():Q', stack='normalize', title='Proportion'),
        color=alt.Color(
    'two_year_recid:N',
    title='Recidivated',
    legend=alt.Legend(
        labelExpr="datum.label == 0 ? 'No' : 'Yes'"
    )
),
        tooltip=['age_group', 'two_year_recid', 'count()']
    ).properties(width=400, height=300), use_container_width=True)

# By Gender
with st.expander("By Gender", expanded=False):
    st.markdown("### COMPAS Score Distribution by Gender")
    st.altair_chart(alt.Chart(filtered_df).mark_boxplot().encode(
        x=alt.X('sex:N', title='Gender'),
        y=alt.Y('decile_score:Q', title='COMPAS Score'),
        tooltip=['sex', 'decile_score']
    ).properties(width=400, height=300), use_container_width=True)

    st.markdown("### Count of Defendants by Gender")
    st.altair_chart(alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('sex:N', title='Gender'),
        y=alt.Y('count():Q', title='Number of Defendants'),
        tooltip=['sex', 'count()']
    ).properties(width=400, height=300), use_container_width=True)

    st.markdown("### Recidivism by Gender")
    st.altair_chart(alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('sex:N', title='Gender'),
        y=alt.Y('count():Q', stack='normalize', title='Proportion'),
        color=alt.Color(
    'two_year_recid:N',
    title='Recidivated',
    legend=alt.Legend(
        labelExpr="datum.label == 0 ? 'No' : 'Yes'"
    )
),
        tooltip=['sex', 'two_year_recid', 'count()']
    ).properties(width=400, height=300), use_container_width=True)

# Define brushing tool to help interaction
df = df[df["decile_score"].notnull()]
brush = alt.selection_interval(name='score_brush', encodings=['x'])

# Histogram for brushing COMPAS scores, add a description to help viewers use the brushing tool over the three graphs
st.markdown("---")
st.markdown("## Interactive Risk Exploration (Brush the Top Histogram to Filter)")
st.markdown("Use the brush tool below to **CLICK AND DRAG THE TOP HISTOGRAM** to select a COMPAS score range. This filters the scatterplot below and stacked bar chart.")
st.markdown("- The **first chart** (Count of defendants in each COMPAS score bracket) shows how many people fall into each COMPAS score bracket.")
st.markdown("- The **second chart** (COMPAS Score vs Prior Convictions) explores whether defendants with higher COMPAS scores tend to have more prior convictions, grouped by race. **1 dot = 1 defendant.** Jitter is applied to prevent overlap of defendant data. **ZOOM AND PAN TO EXPLORE DENSE REGIONS!**") 
st.markdown("- The **third chart** shows the recidivism rate for people in each COMPAS score bracket, to help judge predictive accuracy of COMPAS.")

score_histogram = alt.Chart(filtered_df).mark_bar(size=20).encode(
    x=alt.X('decile_score:Q', title='COMPAS Score'),
    y=alt.Y('count():Q', title='Number of Defendants'),
    tooltip=['decile_score', 'count()']
).add_params(
    brush
).properties(width=600, height=300)

linked_scatter = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.5).encode(
    x=alt.X('priors_count_jitter:Q', title='Prior Convictions'),
    y=alt.Y('decile_score_jitter:Q', title='COMPAS Score'),
    color=alt.Color('race:N', title='Race'),
    tooltip=['race', 'sex', 'priors_count', 'decile_score']
).transform_filter(
    brush
).properties(width=600, height=350).interactive()

recidivism_chart = alt.Chart(filtered_df).mark_bar(size=20).encode(
    x=alt.X('decile_score:Q', title='COMPAS Score'),
    y=alt.Y('count():Q', stack='normalize', title='Proportion'),
     color=alt.Color(
            'two_year_recid:N',
            title='Recidivism Outcome',
            legend=alt.Legend(
                labelExpr="datum.label == 0 ? 'Did not recidivate' : 'Recidivated'"
            )
        ),
    tooltip=['decile_score', 'two_year_recid', 'count()']
).transform_filter(
    brush
).properties(width=600, height=350)

combined_chart = (
    score_histogram
    & linked_scatter
    & recidivism_chart
).resolve_scale(color='independent')

st.altair_chart(combined_chart, use_container_width=True)

# Error Classification Chart containing raw data in number form
def classify(row):
    if row['decile_score'] >= 7:
        return 'True Positive' if row['two_year_recid'] == 1 else 'False Positive'
    else:
        return 'False Negative' if row['two_year_recid'] == 1 else 'True Negative'

filtered_df['error_type'] = filtered_df.apply(classify, axis=1)

# Define the desired order for error types
error_order = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
filtered_df['error_type'] = pd.Categorical(
    filtered_df['error_type'],
    categories=error_order,
    ordered=True
)

rank_map = {'True Positive': 3,
            'False Positive': 2,
            'False Negative': 1,
            'True Negative': 0}

filtered_df['error_rank'] = filtered_df['error_type'].map(rank_map)


error_chart = (
    alt.Chart(filtered_df)
       .mark_bar()
       .encode(
           x='race:N',
           y=alt.Y('count()', stack='normalize', title='Proportion'),
           color=alt.Color(
               'error_type:N',
               title='Prediction Outcome',
               scale=alt.Scale(
                   domain=error_order,                       # legend order
                   range=['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
               ),
               legend=alt.Legend(values=error_order)
           ),
           # ← drives *stack* order to match the legend
           order=alt.Order('error_rank:Q'),                  # new line
           tooltip=['race', 'error_type', 'count()']
       )
       .properties(width=600, height=400)
)



# Error analysis, important
st.markdown("---")
st.markdown("## False Positive/False Negative Error Analysis")
st.markdown("This chart shows the distribution of prediction outcomes (True Positives, False Positives, False Negatives, True Negatives) based on a risk threshold of 7 (the value used for many COMPAS studies).")
st.markdown("- **True Positive** means a defendant was predicted high risk and did recidivate.")
st.markdown("- **False Positive** means predicted high risk but did not recidivate. A **False Negative** means predicted low risk but did recidivate.")
st.markdown("- **False Negative** means predicted low risk but did recidivate.")
st.markdown("- **True Negative** means predicted low risk and did not recidivate.")
st.altair_chart(error_chart, use_container_width=True)

# Fairness Metrics in number form
st.markdown("### Fairness Metrics by Race")
metrics_df = filtered_df.copy()
metrics_df['high_risk'] = metrics_df['decile_score'] >= 7
fairness = metrics_df.groupby('race').apply(lambda x: pd.Series({
    'False Positive Rate': ((x['high_risk']) & (x['two_year_recid'] == 0)).sum() / ((x['two_year_recid'] == 0).sum() or 1),
    'False Negative Rate': ((~x['high_risk']) & (x['two_year_recid'] == 1)).sum() / ((x['two_year_recid'] == 1).sum() or 1),
    'True Positive Rate': ((x['high_risk']) & (x['two_year_recid'] == 1)).sum() / ((x['two_year_recid'] == 1).sum() or 1),
    'True Negative Rate': ((~x['high_risk']) & (x['two_year_recid'] == 0)).sum() / ((x['two_year_recid'] == 0).sum() or 1)
})).reset_index()

st.dataframe(fairness.style.format({
    "False Positive Rate": "{:.1%}",
    "False Negative Rate": "{:.1%}",
    "True Positive Rate": "{:.1%}",
    "True Negative Rate": "{:.1%}"
}))

# Divider and heading for insights in different correlations, only two for now, maybe add more later after feedback
st.markdown("---")
st.markdown("## Correlation Insights")
st.markdown("This section explores how COMPAS scores relate to other variables like age, prior convictions, juvenile charges, and observed recidivism. These insights help assess fairness, predictive accuracy, and potential disparities.")

# 1. COMPAS Score vs Age
st.markdown("### COMPAS Score vs Age")
st.markdown(
    "This scatterplot show the relationship between a defendant's age and COMPAS score (jitter applied on both axes). **1 dot = 1 defendant.** **ZOOM AND PAN TO INSPECT DENSE REGIONS!**")

age_chart = (
    alt.Chart(filtered_df)
        .mark_circle(size=60, opacity=0.45)
        .encode(
            x=alt.X('age_jitter:Q', title='Age'),
            y=alt.Y('decile_score_jitter:Q', title='COMPAS Score'),
            color=alt.Color('race:N', title='Race'),
            tooltip=['age', 'decile_score', 'race']
        )
        .properties(width=600, height=400)
)

st.altair_chart(age_chart.interactive(), use_container_width=True)


# 2. Recidivism Rate by COMPAS Score
st.markdown("### Recidivism Rate by COMPAS Score")
st.markdown("This line chart shows the proportion of people who actually recidivated at each COMPAS score. It may help assess how well COMPAS scores align with observed outcomes.")
recid_rate = (
    filtered_df.groupby('decile_score')
    .agg(recidivated=('two_year_recid', 'mean'), count=('two_year_recid', 'count'))
    .reset_index()
)

recid_rate_chart = alt.Chart(recid_rate).mark_line(point=True).encode(
    x=alt.X('decile_score:O', title='COMPAS Score'),
    y=alt.Y('recidivated:Q', title='Observed Recidivism Rate'),
    tooltip=['decile_score', 'recidivated', 'count']
).properties(width=600, height=400)

st.altair_chart(recid_rate_chart, use_container_width=True)

# 3. Average COMPAS Score by Race and Age Group
st.markdown("### Average COMPAS Score by Race and Age Group")
st.markdown("This heatmap displays the average COMPAS score by race and age group. It may help us visualize patterns or disparities across demographic combinations.")
heatmap_data = (
    filtered_df.groupby(['race', 'age_group'])['decile_score']
    .mean().reset_index(name='avg_score')
)

heatmap = alt.Chart(heatmap_data).mark_rect().encode(
    x=alt.X('age_group:N', title='Age Group'),
    y=alt.Y('race:N', title='Race'),
    color=alt.Color('avg_score:Q', title='Avg COMPAS Score', scale=alt.Scale(scheme='reds')),
    tooltip=['race', 'age_group', 'avg_score']
).properties(width=400, height=300)

st.altair_chart(heatmap, use_container_width=True)

# 4. COMPAS Score vs Juvenile Charges
st.markdown("### COMPAS Score vs Juvenile Charges")
st.markdown("This scatterplot examines how the number of juvenile offenses correlates with COMPAS scores. It may help identify if early-life charges disproportionately impact risk assessments, grouped by race. **1 dot = 1 defendant.** **ZOOM AND PAN TO INSPECT DENSE REGIONS!**")
juv_chart = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.5).encode(
    x=alt.X('juv_other_count_jitter:Q', title='Juvenile Other Charges'),
    y=alt.Y('decile_score_jitter:Q', title='COMPAS Score'),
    color=alt.Color('race:N', title='Race'),
    tooltip=['juv_other_count', 'decile_score', 'race']
).properties(width=600, height=400)

st.altair_chart(juv_chart.interactive(), use_container_width=True)

st.markdown("---")
st.markdown("## Key Takeaways and Further Reflection")

st.markdown("""
### Key Takeaways
- COMPAS scores aim to predict recidivism. Our visualizations imply differences in how scores and outcomes vary by demographics: **race**, **age**, and **gender**.
- **False positives and false negatives** occur at different rates across demographic groups, according to this data.
- Variables like **prior convictions**, **age**, and **juvenile history** all influence COMPAS score, not always in simple ways.

### Questions to Consider (Please reflect on these further!)
- What patterns stood out to you across different demographic groups?
- How might someone be affected by receiving a high COMPAS score that doesn’t match their actual behavior?
- What additional data or context would help improve the fairness of risk assessments?
""")