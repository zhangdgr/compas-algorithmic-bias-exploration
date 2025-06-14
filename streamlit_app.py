import streamlit as st
import pandas as pd
import altair as alt

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

# Introduction!
st.title("Exploring Algorithmic Bias in Criminal Justice: A Case Study on COMPAS")

st.markdown("""
### About the Dataset
This dashboard explores data from the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) tool, which is a risk assessment algorithm used by courts to predict the likelihood of recidivism. The dataset includes information on defendants such as demographics, prior convictions, juvenile charges, COMPAS-assigned risk scores, and actual recidivism outcomes over a two-year period.

### Tips on Exploration
The goal of this dashboard is to help explore the COMPAS dataset, and use it to examine whether COMPAS scores are fair and predictive across different demographic groups, particularly focusing on racial disparities. Through interaction and filtering our visualizations, we can try to uncover potential bias in how scores are distributed, and how accurately they reflect real-world outcomes.
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
        color=alt.Color('two_year_recid:N', title='Recidivated'),
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
        color=alt.Color('two_year_recid:N', title='Recidivated'),
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
        color=alt.Color('two_year_recid:N', title='Recidivated'),
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
st.markdown("- The **second chart** (COMPAS Score vs Prior Convictions) explores whether defendants with higher COMPAS scores tend to have more prior convictions, grouped by race.")
st.markdown("- The **third chart** shows the recidivism rate for people in each COMPAS score bracket, to help judge predictive accuracy of COMPAS.")

score_histogram = alt.Chart(filtered_df).mark_bar(size=20).encode(
    x=alt.X('decile_score:Q', title='COMPAS Score'),
    y=alt.Y('count():Q', title='Number of Defendants'),
    tooltip=['decile_score', 'count()']
).add_params(
    brush
).properties(width=600, height=300)

linked_scatter = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.5).encode(
    x=alt.X('priors_count:Q', title='Prior Convictions'),
    y=alt.Y('decile_score:Q', title='COMPAS Score'),
    color=alt.Color('race:N', title='Race'),
    tooltip=['race', 'sex', 'priors_count', 'decile_score']
).transform_filter(
    brush
).properties(width=600, height=350)

recidivism_chart = alt.Chart(filtered_df).mark_bar(size=20).encode(
    x=alt.X('decile_score:Q', title='COMPAS Score'),
    y=alt.Y('count():Q', stack='normalize', title='Proportion'),
    color=alt.Color('two_year_recid:N', title='Recidivated'),
    tooltip=['decile_score', 'two_year_recid', 'count()']
).transform_filter(
    brush
).properties(width=600, height=350)

combined_chart = score_histogram & linked_scatter & recidivism_chart
st.altair_chart(combined_chart, use_container_width=True)

# Error Classification Chart containing raw data in number form
def classify(row):
    if row['decile_score'] >= 7:
        return 'True Positive' if row['two_year_recid'] == 1 else 'False Positive'
    else:
        return 'False Negative' if row['two_year_recid'] == 1 else 'True Negative'

filtered_df['error_type'] = filtered_df.apply(classify, axis=1)

error_chart = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X('race:N', title='Race'),
    y=alt.Y('count():Q', stack='normalize', title='Proportion'),
    color=alt.Color('error_type:N', title='Prediction Outcome', scale=alt.Scale(domain=['True Positive', 'False Positive', 'False Negative', 'True Negative'], range=['#1b9e77', '#d95f02', '#7570b3', '#e7298a'])),
    tooltip=['race', 'error_type', 'count()']
).properties(width=600, height=400)

# Error analysis, important
st.markdown("---")
st.markdown("## False Positive/False Negative Error Analysis")
st.markdown("This chart shows the distribution of prediction outcomes (True Positives, False Positives, False Negatives, True Negatives) based on a risk threshold of 7. A **True Positive** means a defendant was predicted high risk and did recidivate. A **False Positive** means predicted high risk but did not recidivate. A **False Negative** means predicted low risk but did recidivate. A **True Negative** means predicted low risk and did not recidivate. Explore disparities across race.")
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
st.markdown("This section explores how COMPAS scores relate to other variables like prior convictions and juvenile charges using scatterplots grouped by race. Each includes trend line to visualize any potential linear relationships.")

st.markdown("### COMPAS Score vs Prior Convictions")
corr_chart = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.5).encode(
    x=alt.X('priors_count:Q', title='Prior Convictions'),
    y=alt.Y('decile_score:Q', title='COMPAS Score'),
    color=alt.Color('race:N', title='Race'),
    tooltip=['priors_count', 'decile_score', 'race']
).properties(width=600, height=400)

trend_line = corr_chart.transform_regression('priors_count', 'decile_score', method='linear').mark_line(color='black')

st.altair_chart(corr_chart + trend_line, use_container_width=True)

st.markdown("### COMPAS Score vs Juvenile Charges")
juv_chart = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.5).encode(
    x=alt.X('juv_other_count:Q', title='Juvenile Other Charges'),
    y=alt.Y('decile_score:Q', title='COMPAS Score'),
    color=alt.Color('race:N', title='Race'),
    tooltip=['juv_other_count', 'decile_score', 'race']
).properties(width=600, height=400)

juv_trend = juv_chart.transform_regression('juv_other_count', 'decile_score', method='linear').mark_line(color='black')

st.altair_chart(juv_chart + juv_trend, use_container_width=True)
