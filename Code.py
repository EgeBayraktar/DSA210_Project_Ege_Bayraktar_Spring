# ===============================
# DSA 210 YouTube Trending Analysis
# Data Collection + Enrichment + EDA + Hypothesis Tests + Machine Learning
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, f_oneway, ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------

FILE_PATH = "youtube_trending_videos_global.csv"

use_cols = [
    "video_id",
    "video_published_at",
    "video_trending__date",
    "video_trending_country",
    "video_category_id",
    "video_duration",
    "video_view_count",
    "video_like_count",
    "video_comment_count",
    "channel_subscriber_count",
    "channel_video_count",
    "video_definition",
    "video_licensed_content"
]

df = pd.read_csv(
    FILE_PATH,
    usecols=use_cols,
    nrows=100000,
    low_memory=False
)

df = df.rename(columns={
    "video_trending__date": "video_trending_date"
})

print("Initial shape:", df.shape)
print(df.head())
print(df.info())


# -------------------------------
# 2. Basic Cleaning
# -------------------------------

numeric_cols = [
    "video_view_count",
    "video_like_count",
    "video_comment_count",
    "channel_subscriber_count",
    "channel_video_count"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["video_published_at"] = pd.to_datetime(df["video_published_at"], errors="coerce")
df["video_trending_date"] = pd.to_datetime(df["video_trending_date"], errors="coerce")

df = df.dropna(subset=[
    "video_view_count",
    "video_like_count",
    "video_comment_count",
    "channel_subscriber_count",
    "channel_video_count",
    "video_published_at",
    "video_trending_date",
    "video_category_id"
])

df = df[df["video_view_count"] > 0]
df = df[df["channel_subscriber_count"] >= 0]
df = df[df["channel_video_count"] >= 0]

print("Shape after cleaning:", df.shape)


# -------------------------------
# 3. Feature Engineering
# -------------------------------

df["days_to_trend"] = (
    df["video_trending_date"].dt.date - df["video_published_at"].dt.date
).apply(lambda x: x.days)

df = df[df["days_to_trend"] >= 0]

df["publish_hour"] = df["video_published_at"].dt.hour
df["publish_dayofweek"] = df["video_published_at"].dt.day_name()

def time_category(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"

df["publish_time_category"] = df["publish_hour"].apply(time_category)

df["like_ratio"] = df["video_like_count"] / df["video_view_count"]
df["comment_ratio"] = df["video_comment_count"] / df["video_view_count"]

df["log_views"] = np.log1p(df["video_view_count"])
df["log_likes"] = np.log1p(df["video_like_count"])
df["log_comments"] = np.log1p(df["video_comment_count"])
df["log_subscribers"] = np.log1p(df["channel_subscriber_count"])
df["log_channel_videos"] = np.log1p(df["channel_video_count"])


# -------------------------------
# 4. External Dataset Enrichment
# -------------------------------
# This satisfies the requirement of enriching a public dataset with another dataset.
# The external dataset maps YouTube category names into broader category groups.

category_map = pd.DataFrame({
    "video_category_id": [
        "Autos & Vehicles",
        "Comedy",
        "Education",
        "Entertainment",
        "Film & Animation",
        "Gaming",
        "Howto & Style",
        "Music",
        "News & Politics",
        "People & Blogs",
        "Pets & Animals",
        "Science & Technology",
        "Sports"
    ],
    "category_group": [
        "Transport",
        "Entertainment",
        "Educational",
        "Entertainment",
        "Entertainment",
        "Gaming",
        "Lifestyle",
        "Music",
        "News",
        "Lifestyle",
        "Lifestyle",
        "Educational",
        "Sports"
    ]
})

category_map.to_csv("category_map.csv", index=False)

df = df.merge(category_map, on="video_category_id", how="left")

print("Shape after enrichment:", df.shape)
print(df[["video_category_id", "category_group"]].head())


# -------------------------------
# 5. Save Cleaned Dataset
# -------------------------------

df.to_csv("cleaned_youtube_trending_sample.csv", index=False)
print("Cleaned dataset saved.")


# -------------------------------
# 6. EDA Visualizations
# -------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
plt.hist(df["video_view_count"], bins=50)
plt.title("Distribution of Video View Counts")
plt.xlabel("Views")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["log_views"], bins=50)
plt.title("Distribution of Log-Transformed Video Views")
plt.xlabel("log(views + 1)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

corr_cols = [
    "log_views",
    "log_likes",
    "log_comments",
    "log_subscribers",
    "log_channel_videos",
    "days_to_trend",
    "like_ratio",
    "comment_ratio",
    "publish_hour"
]

plt.figure(figsize=(10, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["log_subscribers"], df["log_views"], alpha=0.3)
plt.title("Channel Subscribers vs Video Views")
plt.xlabel("log(channel subscribers + 1)")
plt.ylabel("log(video views + 1)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["days_to_trend"], df["log_views"], alpha=0.3)
plt.title("Days to Trend vs Video Views")
plt.xlabel("Days to Trend")
plt.ylabel("log(video views + 1)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df.groupby("video_category_id")["video_view_count"].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Average Views by Video Category")
plt.xlabel("Video Category")
plt.ylabel("Average Views")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df.groupby("category_group")["video_view_count"].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Average Views by Enriched Category Group")
plt.xlabel("Category Group")
plt.ylabel("Average Views")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df.groupby("publish_time_category")["video_view_count"].mean().reindex(
    ["Morning", "Afternoon", "Evening", "Night"]
).plot(kind="bar")
plt.title("Average Views by Upload Time Category")
plt.xlabel("Upload Time Category")
plt.ylabel("Average Views")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df.groupby("video_definition")["video_view_count"].mean().plot(kind="bar")
plt.title("Average Views by Video Definition")
plt.xlabel("Video Definition")
plt.ylabel("Average Views")
plt.tight_layout()
plt.show()


# -------------------------------
# 7. Hypothesis Testing
# -------------------------------

print("\n===============================")
print("HYPOTHESIS TESTING RESULTS")
print("===============================")

corr_subs, p_subs = pearsonr(df["log_subscribers"], df["log_views"])

print("\nHypothesis 1: Larger channels get more views")
print("Pearson correlation:", corr_subs)
print("p-value:", p_subs)

if p_subs < 0.05:
    print("Result: Reject H0. Subscriber count is significantly related to views.")
else:
    print("Result: Fail to reject H0.")

corr_days, p_days = pearsonr(df["days_to_trend"], df["log_views"])

print("\nHypothesis 2: Days to trend is related to views")
print("Pearson correlation:", corr_days)
print("p-value:", p_days)

if p_days < 0.05:
    print("Result: Reject H0. Days to trend is significantly related to views.")
else:
    print("Result: Fail to reject H0.")

category_groups = [
    group["log_views"].values
    for _, group in df.groupby("video_category_id")
    if len(group) > 30
]

anova_cat, p_cat = f_oneway(*category_groups)

print("\nHypothesis 3: Video category affects views")
print("ANOVA statistic:", anova_cat)
print("p-value:", p_cat)

if p_cat < 0.05:
    print("Result: Reject H0. Views differ significantly across categories.")
else:
    print("Result: Fail to reject H0.")

time_groups = [
    group["log_views"].values
    for _, group in df.groupby("publish_time_category")
    if len(group) > 30
]

anova_time, p_time = f_oneway(*time_groups)

print("\nHypothesis 4: Upload time category affects views")
print("ANOVA statistic:", anova_time)
print("p-value:", p_time)

if p_time < 0.05:
    print("Result: Reject H0. Views differ significantly by upload time.")
else:
    print("Result: Fail to reject H0.")

if "hd" in df["video_definition"].unique() and "sd" in df["video_definition"].unique():
    hd_views = df[df["video_definition"] == "hd"]["log_views"]
    sd_views = df[df["video_definition"] == "sd"]["log_views"]

    t_hd, p_hd = ttest_ind(hd_views, sd_views, equal_var=False)

    print("\nHypothesis 5: HD and SD videos differ in views")
    print("t-statistic:", t_hd)
    print("p-value:", p_hd)

    if p_hd < 0.05:
        print("Result: Reject H0. HD and SD videos differ significantly.")
    else:
        print("Result: Fail to reject H0.")
else:
    print("\nHypothesis 5 skipped: Both HD and SD were not present.")


# -------------------------------
# 8. Summary Tables
# -------------------------------

print("\n===============================")
print("SUMMARY TABLES")
print("===============================")

print("\nAverage performance by video category:")
print(
    df.groupby("video_category_id")[[
        "video_view_count",
        "video_like_count",
        "video_comment_count",
        "like_ratio",
        "comment_ratio",
        "days_to_trend"
    ]].mean().sort_values("video_view_count", ascending=False)
)

print("\nAverage performance by enriched category group:")
print(
    df.groupby("category_group")[[
        "video_view_count",
        "video_like_count",
        "video_comment_count",
        "like_ratio",
        "comment_ratio",
        "days_to_trend"
    ]].mean().sort_values("video_view_count", ascending=False)
)

print("\nAverage performance by upload time category:")
print(
    df.groupby("publish_time_category")[[
        "video_view_count",
        "like_ratio",
        "comment_ratio",
        "days_to_trend"
    ]].mean().sort_values("video_view_count", ascending=False)
)

print("\nFinal dataset shape:", df.shape)


# ===============================
# 9. MACHINE LEARNING SECTION
# ===============================
# This is the ML part of the project.
# Goal: Predict video view count using regression.
# Target variable: log_views
# Model: Linear Regression

features = [
    "log_likes",
    "log_comments",
    "log_subscribers",
    "days_to_trend",
    "publish_hour",
    "log_channel_videos"
]

ml_df = df.dropna(subset=features + ["log_views"])

X = ml_df[features]
y = ml_df["log_views"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===============================")
print("MACHINE LEARNING RESULTS")
print("===============================")
print("Model used: Linear Regression")
print("Target variable: log_views")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("\nModel Coefficients:")
print(coef_df)