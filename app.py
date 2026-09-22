import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Page settings
st.set_page_config(
    page_title="GPA Analyzer & Predictor",
    page_icon="🎓",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🎓 GPA Analyzer")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "What-If Calculator",
        "Target CGPA",
        "AI Prediction"
    ]
)

# Title
st.title("🎓 GPA Analyzer & Predictor")

st.write(
    "Analyze your academic performance, calculate your CGPA, "
    "and explore your future GPA."
)

# Load dataset
data = pd.read_csv("results.csv")


# Grade points
grade_points = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
    "F": 0
}

data["grade_point"] = data["grade"].map(grade_points)
data["quality_points"] = data["units"] * data["grade_point"]

# Semester GPA
semester_gpa = data.groupby("semester").agg(
    total_units=("units", "sum"),
    total_quality_points=("quality_points", "sum")
)

semester_gpa["GPA"] = (
    semester_gpa["total_quality_points"] /
    semester_gpa["total_units"]
)

semester_gpa["GPA"] = semester_gpa["GPA"].round(2)

# Overall CGPA
total_units = data["units"].sum()
total_quality_points = data["quality_points"].sum()

cgpa = total_quality_points / total_units


if page == "Dashboard":

    st.subheader("Academic Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Current CGPA", f"{cgpa:.2f}")
    col2.metric("Total Credit Units", total_units)
    col3.metric("Semesters Completed", len(semester_gpa))

    st.subheader("📊 Semester Performance")

    st.dataframe(
        semester_gpa[["GPA"]],
        use_container_width=True
    )

    st.subheader("📈 GPA Trend")

    fig, ax = plt.subplots()

    ax.plot(
        semester_gpa.index,
        semester_gpa["GPA"],
        marker="o"
    )

    ax.set_xlabel("Semester")
    ax.set_ylabel("GPA")
    ax.set_ylim(0, 5)
    ax.set_xticks(semester_gpa.index)
    ax.grid(True)

    st.pyplot(fig)
# What-If CGPA Calculator
if page == "What-If Calculator":

    st.subheader("🔮 What-If CGPA Calculator")

    st.write(
        "Enter your expected GPA and credit units for the next "
        "semester to see how your CGPA could change."
    )

    # Your existing What-If code goes here

col1, col2 = st.columns(2)

with col1:
    next_semester_gpa = st.number_input(
        "Expected GPA",
        min_value=0.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

with col2:
    next_semester_units = st.number_input(
        "Expected Credit Units",
        min_value=1,
        value=15,
        step=1
    )

if st.button("Calculate Projected CGPA"):

    projected_quality_points = (
        next_semester_gpa * next_semester_units
    )

    projected_total_quality_points = (
        total_quality_points + projected_quality_points
    )

    projected_total_units = (
        total_units + next_semester_units
    )

    projected_cgpa = (
        projected_total_quality_points /
        projected_total_units
    )

    st.success(
        f"Your projected CGPA is **{projected_cgpa:.2f}**"
    )

    if projected_cgpa > cgpa:
        st.info("📈 Your CGPA is expected to improve.")
    elif projected_cgpa < cgpa:
        st.warning("📉 Your CGPA is expected to decrease.")
    else:
        st.info("➡️ Your CGPA is expected to remain the same.")
        # Target CGPA Calculator
if page == "Target CGPA":

    st.subheader("🎯 Target CGPA Calculator")

    # Your existing Target CGPA code goes here
st.write(
    "Find the GPA you need in your next semester "
    "to reach your target CGPA."
)

col1, col2 = st.columns(2)

with col1:
    target_cgpa = st.number_input(
        "Target CGPA",
        min_value=0.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

with col2:
    target_units = st.number_input(
        "Next Semester Credit Units",
        min_value=1,
        value=15,
        step=1
    )

if st.button("Calculate Required GPA"):

    required_quality_points = (
        target_cgpa * (total_units + target_units)
    ) - total_quality_points

    required_gpa = (
        required_quality_points / target_units
    )

    st.write(f"Current CGPA: **{cgpa:.2f}**")
    st.write(f"Target CGPA: **{target_cgpa:.2f}**")

    if required_gpa > 5:
        st.error(
            f"You would need a GPA of {required_gpa:.2f}, "
            "which is above the maximum GPA of 5.00."
        )

    elif required_gpa <= 0:
        st.success(
            "🎉 You have already reached or exceeded this target."
        )

    else:
        st.success(
            f"You need a GPA of **{required_gpa:.2f}** "
            "next semester to reach your target."
        )
if page == "AI Prediction":

    st.subheader("🤖 AI-Based GPA Prediction")

    # Your existing AI prediction
    # and Predicted Future CGPA code goes here
st.write(
    "The system uses Linear Regression to analyze previous "
    "semester GPAs and predict the GPA for the next semester."
)

X = semester_gpa.index.values.reshape(-1, 1)
y = semester_gpa["GPA"].values

model = LinearRegression()
model.fit(X, y)

next_semester = np.array([
    [semester_gpa.index.max() + 1]
])

predicted_gpa = model.predict(next_semester)[0]

# Keep prediction within the valid GPA range
predicted_gpa = max(0, min(5, predicted_gpa))

st.metric(
    "Predicted Next Semester GPA",
    f"{predicted_gpa:.2f}"
)

if predicted_gpa > cgpa:
    st.success("📈 The prediction shows a possible improvement.")
elif predicted_gpa < cgpa:
    st.warning("📉 The prediction shows a possible decrease.")
else:
    st.info("➡️ The prediction is close to your current CGPA.")
# Predicted Future CGPA
st.subheader("🔮 Predicted Future CGPA")

predicted_units = st.number_input(
    "Expected Credit Units for Predicted Semester",
    min_value=1,
    value=15,
    step=1
)

if st.button("Calculate Predicted CGPA"):

    predicted_quality_points = (
        predicted_gpa * predicted_units
    )

    predicted_cgpa = (
        total_quality_points + predicted_quality_points
    ) / (
        total_units + predicted_units
    )

    st.metric(
        "Predicted Future CGPA",
        f"{predicted_cgpa:.2f}"
    )

    if predicted_cgpa > cgpa:
        st.success(
            f"📈 Your CGPA could increase from {cgpa:.2f} "
            f"to approximately {predicted_cgpa:.2f}."
        )

    elif predicted_cgpa < cgpa:
        st.warning(
            f"📉 Your CGPA could decrease from {cgpa:.2f} "
            f"to approximately {predicted_cgpa:.2f}."
        )

    else:
        st.info(
            "➡️ Your predicted CGPA is approximately the same "
            "as your current CGPA."
        )
# Performance Insights
st.subheader("💡 Performance Insights")

first_gpa = semester_gpa["GPA"].iloc[0]
last_gpa = semester_gpa["GPA"].iloc[-1]

if last_gpa > first_gpa:
    st.success(
        f"📈 Your GPA improved from {first_gpa:.2f} "
        f"in Semester 1 to {last_gpa:.2f} in your latest semester."
    )

elif last_gpa < first_gpa:
    st.warning(
        f"📉 Your GPA decreased from {first_gpa:.2f} "
        f"in Semester 1 to {last_gpa:.2f} in your latest semester."
    )

else:
    st.info(
        f"➡️ Your GPA has remained stable at approximately "
        f"{last_gpa:.2f}."
    )

if last_gpa >= 4.5:
    st.success(
        "🌟 Excellent academic performance. "
        "Keep maintaining your current level."
    )

elif last_gpa >= 3.5:
    st.info(
        "👍 Good academic performance. "
        "Continue working toward further improvement."
    )

elif last_gpa >= 2.5:
    st.warning(
        "⚠️ Your performance is average. "
        "Consider focusing on weaker courses."
    )

else:
    st.error(
        "⚠️ Your GPA needs improvement. "
        "Consider creating a structured study plan."
    )
