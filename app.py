import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="GPA Analyzer & Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD CUSTOM CSS
# =========================================================

def load_css():
    css_file = Path("style.css")

    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# =========================================================
# SESSION STATE
# =========================================================

if "manual_entries" not in st.session_state:
    st.session_state.manual_entries = []

if "manual_saved" not in st.session_state:
    st.session_state.manual_saved = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # BRAND
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🎓 GPA Analyzer</div>
            <p class="sidebar-brand-subtitle">
                Academic performance intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # DATA SOURCE
    st.markdown(
        '<div class="sidebar-section-title">DATA SOURCE</div>',
        unsafe_allow_html=True
    )

    data_source = st.radio(
        "Choose your results source",
        [
            "Use Sample CSV",
            "Upload CSV",
            "Enter Manually"
        ],
        key="data_source",
        label_visibility="collapsed"
    )

    # SHOW UPLOADER ONLY WHEN UPLOAD CSV IS SELECTED
    uploaded_file = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your results CSV",
            type=["csv"],
            key="results_csv_uploader"
        )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # NAVIGATION
    st.markdown(
        '<div class="sidebar-section-title">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Enter Results",
            "What-If Calculator",
            "Target CGPA",
            "AI Prediction"
        ],
        key="main_navigation",
        label_visibility="collapsed"
    )


    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)


    # FOOTER
    st.markdown(
        """
        <div class="sidebar-footer">
            <strong>GPA ANALYZER</strong>
            <small>Analyze • Predict • Improve</small>
        </div>
        """,
        unsafe_allow_html=True
    )
# =========================================================
# DATA SOURCE
# =========================================================

if data_source == "Use Sample CSV":

    try:
        data = pd.read_csv("results.csv")
    except FileNotFoundError:
        st.error("results.csv could not be found.")
        st.stop()


elif data_source == "Upload CSV":

    if uploaded_file is None:

        st.markdown(
            """
            <div class="content-card">
                <div class="section-title">Upload your results</div>
                <div class="section-subtitle">
                    Upload a CSV file using the uploader in the sidebar
                    to begin analysing your performance.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.stop()

    # Read the file selected in the sidebar
    data = pd.read_csv(uploaded_file)


else:

    if page == "Enter Results":

        st.markdown(
            """
            <div class="page-header">
                <div class="page-title">Enter Your Results</div>
                <div class="page-subtitle">
                    Add, edit and manage your academic results.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # CURRENT COURSES
        # -------------------------------------------------

        st.markdown(
            """
            <div class="section-title">Your Courses</div>
            <div class="section-subtitle">
                Edit existing courses directly or remove courses from the table.
            </div>
            """,
            unsafe_allow_html=True
        )

        if len(st.session_state.manual_entries) == 0:

            st.info(
                "No courses have been added yet. Use the form below to add your first course."
            )

        else:

            current_table = pd.DataFrame(
                st.session_state.manual_entries
            )

            edited_table = st.data_editor(
                current_table,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "semester": st.column_config.NumberColumn(
                        "Semester",
                        min_value=1,
                        step=1
                    ),
                    "course": st.column_config.TextColumn(
                        "Course Name"
                    ),
                    "units": st.column_config.NumberColumn(
                        "Units",
                        min_value=1,
                        step=1
                    ),
                    "grade": st.column_config.SelectboxColumn(
                        "Grade",
                        options=["A", "B", "C", "D", "E", "F"]
                    )
                },
                key="manual_results_table"
            )

            st.caption(
                "✏️ Edit any cell directly. Use the table's delete option to remove a course."
            )

            if st.button(
                "Apply Table Changes",
                type="primary",
                use_container_width=True,
                key="apply_table_changes"
            ):

                edited_data = edited_table.copy()

                edited_data["course"] = (
                    edited_data["course"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                edited_data["grade"] = (
                    edited_data["grade"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )

                if edited_data.empty:

                    st.session_state.manual_entries = []
                    st.session_state.manual_saved = False

                    if "active_data" in st.session_state:
                        del st.session_state.active_data

                    st.success("All courses have been removed.")
                    st.rerun()

                if (edited_data["course"] == "").any():

                    st.error(
                        "Every course must have a course name."
                    )
                    st.stop()

                edited_data["semester"] = pd.to_numeric(
                    edited_data["semester"],
                    errors="coerce"
                )

                if edited_data["semester"].isna().any():

                    st.error(
                        "Every course must have a valid semester."
                    )
                    st.stop()

                if (edited_data["semester"] <= 0).any():

                    st.error(
                        "Semester must be greater than 0."
                    )
                    st.stop()

                if (edited_data["semester"] % 1 != 0).any():

                    st.error(
                        "Semester must be a whole number."
                    )
                    st.stop()

                edited_data["units"] = pd.to_numeric(
                    edited_data["units"],
                    errors="coerce"
                )

                if edited_data["units"].isna().any():

                    st.error(
                        "Every course must have a valid number of units."
                    )
                    st.stop()

                if (edited_data["units"] <= 0).any():

                    st.error(
                        "Course units must be greater than 0."
                    )
                    st.stop()

                valid_grades = [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F"
                ]

                if (edited_data["grade"] == "").any():

                    st.error(
                        "Every course must have a grade."
                    )
                    st.stop()

                if not edited_data["grade"].isin(
                    valid_grades
                ).all():

                    st.error(
                        "Grades must be A, B, C, D, E or F."
                    )
                    st.stop()

                duplicate_mask = edited_data.duplicated(
                    subset=["semester", "course"],
                    keep=False
                )

                if duplicate_mask.any():

                    duplicate_rows = edited_data[
                        duplicate_mask
                    ]

                    duplicate_names = duplicate_rows[
                        ["semester", "course"]
                    ].drop_duplicates()

                    duplicate_messages = []

                    for _, row in duplicate_names.iterrows():

                        duplicate_messages.append(
                            f"{row['course']} "
                            f"(Semester {int(row['semester'])})"
                        )

                    st.error(
                        "Duplicate courses are not allowed "
                        "in the same semester."
                    )

                    st.warning(
                        "Duplicate(s): "
                        + ", ".join(duplicate_messages)
                    )

                    st.stop()

                st.session_state.manual_entries = (
                    edited_data.to_dict("records")
                )

                st.session_state.active_data = (
                    edited_data.copy()
                )

                st.session_state.manual_saved = True

                st.success(
                    "Table changes applied successfully."
                )

                st.rerun()

        st.divider()

        # -------------------------------------------------
        # ADD COURSE
        # -------------------------------------------------

        st.markdown(
            """
            <div class="section-title">Add New Course</div>
            <div class="section-subtitle">
                Enter one course at a time.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form(
            "course_entry_form",
            clear_on_submit=True
        ):

            col1, col2 = st.columns(2)

            with col1:

                semester_input = st.number_input(
                    "Semester",
                    min_value=-100,
                    max_value=100,
                    value=1,
                    step=1,
                    key="new_semester"
                )

                course_input = st.text_input(
                    "Course Name",
                    placeholder="e.g. Mathematics",
                    key="new_course"
                )

            with col2:

                units_input = st.number_input(
                    "Units",
                    min_value=0,
                    max_value=30,
                    value=3,
                    step=1,
                    key="new_units"
                )

                grade_input = st.selectbox(
                    "Grade",
                    [
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F"
                    ],
                    key="new_grade"
                )

            add_course = st.form_submit_button(
                "Add Course",
                type="primary",
                use_container_width=True
            )

        if add_course:

            course_name = course_input.strip()

            if course_name == "":
                st.error(
                    "Please enter a course name."
                )
                st.stop()

            if semester_input <= 0:

                st.error(
                    "Semester must be greater than 0."
                )
                st.stop()

            if units_input <= 0:

                st.error(
                    "Units must be greater than 0."
                )
                st.stop()

            duplicate_exists = False

            for existing_course in (
                st.session_state.manual_entries
            ):

                existing_semester = int(
                    existing_course["semester"]
                )

                existing_name = str(
                    existing_course["course"]
                ).strip().lower()

                if (
                    existing_semester
                    == int(semester_input)
                    and existing_name
                    == course_name.lower()
                ):

                    duplicate_exists = True
                    break

            if duplicate_exists:

                st.error(
                    f"{course_name} already exists "
                    f"in Semester {int(semester_input)}."
                )

                st.warning(
                    "The same course cannot be added "
                    "twice in the same semester."
                )

                st.stop()

            new_course = {
                "semester": int(semester_input),
                "course": course_name,
                "units": int(units_input),
                "grade": grade_input
            }

            st.session_state.manual_entries.append(
                new_course
            )

            st.session_state.active_data = pd.DataFrame(
                st.session_state.manual_entries
            )

            st.session_state.manual_saved = True

            st.success(
                f"{course_name} added successfully."
            )

            st.rerun()

    # -----------------------------------------------------
    # OTHER PAGES WITH MANUAL DATA
    # -----------------------------------------------------

    if st.session_state.get(
        "manual_saved",
        False
    ):

        data = st.session_state.active_data.copy()

    else:

        if page != "Enter Results":

            st.markdown(
                """
                <div class="content-card">
                    <div class="section-title">
                        No academic results yet
                    </div>
                    <div class="section-subtitle">
                        Go to <b>Enter Results</b> and add your
                        academic results before using this section.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.stop()


# =========================================================
# VALIDATE DATA
# =========================================================

required_columns = [
    "semester",
    "course",
    "units",
    "grade"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:

    st.error(
        "Your CSV is missing these columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "Your CSV should contain: "
        "semester, course, units, grade"
    )

    st.stop()


data = data[
    required_columns
].copy()


data["course"] = (
    data["course"]
    .fillna("")
    .astype(str)
    .str.strip()
)


data["grade"] = (
    data["grade"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.strip()
)


if data.empty:

    st.error("No results were found.")
    st.stop()


empty_courses = data["course"] == ""

if empty_courses.any():

    rows = list(
        data.index[empty_courses] + 1
    )

    st.error(
        f"Course name is missing in row(s): {rows}"
    )

    st.stop()


empty_grades = data["grade"] == ""

if empty_grades.any():

    rows = list(
        data.index[empty_grades] + 1
    )

    st.error(
        f"Grade is missing in row(s): {rows}"
    )

    st.stop()


data["semester"] = pd.to_numeric(
    data["semester"],
    errors="coerce"
)


invalid_semesters = data["semester"].isna()

if invalid_semesters.any():

    rows = list(
        data.index[invalid_semesters] + 1
    )

    st.error(
        f"Invalid or missing semester "
        f"in row(s): {rows}"
    )

    st.stop()


if (
    data["semester"] % 1 != 0
).any():

    rows = list(
        data.index[
            data["semester"] % 1 != 0
        ] + 1
    )

    st.error(
        "Semester must be a whole number. "
        f"Check row(s): {rows}"
    )

    st.stop()


if (
    data["semester"] <= 0
).any():

    rows = list(
        data.index[
            data["semester"] <= 0
        ] + 1
    )

    st.error(
        "Semester must be greater than 0. "
        f"Check row(s): {rows}"
    )

    st.stop()


data["units"] = pd.to_numeric(
    data["units"],
    errors="coerce"
)


invalid_units = data["units"].isna()

if invalid_units.any():

    rows = list(
        data.index[invalid_units] + 1
    )

    st.error(
        f"Invalid or missing units "
        f"in row(s): {rows}"
    )

    st.stop()


if (
    data["units"] <= 0
).any():

    rows = list(
        data.index[
            data["units"] <= 0
        ] + 1
    )

    st.error(
        "Units must be greater than 0. "
        f"Check row(s): {rows}"
    )

    st.stop()


grade_points = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
    "F": 0
}

valid_grades = list(
    grade_points.keys()
)


invalid_grades = data[
    ~data["grade"].isin(valid_grades)
]


if not invalid_grades.empty:

    invalid_values = ", ".join(
        invalid_grades["grade"].unique()
    )

    rows = list(
        invalid_grades.index + 1
    )

    st.error(
        f"Invalid grade(s): {invalid_values}"
    )

    st.info(
        "Valid grades are A, B, C, D, E and F."
    )

    st.warning(
        f"Please check row(s): {rows}"
    )

    st.stop()


duplicates = data[
    data.duplicated(
        subset=[
            "semester",
            "course"
        ],
        keep=False
    )
]


if not duplicates.empty:

    duplicate_list = duplicates[
        ["semester", "course"]
    ].drop_duplicates()

    duplicate_messages = []

    for _, row in duplicate_list.iterrows():

        duplicate_messages.append(
            f"{row['course']} "
            f"(Semester {int(row['semester'])})"
        )

    st.error(
        "Duplicate courses found."
    )

    st.warning(
        "The same course cannot appear twice "
        "in the same semester: "
        + ", ".join(duplicate_messages)
    )

    st.stop()


# =========================================================
# CALCULATIONS
# =========================================================

data["grade_point"] = data[
    "grade"
].map(grade_points)


data["quality_points"] = (
    data["units"]
    * data["grade_point"]
)


semester_summary = (
    data.groupby("semester")
    .agg(
        total_units=(
            "units",
            "sum"
        ),
        total_quality_points=(
            "quality_points",
            "sum"
        )
    )
    .reset_index()
)


semester_summary["GPA"] = (
    semester_summary[
        "total_quality_points"
    ]
    /
    semester_summary[
        "total_units"
    ]
)


semester_summary = (
    semester_summary
    .sort_values("semester")
)


total_units = data["units"].sum()

total_quality_points = (
    data["quality_points"].sum()
)

current_cgpa = (
    total_quality_points
    / total_units
)

number_of_semesters = (
    data["semester"].nunique()
)


latest_gpa = semester_summary.iloc[-1]["GPA"]


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="dashboard-banner">
            <h1>Academic Performance Dashboard</h1>
            <p>
                Monitor your GPA, CGPA and semester-by-semester
                academic performance in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card cgpa-card">
                <div class="metric-label">Current CGPA</div>
                <div class="cgpa-value">
                    {current_cgpa:.2f}
                </div>
                <div class="metric-description">
                    Overall academic performance
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Latest GPA
                </div>
                <div class="metric-value">
                    {latest_gpa:.2f}
                </div>
                <div class="metric-description">
                    Most recent semester
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Total Units
                </div>
                <div class="metric-value">
                    {int(total_units)}
                </div>
                <div class="metric-description">
                    Completed credit units
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Semesters
                </div>
                <div class="metric-value">
                    {number_of_semesters}
                </div>
                <div class="metric-description">
                    Academic semesters
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # SEMESTER PERFORMANCE
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Semester Performance
        </div>
        <div class="section-subtitle">
            GPA and credit-unit breakdown by semester.
        </div>
        """,
        unsafe_allow_html=True
    )

    display_table = semester_summary.copy()

    display_table["semester"] = (
        display_table["semester"]
        .astype(int)
    )

    display_table["GPA"] = (
        display_table["GPA"]
        .round(2)
    )

    display_table = display_table.rename(
        columns={
            "semester": "Semester",
            "total_units": "Units",
            "total_quality_points":
                "Quality Points",
            "GPA": "GPA"
        }
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # GPA TREND
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            GPA Trend
        </div>
        <div class="section-subtitle">
            Your GPA across completed semesters.
        </div>
        """,
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    ax.plot(
        semester_summary["semester"],
        semester_summary["GPA"],
        marker="o",
        linewidth=2.5
    )

    ax.set_xlabel("Semester")
    ax.set_ylabel("GPA")
    ax.set_ylim(0, 5)
    ax.grid(
        alpha=0.2,
        linestyle="--"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # INSIGHTS
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Performance Insights
        </div>
        <div class="section-subtitle">
            A quick summary of your academic progress.
        </div>
        """,
        unsafe_allow_html=True
    )

    best_semester = semester_summary.loc[
        semester_summary["GPA"].idxmax()
    ]

    first_gpa = semester_summary.iloc[0]["GPA"]

    change = latest_gpa - first_gpa

    i1, i2, i3 = st.columns(3)

    with i1:

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">
                    Highest Semester GPA
                </div>
                <div class="insight-text">
                    Semester {int(best_semester["semester"])}
                    recorded your highest GPA of
                    <b>{best_semester["GPA"]:.2f}</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:

        direction = (
            "increased"
            if change >= 0
            else "decreased"
        )

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">
                    Academic Trend
                </div>
                <div class="insight-text">
                    Your latest GPA has
                    <b>{direction}</b> by
                    <b>{abs(change):.2f}</b>
                    compared with your first semester.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">
                    Quality Points
                </div>
                <div class="insight-text">
                    You have accumulated
                    <b>{total_quality_points:.0f}</b>
                    quality points across
                    <b>{int(total_units)}</b> units.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# ENTER RESULTS
# =========================================================

elif page == "Enter Results":

    if data_source != "Enter Manually":

        st.markdown(
            """
            <div class="page-header">
                <div class="page-title">
                    Enter Results
                </div>
                <div class="page-subtitle">
                    Manual result entry is currently disabled.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "Select **Enter Manually** under Data Source "
            "in the sidebar to add or edit courses."
        )


# =========================================================
# WHAT-IF CALCULATOR
# =========================================================

elif page == "What-If Calculator":

    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">
                What-If Calculator
            </div>
            <div class="page-subtitle">
                See how a future semester could affect your CGPA.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">
                Simulate Your Next Semester
            </div>
            <div class="section-subtitle">
                Enter an expected GPA and number of credit units.
            </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        expected_gpa = st.number_input(
            "Expected GPA",
            min_value=0.0,
            max_value=5.0,
            value=4.5,
            step=0.1
        )

    with c2:

        expected_units = st.number_input(
            "Expected Units",
            min_value=1,
            max_value=30,
            value=15,
            step=1
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    new_quality_points = (
        expected_gpa
        * expected_units
    )

    projected_cgpa = (
        total_quality_points
        + new_quality_points
    ) / (
        total_units
        + expected_units
    )

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">
                Projected CGPA
            </div>
            <div class="prediction-value">
                {projected_cgpa:.2f}
            </div>
            <div style="color:#6b7280;font-size:13px;margin-top:5px;">
                Based on a GPA of {expected_gpa:.2f}
                across {int(expected_units)} units.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    difference = projected_cgpa - current_cgpa

    if difference > 0:

        st.success(
            f"Your CGPA would increase by "
            f"{difference:.2f} points."
        )

    elif difference < 0:

        st.warning(
            f"Your CGPA would decrease by "
            f"{abs(difference):.2f} points."
        )

    else:

        st.info(
            "Your CGPA would remain approximately the same."
        )


# =========================================================
# TARGET CGPA
# =========================================================

elif page == "Target CGPA":

    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">
                Target CGPA
            </div>
            <div class="page-subtitle">
                Calculate the GPA required to reach your target CGPA.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        target_cgpa = st.number_input(
            "Target CGPA",
            min_value=0.0,
            max_value=5.0,
            value=4.5,
            step=0.1
        )

    with c2:

        target_units = st.number_input(
            "Next Semester Units",
            min_value=1,
            max_value=30,
            value=15,
            step=1
        )

    required_gpa = (
        target_cgpa
        * (total_units + target_units)
        - total_quality_points
    ) / target_units

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    if required_gpa <= 0:

        st.success(
            f"You have already accumulated enough "
            f"quality points to reach a CGPA of "
            f"{target_cgpa:.2f} under this scenario."
        )

    elif required_gpa > 5:

        st.error(
            f"A GPA of {required_gpa:.2f} would be required, "
            "which is above the maximum GPA of 5.00."
        )

        st.info(
            "Consider spreading the target across multiple semesters."
        )

    else:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Required GPA
                </div>
                <div class="prediction-value">
                    {required_gpa:.2f}
                </div>
                <div style="color:#6b7280;font-size:13px;margin-top:5px;">
                    GPA required next semester to reach
                    a CGPA of {target_cgpa:.2f}.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        progress = (
            current_cgpa / target_cgpa
        ) if target_cgpa > 0 else 0

        progress = min(
            max(progress, 0),
            1
        )

        st.write("Current progress toward target")

        st.progress(progress)

        st.caption(
            f"Current CGPA: {current_cgpa:.2f} "
            f"of target {target_cgpa:.2f}"
        )


# =========================================================
# AI PREDICTION
# =========================================================

elif page == "AI Prediction":

    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">
                AI Performance Prediction
            </div>
            <div class="page-subtitle">
                A Linear Regression model estimates your next semester GPA
                using your previous semester performance.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(semester_summary) < 2:

        st.warning(
            "At least two semesters are required "
            "to generate a prediction."
        )

        st.stop()

    X = semester_summary[
        ["semester"]
    ]

    y = semester_summary[
        "GPA"
    ]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    next_semester = (
        int(
            semester_summary[
                "semester"
            ].max()
        )
        + 1
    )

    prediction_input = pd.DataFrame(
        {
            "semester": [
                next_semester
            ]
        }
    )

    predicted_gpa = model.predict(
        prediction_input
    )[0]

    predicted_gpa = float(
        np.clip(
            predicted_gpa,
            0,
            5
        )
    )

    expected_units = st.number_input(
        "Expected units next semester",
        min_value=1,
        max_value=30,
        value=15,
        step=1
    )

    predicted_future_cgpa = (
        total_quality_points
        +
        (
            predicted_gpa
            * expected_units
        )
    ) / (
        total_units
        + expected_units
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Predicted Next GPA
                </div>
                <div class="prediction-value">
                    {predicted_gpa:.2f}
                </div>
                <div style="color:#6b7280;font-size:13px;">
                    Estimated for Semester {next_semester}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Predicted Future CGPA
                </div>
                <div class="prediction-value">
                    {predicted_future_cgpa:.2f}
                </div>
                <div style="color:#6b7280;font-size:13px;">
                    Assuming {int(expected_units)} future units
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="section-title">
            GPA Prediction Trend
        </div>
        <div class="section-subtitle">
            Historical GPA compared with the model's next-semester prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

    chart_data = semester_summary[
        ["semester", "GPA"]
    ].copy()

    chart_data = pd.concat(
        [
            chart_data,
            pd.DataFrame(
                {
                    "semester": [
                        next_semester
                    ],
                    "GPA": [
                        predicted_gpa
                    ]
                }
            )
        ],
        ignore_index=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    ax.plot(
        chart_data["semester"],
        chart_data["GPA"],
        marker="o",
        linewidth=2.5
    )

    ax.axvline(
        next_semester,
        linestyle="--",
        alpha=0.5
    )

    ax.set_xlabel(
        "Semester"
    )

    ax.set_ylabel(
        "GPA"
    )

    ax.set_ylim(
        0,
        5
    )

    ax.grid(
        alpha=0.2,
        linestyle="--"
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.caption(
        "Note: This prediction is based only on the historical semester GPA trend. "
        "It is an estimate, not a guaranteed future result."
    )