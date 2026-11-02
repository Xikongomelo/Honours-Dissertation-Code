import joblib
import pandas as pd
import streamlit as st
from datetime import date, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Gauteng Festive-Season Fatality Predictor",
    page_icon="🚗",
    layout="wide",
)

# Resolve artifacts relative to this app.py file, regardless of launch folder.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "rf_fatality_model.pkl"
PREPROCESSOR_PATH = APP_DIR / "rf_preprocessor.pkl"
REFERENCE_PATH = APP_DIR / "gui_reference_data.pkl"
HISTORICAL_DATA_PATH = APP_DIR / "rtmc_historical_data.pkl"

# These are the eight predictors used by the application.
# Keep this order consistent with the original training pipeline.
FEATURES = [
    "DISTRICT",
    "CRASH_TYPE_DESC",
    "TOTAL_VEHICLE_INVOLVED_CLEAN",
    "ROAD_CATEGORY",
    "ACCIDENT_HOUR",
    "DAY_OF_WEEK",
    "FESTIVE_DAY",
    "FESTIVE_SEASON_YEAR",
]

# Based on the study design currently described for the project.
# Seasons 2020/21 and 2021/22 are intentionally excluded as COVID-affected.
# 2023/24 and 2024/25 are held-out evaluation seasons, not training seasons.
STUDY_SEASONS = {
    "2018/19": "Study season",
    "2019/20": "Study season",
    "2022/23": "Study season",
    "2023/24": "Held-out evaluation season",
    "2024/25": "Held-out evaluation season",
}

# Reported individual-crash model metrics supplied in the existing app.
# The exact split/season provenance must be verified against the final analysis notebook.
MODEL_METRICS = {
    "RMSE": 0.331339,
    "MAE": 0.130352,
    "R²": 0.082353,
    "MAPE (%)": 9.73869,
}

# ------------------------------------------------------------
# LOAD ARTIFACTS WITH CLEAR ERROR HANDLING
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    reference = joblib.load(REFERENCE_PATH)
    if not hasattr(preprocessor, "transform"):
        raise TypeError("The saved preprocessor does not provide transform().")
    if not hasattr(model, "predict"):
        raise TypeError("The saved model does not provide predict().")
    return preprocessor, model, reference


try:
    preprocessor, model, reference = load_artifacts()
except FileNotFoundError as exc:
    st.error(
        f"Required file not found: {exc.filename}. "
        "Place app.py, rf_preprocessor.pkl, rf_fatality_model.pkl, "
        "and gui_reference_data.pkl in the same folder."
    )
    st.stop()
except Exception as exc:
    st.error(
        "The saved model files could not be loaded. Check that the files "
        "are valid and were created with compatible Python/library versions."
    )
    st.exception(exc)
    st.stop()

if not isinstance(reference, pd.DataFrame):
    st.error("gui_reference_data.pkl must contain a pandas DataFrame.")
    st.stop()

required_reference_columns = {
    "DISTRICT",
    "POLICE_STATION",
    "ROAD_CATEGORY",
    "ROAD_NAME_NUMBER",
    "CRASH_TYPE_DESC",
}
missing_reference_columns = required_reference_columns - set(reference.columns)
if missing_reference_columns:
    st.error(
        "The reference dataset is missing required columns: "
        + ", ".join(sorted(missing_reference_columns))
    )
    st.stop()

# Check that the preprocessor exposes the expected feature schema when possible.
preprocessor_features = getattr(preprocessor, "feature_names_in_", None)
if preprocessor_features is not None:
    preprocessor_features = list(preprocessor_features)
    if preprocessor_features != FEATURES:
        st.error(
            "The saved preprocessor's input columns do not match the GUI's "
            "eight expected predictors. Do not make predictions until the "
            "GUI and training pipeline have been reconciled."
        )
        st.write("GUI expects:", FEATURES)
        st.write("Preprocessor expects:", preprocessor_features)
        st.stop()

# ------------------------------------------------------------
# STYLING
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* High-contrast, consistent styling across every tab and widget. */
    .stApp, [data-testid="stAppViewContainer"] { background: #f5f7fa !important; color: #172b4d !important; }
    /* Keep the page title safely below Streamlit's top toolbar; the previous 1.5rem gap was too small. */
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 3rem !important;
        width: 100%;
        max-width: 100%;
    }
    .main-title {
        display: block;
        position: relative;
        margin-top: 0.25rem;
        padding-top: 0.25rem;
        font-size: clamp(1.45rem, 3vw, 2.1rem);
        font-weight: 800;
        color: #12355b !important;
        line-height: 1.25;
        margin-bottom: .45rem;
        white-space: normal;
        overflow-wrap: break-word;
        word-break: normal;
        max-width: 100%;
    }
    
    .subtitle { font-size: 1rem; color: #40536b !important; margin-bottom: 1rem; }
    .section-title { color: #12355b !important; font-size: 1.3rem; font-weight: 750; }
    /* Streamlit markdown, captions, labels and help text were low-contrast in the screenshot. */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    label, .stCaption, small { color: #26384d !important; }
    [data-testid="stWidgetLabel"] p { font-weight: 650 !important; }
    /* Input controls: light background with dark, readable values. */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div, [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input, [data-testid="stNumberInput"] input {
        background-color: #ffffff !important; color: #172b4d !important;
        border-color: #8796a8 !important;
    }
    div[data-baseweb="select"] input, div[data-baseweb="input"] input,
    [data-testid="stDateInput"] input, [data-testid="stTimeInput"] input,
    [data-testid="stNumberInput"] input { color: #172b4d !important; -webkit-text-fill-color: #172b4d !important; }
    div[data-baseweb="select"] svg { fill: #26384d !important; }
    [data-baseweb="popover"], [role="listbox"], [role="option"] { background: #ffffff !important; color: #172b4d !important; }
    [role="option"] * { color: #172b4d !important; }
    [data-testid="stMetricValue"] { color: #8a6200 !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #12355b !important; font-weight: 650; }
    [data-testid="stAlert"] { color: #172b4d !important; }
    [data-testid="stTabs"] button { color: #26384d !important; font-weight: 650 !important; }
    /* Keep primary action labels visible even if global markdown/text styles change. */
    div.stButton > button[kind="primary"],
    div.stButton > button[kind="primary"] *,
    div.stButton > button[data-testid="baseButton-primary"],
    div.stButton > button[data-testid="baseButton-primary"] * {
        background-color: #12355b !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
        border-color: #12355b !important;
        font-weight: 750 !important;
    }
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] { border-radius: 8px !important; min-height: 2.75rem !important; }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover { background-color: #0b2744 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def festive_season_details(selected_date):
    """Return (season_label, festive_day) for 1 Dec–11 Jan; otherwise (None, None)."""
    if selected_date.month == 12:
        season = f"{selected_date.year}/{str(selected_date.year + 1)[-2:]}"
        festive_day = selected_date.day
        return season, festive_day

    if selected_date.month == 1 and selected_date.day <= 11:
        season = f"{selected_date.year - 1}/{str(selected_date.year)[-2:]}"
        festive_day = 31 + selected_date.day
        return season, festive_day

    return None, None


def build_model_input(
    district_value,
    crash_type_value,
    vehicles_value,
    road_category_value,
    accident_hour_value,
    weekday_value,
    festive_day_value,
    season_value,
):
    # Column names and ordering must match the model training pipeline.
    return pd.DataFrame(
        [{
            "DISTRICT": district_value,
            "CRASH_TYPE_DESC": crash_type_value,
            "TOTAL_VEHICLE_INVOLVED_CLEAN": int(vehicles_value),
            "ROAD_CATEGORY": road_category_value,
            "ACCIDENT_HOUR": int(accident_hour_value),
            "DAY_OF_WEEK": weekday_value,
            "FESTIVE_DAY": int(festive_day_value),
            "FESTIVE_SEASON_YEAR": season_value,
        }],
        columns=FEATURES,
    )


def predict_one(input_frame):
    if list(input_frame.columns) != FEATURES:
        raise ValueError(
            f"Input feature order mismatch. Expected {FEATURES}; got {list(input_frame.columns)}"
        )
    transformed = preprocessor.transform(input_frame)
    raw_prediction = model.predict(transformed)
    if len(raw_prediction) != 1:
        raise ValueError("Expected exactly one prediction for one scenario.")
    prediction = float(raw_prediction[0])
    if not pd.notna(prediction):
        raise ValueError("The model returned a non-finite prediction.")
    return prediction


def make_scenario_summary(
    selected_date, selected_time, season, festive_day, district_value,
    station_value, category_value, road_value, crash_type_value, vehicles_value
):
    return pd.DataFrame({
        "Characteristic": [
            "Crash date", "Crash time", "Day of week", "Festive season",
            "Day of festive season", "District", "Reporting police station",
            "Road category", "Road name/number", "Vehicles involved",
        ],
        "Selected value": [
            selected_date.strftime("%d %B %Y"),
            selected_time.strftime("%H:%M"),
            selected_date.strftime("%A"),
            season,
            festive_day,
            district_value,
            station_value,
            category_value,
            road_value,
            int(vehicles_value),
        ],
    })


# ------------------------------------------------------------
# AREA-ANALYSIS HELPERS
# ------------------------------------------------------------
@st.cache_data
def load_historical_data(path_string):
    """Load the full RTMC dataset separately from the model reference data."""
    return pd.read_pickle(path_string)


def prepare_area_data(source):
    """Prepare historical RTMC records for descriptive area statistics."""
    data = source.copy()
    if "ACCIDENT_DATE" not in data.columns:
        return None, "The reference data does not include ACCIDENT_DATE."

    data["_ACCIDENT_DATE_PARSED"] = pd.to_datetime(data["ACCIDENT_DATE"], errors="coerce", dayfirst=True)
    data = data[data["_ACCIDENT_DATE_PARSED"].notna()].copy()
    data["_SEASON"] = data["_ACCIDENT_DATE_PARSED"].apply(
        lambda d: (f"{d.year}/{str(d.year + 1)[-2:]}" if d.month == 12 else
                   (f"{d.year - 1}/{str(d.year)[-2:]}" if d.month == 1 and d.day <= 11 else None))
    )
    data = data[data["_SEASON"].notna()].copy()

    # Prefer the already-audited target column; subtype fields have substantial missingness.
    if "TOTAL_FATALITIES" in data.columns:
        data["_FATALITIES"] = pd.to_numeric(data["TOTAL_FATALITIES"], errors="coerce").fillna(0)
    else:
        fatality_fields = [c for c in ["PASSENGER", "PEDESTRIAN", "DRIVER", "CYCLIST", "UNKNOWN"] if c in data.columns]
        if fatality_fields:
            for col in fatality_fields:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
            data["_FATALITIES"] = data[fatality_fields].sum(axis=1)
        else:
            return None, "The historical dataset does not contain the fatality fields needed to calculate totals."
    return data, None


# ------------------------------------------------------------
# HEADER AND SCOPE
# ------------------------------------------------------------
st.markdown(
    '<div class="main-title">Gauteng Festive-Season Road Safety Planning Tool</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">RTMC research prototype · Individual crash scenario estimate + dynamic province, district, police-station and road analysis</div>',
    unsafe_allow_html=True,
)

st.info(
    "Purpose: help RTMC staff review the model's estimated fatality count for a specified crash scenario "
    "during the study's festive-season period (1 December–11 January). "
    "The Crash Scenario tab estimates fatalities for a hypothetical individual crash using the selected historical-period date; it is not a forecast of future crash occurrence. Road-level evaluation tests a retrospective estimate of aggregate fatalities on a selected road. These are separate targets and must be evaluated separately."
)

with st.expander("Important interpretation and limitations", expanded=False):
    st.markdown(
        """
        - The output is a **regression estimate**, not a probability and not a guaranteed count.
        - A decimal estimate is possible because the model predicts a numerical value; it does not mean a fraction of a person will die.
        - The reported test-set R² is low, so the model explains only a limited share of variation in held-out fatality counts.
        - Road name and police station are displayed for context but are **not predictors** in this model. Changing them alone will not change the prediction.
        - This is a dissertation demonstration, not a validated live RTMC decision-support system. Do not use it alone for enforcement, resource allocation, or public safety decisions.
        """
    )

tab_predict, tab_compare, tab_area, tab_performance, tab_intended = st.tabs([
    "Crash Scenario",
    "Compare Scenarios",
    "Road Forecast Evaluation",
    "Model Performance",
    "Intended Use",
])

# ------------------------------------------------------------
# SHARED INPUT FORM
# ------------------------------------------------------------
district_options = sorted(reference["DISTRICT"].dropna().astype(str).unique().tolist())
crash_type_options = sorted(reference["CRASH_TYPE_DESC"].dropna().astype(str).unique().tolist())

if not district_options or not crash_type_options:
    st.error("The reference data does not contain usable district or crash-type options.")
    st.stop()

# Invisible model input, as agreed: use the most frequent observed crash type
# from the reference data. The staff-facing GUI does not ask the user to choose it.
_crash_type_counts = reference["CRASH_TYPE_DESC"].dropna().astype(str).value_counts()
if _crash_type_counts.empty:
    st.error("No crash-type value is available for the model's hidden input.")
    st.stop()
HIDDEN_CRASH_TYPE = str(_crash_type_counts.index[0])

# Date range covers the study years, but the explicit season validation below
# prevents non-festive dates and unsupported seasons from being used.
with tab_predict:
    st.markdown('<div class="section-title">Estimate an individual crash scenario</div>', unsafe_allow_html=True)
    st.caption("This is a conditional estimate for a hypothetical crash on a date within the observed study period (2018–2025), not a prediction of a future crash. Road name and police station are displayed for context but are not predictors in this individual-crash model.")

    c1, c2 = st.columns(2)
    with c1:
        accident_date = st.date_input(
            "Scenario date (historical study period)",
            value=date(2024, 12, 20),
            min_value=date(2018, 1, 1),
            max_value=date(2025, 12, 31),
            help="Choose a date within the dataset period and the festive-season window (1 December–11 January). This date contextualises a hypothetical crash; it does not make the result a future forecast.",
            key="predict_date",
        )
        accident_time = st.time_input("Crash time", value=time(20, 0), key="predict_time")
        district = st.selectbox("District / municipality", district_options, key="predict_district")

    with c2:
        station_frame = reference[
            reference["DISTRICT"].astype(str) == str(district)
        ]
        station_options = sorted(station_frame["POLICE_STATION"].dropna().astype(str).unique().tolist())
        if not station_options:
            station_options = ["Not available in reference data"]
        police_station = st.selectbox(
            "Police station / local area (context only)",
            station_options,
            key="predict_station",
        )

        category_frame = station_frame
        category_options = sorted(category_frame["ROAD_CATEGORY"].dropna().astype(str).unique().tolist())
        if not category_options:
            category_options = sorted(reference["ROAD_CATEGORY"].dropna().astype(str).unique().tolist())
        if not category_options:
            st.error("No road-category options are available in the reference data.")
            st.stop()
        road_category = st.selectbox("Road category", category_options, key="predict_category")

    road_frame = station_frame[
        station_frame["ROAD_CATEGORY"].astype(str) == str(road_category)
    ]
    road_options = sorted(road_frame["ROAD_NAME_NUMBER"].dropna().astype(str).unique().tolist())
    if not road_options:
        road_options = sorted(reference["ROAD_NAME_NUMBER"].dropna().astype(str).unique().tolist())
    if not road_options:
        road_options = ["Not available in reference data"]
    road = st.selectbox("Road name or number (context only)", road_options, key="predict_road")

    # CRASH_TYPE_DESC is supplied invisibly to preserve the saved model's input schema.
    crash_type = HIDDEN_CRASH_TYPE
    c4 = st.container()
    with c4:
        # Reference data defines the observed range; it does not imply all counts are equally common.
        max_vehicles = 13
        if "TOTAL_VEHICLE_INVOLVED_CLEAN" in reference.columns:
            numeric_vehicles = pd.to_numeric(reference["TOTAL_VEHICLE_INVOLVED_CLEAN"], errors="coerce").dropna()
            if not numeric_vehicles.empty:
                max_vehicles = max(1, int(numeric_vehicles.max()))
        vehicles = st.number_input(
            "Vehicles involved (scenario assumption)",
            min_value=1,
            max_value=max_vehicles,
            value=min(2, max_vehicles),
            step=1,
            help=f"Allowed range is 1–{max_vehicles}, based on the cleaned reference data.",
            key="predict_vehicles",
        )

    season, festive_day = festive_season_details(accident_date)
    day_of_week = accident_date.strftime("%A")
    accident_hour = accident_time.hour

    st.markdown('<div class="section-title">Automatically calculated details</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.metric("Day of week", day_of_week)
    d2.metric("Festive season", season if season else "Not festive")
    d3.metric("Day of festive season", festive_day if festive_day is not None else "—")

    if season is None:
        st.error("This date is outside the festive-season window. Select 1 December–11 January.")
    elif season not in STUDY_SEASONS:
        st.warning(
            f"{season} is outside the supported seasons configured for this prototype. "
            "No prediction will be generated for this date."
        )
    elif STUDY_SEASONS[season] == "Held-out evaluation season":
        st.warning(
            f"{season} was used as a held-out evaluation season in the reported analysis. "
            "A prediction here is a scenario demonstration, not independent validation of a new season."
        )

    predict_clicked = st.button("Estimate fatalities for this scenario", type="primary", width="stretch")

    if predict_clicked:
        if season is None:
            st.error("Prediction stopped: the selected date is not within 1 December–11 January.")
        elif season not in STUDY_SEASONS:
            st.error("Prediction stopped: the selected season is outside the supported study scope.")
        elif not (1 <= int(vehicles) <= max_vehicles):
            st.error(f"Prediction stopped: vehicle count must be between 1 and {max_vehicles}.")
        else:
            try:
                model_input = build_model_input(
                    district, crash_type, vehicles, road_category, accident_hour,
                    day_of_week, festive_day, season,
                )
                prediction = predict_one(model_input)
                if not pd.notna(prediction):
                    raise ValueError("The model returned a non-finite prediction.")

                st.markdown("---")
                st.subheader("Estimated outcome for the selected scenario")
                rounded_estimate = max(0, int(prediction + 0.5))
                result_col1, result_col2 = st.columns(2)
                result_col1.metric("Plain-language estimate (rounded)", f"About {rounded_estimate}")
                result_col2.metric("Exact model estimate", f"{prediction:.2f}")
                st.caption(
                    "The rounded figure is a simpler presentation of the model output, not a guaranteed count or a prediction interval. "
                    "The decimal is the Random Forest's numerical estimate for the supplied crash characteristics; it is not a probability."
                )

                with st.expander("Scenario summary", expanded=True):
                    summary = make_scenario_summary(
                        accident_date, accident_time, season, festive_day, district,
                        police_station, road_category, road, crash_type, vehicles,
                    )
                    st.dataframe(summary, width="stretch", hide_index=True)

                st.warning(
                    "The model uses district, an internally supplied hidden crash-type value, vehicle count, road category, hour, day of week, "
                    "festive day and festive-season label. Police station and road name document the area "
                    "being reviewed but do not affect this prediction. Because crash type and vehicle count "
                    "are inputs, this is a conditional scenario estimate—not a forecast of which crash will occur."
                )
            except Exception as exc:
                st.error(
                    "The prediction could not be generated. This may indicate a mismatch between the GUI "
                    "inputs and the saved preprocessing/model pipeline. Do not interpret a failed prediction "
                    "as a model result."
                )
                st.exception(exc)

# ------------------------------------------------------------
# SCENARIO COMPARISON
# ------------------------------------------------------------
with tab_compare:
    st.markdown('<div class="section-title">Compare two hypothetical crash scenarios</div>', unsafe_allow_html=True)
    st.write(
        "This comparison shows how the trained model responds to two sets of input characteristics. "
        "It does not establish causation or prove that changing one factor would change real-world fatalities."
    )

    default_season = "2024/25"
    default_date = date(2024, 12, 20)

    base1, base2 = st.columns(2)
    with base1:
        compare_date = st.date_input(
            "Scenario date (historical study period)",
            value=default_date,
            min_value=date(2018, 1, 1),
            max_value=date(2025, 12, 31),
            key="compare_date",
        )
        compare_time_a = st.time_input("Scenario A time", value=time(18, 0), key="compare_time_a")
        compare_district = st.selectbox("District", district_options, key="compare_district")
        compare_crash = HIDDEN_CRASH_TYPE

    with base2:
        compare_category_options = sorted(reference["ROAD_CATEGORY"].dropna().astype(str).unique().tolist())
        if not compare_category_options:
            st.error("No road-category options are available.")
            st.stop()
        compare_category = st.selectbox("Road category", compare_category_options, key="compare_category")
        observed_vehicle_max = 13
        if "TOTAL_VEHICLE_INVOLVED_CLEAN" in reference.columns:
            vehicle_values = pd.to_numeric(
                reference["TOTAL_VEHICLE_INVOLVED_CLEAN"], errors="coerce"
            ).dropna()
            if not vehicle_values.empty:
                observed_vehicle_max = max(1, int(vehicle_values.max()))
        compare_vehicles_a = st.number_input(
            "Scenario A vehicles", min_value=1, max_value=observed_vehicle_max,
            value=min(2, observed_vehicle_max), step=1, key="compare_vehicles_a"
        )
        compare_vehicles_b = st.number_input(
            "Scenario B vehicles", min_value=1, max_value=observed_vehicle_max,
            value=min(3, observed_vehicle_max), step=1, key="compare_vehicles_b"
        )
        compare_time_b = st.time_input("Scenario B time", value=time(20, 0), key="compare_time_b")

    compare_season, compare_festive_day = festive_season_details(compare_date)
    if compare_season is None or compare_season not in STUDY_SEASONS:
        st.warning("Choose a date within a supported festive season to compare scenarios.")
    elif st.button("Compare scenario estimates", type="primary", width="stretch"):
        try:
            weekday = compare_date.strftime("%A")
            a_input = build_model_input(
                compare_district, compare_crash, compare_vehicles_a, compare_category,
                compare_time_a.hour, weekday, compare_festive_day, compare_season,
            )
            b_input = build_model_input(
                compare_district, compare_crash, compare_vehicles_b, compare_category,
                compare_time_b.hour, weekday, compare_festive_day, compare_season,
            )
            pred_a = predict_one(a_input)
            pred_b = predict_one(b_input)
            if not (pd.notna(pred_a) and pd.notna(pred_b)):
                raise ValueError("The model returned a non-finite prediction.")
            comp1, comp2, comp3 = st.columns(3)
            comp1.metric("Scenario A estimate (rounded)", f"About {max(0, int(pred_a + 0.5))}", help=f"Exact model estimate: {pred_a:.2f}")
            comp2.metric("Scenario B estimate (rounded)", f"About {max(0, int(pred_b + 0.5))}", help=f"Exact model estimate: {pred_b:.2f}")
            comp3.metric("Exact difference (B − A)", f"{pred_b - pred_a:+.2f}")
            st.caption(
                "Only the specified scenario inputs differ. The difference is a model response, "
                "not evidence that the changed input causes a real-world change in fatalities."
            )
        except Exception as exc:
            st.error("Comparison failed because the saved model and GUI inputs may not be compatible.")
            st.exception(exc)

# ------------------------------------------------------------
# LOCATION-LEVEL RETROSPECTIVE FORECAST EVALUATION
# ------------------------------------------------------------
with tab_area:
    st.markdown('<div class="section-title">Festive-season location forecast evaluation</div>', unsafe_allow_html=True)
    st.write(
        "Evaluate historical festive-season fatality totals at province, district, police-station, or road level. "
        "The selected season is held out: the linear-trend baseline is fitted only on earlier eligible seasons, "
        "then compared with the observed total. This is separate from the individual-crash estimator."
    )
    st.caption(
        "These are retrospective test predictions, not current operational forecasts. "
        "Police stations are distinguished by district; roads are distinguished by district and police station."
    )
    st.markdown('<div class="section-title">Dynamic RTF totals</div>', unsafe_allow_html=True)
    st.write(
        "Choose a festive season and geographic level to calculate recorded road-traffic-fatality (RTF) totals. "
        "The summary updates when the selections change."
    )

    if not HISTORICAL_DATA_PATH.is_file():
        st.warning(f"Forecast data file not found: {HISTORICAL_DATA_PATH.name}")
        st.info("Place rtmc_historical_data.pkl in the same folder as app.py. This is the full RTMC dataset saved from your notebook.")
    else:
        try:
            historical_source = load_historical_data(str(HISTORICAL_DATA_PATH))
            area_data, area_error = prepare_area_data(historical_source)
        except Exception:
            area_data, area_error = None, "Could not read rtmc_historical_data.pkl. Re-save it from the notebook and restart the app."
            st.error(area_error)

        if area_error:
            st.warning(area_error)
        else:
            summary_seasons = sorted(area_data["_SEASON"].dropna().astype(str).unique(), key=lambda value: int(value[:4]))
            preferred_season = "2024/25" if "2024/25" in summary_seasons else summary_seasons[-1]
            summary_cols = st.columns([1, 1])
            with summary_cols[0]:
                summary_season = st.selectbox(
                    "Festive season for RTF totals", summary_seasons,
                    index=summary_seasons.index(preferred_season), key="dynamic_rtf_season"
                )
            with summary_cols[1]:
                summary_level = st.selectbox(
                    "Show totals by", ["Province", "District", "Police station", "Road"],
                    index=1, key="dynamic_rtf_level"
                )

            selected_summary = area_data[area_data["_SEASON"].astype(str).eq(summary_season)].copy()
            selected_summary["_FATALITIES"] = pd.to_numeric(selected_summary["_FATALITIES"], errors="coerce").fillna(0)
            province_total = float(selected_summary["_FATALITIES"].sum())
            summary_groupings = {
                "Province": [],
                "District": ["DISTRICT"],
                "Police station": ["DISTRICT", "POLICE_STATION"],
                "Road": ["DISTRICT", "POLICE_STATION", "ROAD_NAME_NUMBER"],
            }
            summary_groups = summary_groupings[summary_level]
            if not summary_groups:
                summary_table = pd.DataFrame([{
                    "Province / dataset scope": "Gauteng", "Total RTFs": province_total,
                    "Crash records": len(selected_summary)
                }])
            elif not set(summary_groups).issubset(selected_summary.columns):
                summary_table = pd.DataFrame()
                st.warning("The dataset does not contain the location columns required for this summary level.")
            else:
                summary_clean = selected_summary.copy()
                for col in summary_groups:
                    summary_clean[col] = summary_clean[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
                summary_table = (
                    summary_clean.groupby(summary_groups, dropna=False)
                    .agg(**{"Total RTFs": ("_FATALITIES", "sum"), "Crash records": ("_FATALITIES", "size")})
                    .reset_index()
                    .rename(columns={"DISTRICT": "District", "POLICE_STATION": "Police station", "ROAD_NAME_NUMBER": "Road"})
                    .sort_values("Total RTFs", ascending=False)
                )
            c1, c2, c3 = st.columns(3)
            c1.metric("Gauteng RTFs", f"{province_total:,.0f}")
            c2.metric("Crash records", f"{len(selected_summary):,}")
            c3.metric("Districts represented", f"{selected_summary['DISTRICT'].nunique():,}" if "DISTRICT" in selected_summary.columns else "N/A")
            if not summary_table.empty:
                st.dataframe(summary_table, use_container_width=True, hide_index=True)
                st.caption(f"Totals reflect records labelled {summary_season}. The 2025/26 season is incomplete in the supplied dataset; interpret it as partial if selected.")

            st.divider()
            st.markdown('**Retrospective forecast evaluation**')
            area_data = area_data.copy()
            area_data["_PROVINCE"] = "Gauteng"
            location_levels = {
                "Province": ["_PROVINCE"],
                "District": ["DISTRICT"],
                "Police station": ["DISTRICT", "POLICE_STATION"],
                "Road": ["DISTRICT", "POLICE_STATION", "ROAD_NAME_NUMBER"],
            }
            level_col1, level_col2 = st.columns([1, 1])
            with level_col1:
                analysis_level = st.selectbox(
                    "Analysis level", list(location_levels.keys()), index=3, key="location_analysis_level",
                    help="Choose the geographic detail for the forecast evaluation."
                )
            with level_col2:
                holdout_season = st.selectbox(
                    "Completed festive season to evaluate", ["2023/24", "2024/25"], index=1,
                    key="location_holdout_season",
                    help="The selected season is held out. Only earlier eligible seasons are used to fit the trend baseline."
                )

            group_cols = location_levels[analysis_level]
            missing_location_cols = [c for c in group_cols if c not in area_data.columns]
            if missing_location_cols:
                st.warning("The historical dataset is missing required location field(s): " + ", ".join(missing_location_cols))
            else:
                # Explicit, consistent season policy. COVID-affected seasons and incomplete/legacy seasons are excluded.
                excluded_seasons = {"2017/18", "2020/21", "2021/22", "2025/26"}
                usable = area_data[~area_data["_SEASON"].astype(str).isin(excluded_seasons)].copy()
                usable["_SEASON_START_YEAR"] = usable["_SEASON"].astype(str).str.slice(0, 4).astype(int)
                for col in group_cols:
                    usable[col] = usable[col].fillna("").astype(str).str.strip()
                usable = usable[~usable[group_cols].eq("").any(axis=1)].copy()

                # Keep only eligible seasons strictly before the holdout, plus the holdout itself.
                holdout_start_year = int(holdout_season[:4])
                prior_seasons = sorted(
                    [s for s in usable["_SEASON"].unique() if int(str(s)[:4]) < holdout_start_year],
                    key=lambda value: int(str(value)[:4])
                )
                train_seasons = [s for s in prior_seasons if s not in excluded_seasons]
                actual_data = usable[usable["_SEASON"].astype(str).eq(holdout_season)].copy()
                if not train_seasons:
                    st.warning("There are no eligible training seasons before this holdout season.")
                elif actual_data.empty:
                    st.warning(f"No records were found for the selected holdout season ({holdout_season}).")
                else:
                    # Aggregate recorded fatalities for each location-season.
                    seasonal_totals = (
                        usable[usable["_SEASON"].astype(str).isin(train_seasons + [holdout_season])]
                        .groupby(group_cols + ["_SEASON", "_SEASON_START_YEAR"], as_index=False)
                        .agg(Fatalities=("_FATALITIES", "sum"), Crash_records=("_FATALITIES", "size"))
                    )
                    train_observed = seasonal_totals[seasonal_totals["_SEASON"].isin(train_seasons)].copy()
                    actual_observed = seasonal_totals[seasonal_totals["_SEASON"].eq(holdout_season)].copy()

                    # Define the location universe from training data only to avoid using holdout-only locations.
                    location_universe = train_observed[group_cols].drop_duplicates().copy()
                    if location_universe.empty:
                        st.warning("No locations were observed in the eligible training seasons.")
                    else:
                        # Build a complete panel for the known training locations. A missing location-season
                        # record is treated as zero recorded fatalities for that location-season.
                        # The note below flags that inconsistent location naming can affect this assumption.
                        season_frame = pd.DataFrame({"_SEASON": train_seasons + [holdout_season]})
                        season_frame["_SEASON_START_YEAR"] = season_frame["_SEASON"].astype(str).str.slice(0, 4).astype(int)
                        location_universe["_join_key"] = 1
                        season_frame["_join_key"] = 1
                        panel = location_universe.merge(season_frame, on="_join_key").drop(columns="_join_key")
                        panel = panel.merge(
                            seasonal_totals,
                            on=group_cols + ["_SEASON", "_SEASON_START_YEAR"], how="left",
                            suffixes=("", "_observed")
                        )
                        panel["Fatalities"] = pd.to_numeric(panel["Fatalities"], errors="coerce").fillna(0.0)
                        panel["Crash_records"] = pd.to_numeric(panel["Crash_records"], errors="coerce").fillna(0).astype(int)
                        panel["_LOCATION_KEY"] = panel[group_cols].astype(str).agg("|||".join, axis=1)

                        train_panel = panel[panel["_SEASON"].isin(train_seasons)].copy()
                        holdout_panel = panel[panel["_SEASON"].eq(holdout_season)].copy()
                        # Require at least three eligible prior seasonal points; these include zero-record seasons.
                        n_train_seasons = train_panel.groupby("_LOCATION_KEY")["_SEASON"].nunique()
                        eligible_keys = set(n_train_seasons[n_train_seasons >= 3].index)
                        train_panel = train_panel[train_panel["_LOCATION_KEY"].isin(eligible_keys)].copy()
                        holdout_panel = holdout_panel[holdout_panel["_LOCATION_KEY"].isin(eligible_keys)].copy()

                        def location_label(key):
                            values = key.split("|||")
                            return " | ".join(f"{col.replace('_', ' ').title()}: {value}" for col, value in zip(group_cols, values))

                        eligible_keys_sorted = sorted(eligible_keys)
                        if not eligible_keys_sorted:
                            st.warning("No locations have at least three eligible prior seasons for this holdout evaluation.")
                        else:
                            labels = {location_label(key): key for key in eligible_keys_sorted}
                            selected_label = st.selectbox(
                                f"Select {analysis_level.lower()}", list(labels.keys()), key="selected_forecast_location"
                            )
                            selected_key = labels[selected_label]
                            selected_train = train_panel[train_panel["_LOCATION_KEY"].eq(selected_key)].sort_values("_SEASON_START_YEAR")
                            selected_actual = holdout_panel[holdout_panel["_LOCATION_KEY"].eq(selected_key)]
                            if selected_train.empty or selected_actual.empty:
                                st.warning("The selected location does not have enough prior seasonal data for this holdout evaluation.")
                            else:
                                trend_model = LinearRegression().fit(
                                    selected_train[["_SEASON_START_YEAR"]].to_numpy(dtype=float),
                                    selected_train["Fatalities"].to_numpy(dtype=float)
                                )
                                predicted = max(0.0, float(trend_model.predict(np.array([[holdout_start_year]], dtype=float))[0]))
                                actual = float(selected_actual.iloc[0]["Fatalities"])
                                error = predicted - actual
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Predicted fatalities", f"{predicted:,.1f}")
                                c2.metric("Observed fatalities", f"{actual:,.0f}")
                                c3.metric("Prediction error", f"{error:+,.1f}", help="Predicted minus observed; positive means overprediction.")
                                location_train_seasons = selected_train["_SEASON"].astype(str).tolist()
                                st.caption(f"Holdout season: {holdout_season}. Training seasons actually used for this location: {', '.join(location_train_seasons)}.")
                                history_view = selected_train[group_cols + ["_SEASON", "Crash_records", "Fatalities"]].rename(
                                    columns={"_SEASON": "Training season", "Crash_records": "Crash records", "Fatalities": "Observed fatalities"}
                                )
                                st.markdown("**Training history used for this location**")
                                st.dataframe(history_view, use_container_width=True, hide_index=True)

                            # Evaluate every eligible location in the training-defined universe, including
                            # zero observed holdout totals where no crash record exists for that location.
                            comparison_rows = []
                            for key in eligible_keys_sorted:
                                group = train_panel[train_panel["_LOCATION_KEY"].eq(key)].sort_values("_SEASON_START_YEAR")
                                observed = holdout_panel[holdout_panel["_LOCATION_KEY"].eq(key)]
                                if len(group) >= 3 and not observed.empty:
                                    model = LinearRegression().fit(
                                        group[["_SEASON_START_YEAR"]].to_numpy(dtype=float),
                                        group["Fatalities"].to_numpy(dtype=float)
                                    )
                                    estimate = max(0.0, float(model.predict(np.array([[holdout_start_year]], dtype=float))[0]))
                                    actual_value = float(observed.iloc[0]["Fatalities"])
                                    result_row = {col.replace("_", " ").title(): value for col, value in zip(group_cols, key.split("|||"))}
                                    # Preserve full precision for metrics; round only the display values.
                                    last_training_value = float(
                                        group.sort_values("_SEASON_START_YEAR").iloc[-1]["Fatalities"]
                                    )
                                    result_row.update({
                                        "Predicted fatalities": estimate,
                                        "Observed fatalities": actual_value,
                                        "Absolute error": abs(estimate - actual_value),
                                        "Last-season baseline prediction": last_training_value,
                                        "Baseline absolute error": abs(last_training_value - actual_value),
                                    })
                                    comparison_rows.append(result_row)
                            comparison = pd.DataFrame(comparison_rows)
                            if not comparison.empty:
                                # The detailed location-by-location comparison table is intentionally hidden
                                # to keep the interface uncluttered; aggregate metrics remain visible below.
                                mae = float(comparison["Absolute error"].mean())
                                rmse = float(np.sqrt(np.mean((comparison["Predicted fatalities"] - comparison["Observed fatalities"]) ** 2)))
                                baseline_mae = float(comparison["Baseline absolute error"].mean())
                                baseline_rmse = float(np.sqrt(np.mean((comparison["Last-season baseline prediction"] - comparison["Observed fatalities"]) ** 2)))
                                k1, k2, k3 = st.columns(3)
                                k1.metric(f"Trend {analysis_level} MAE", f"{mae:.2f}")
                                k2.metric(f"Trend {analysis_level} RMSE", f"{rmse:.2f}")
                                k3.metric("Locations evaluated", f"{len(comparison):,}")
                                b1, b2 = st.columns(2)
                                b1.metric("Last-season baseline MAE", f"{baseline_mae:.2f}", delta=f"{mae - baseline_mae:+.2f} vs baseline", delta_color="inverse")
                                b2.metric("Last-season baseline RMSE", f"{baseline_rmse:.2f}", delta=f"{rmse - baseline_rmse:+.2f} vs baseline", delta_color="inverse")
                                if mae > baseline_mae or rmse > baseline_rmse:
                                    st.info("The linear-trend approach does not outperform the simple last-season baseline on every metric. Report this honestly; a more complex model is not automatically better.")
                                else:
                                    st.success("The linear-trend approach performs better than the last-season baseline on both displayed error metrics for this holdout. This is evidence for this backtest only, not proof of future performance.")

                st.warning(
                    "This is an exploratory linear-trend backtest compared with a simple last-season baseline, not proof that either method can reliably forecast future seasons. "
                    "Locations absent from a season are treated as having zero recorded fatalities for that season; inconsistent naming or incomplete reporting can make that assumption imperfect. "
                    "The dataset lacks traffic-volume or road-exposure measures, so fatality totals are not per-vehicle risk."
                )
                st.caption(
                    "The training location list is created from prior seasons only. The holdout season is not used to fit the trend. "
                    "Seasons 2020/21 and 2021/22 are excluded as COVID-affected; 2017/18 and incomplete 2025/26 are excluded from evaluation."
                )

# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------
with tab_performance:
    st.markdown('<div class="section-title">Reported individual-crash model performance</div>', unsafe_allow_html=True)
    st.write(
        "These are the metric values currently recorded in this app. Confirm the test split and evaluation procedure against the final analysis notebook before reporting them as held-out performance. "
        "They are not a guarantee of accuracy for every crash or future festive season."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{MODEL_METRICS['RMSE']:.3f}")
    m2.metric("MAE", f"{MODEL_METRICS['MAE']:.3f}")
    m3.metric("R²", f"{MODEL_METRICS['R²']:.3f}")
    m4.metric("MAPE", f"{MODEL_METRICS['MAPE (%)']:.2f}%")

    st.markdown("**How to interpret these metrics**")
    st.markdown(
        """
        - **RMSE:** penalises larger prediction errors more heavily; lower is better.
        - **MAE:** the average absolute difference between predicted and observed fatality counts; lower is better.
        - **R²:** indicates the proportion of variation explained relative to a mean-baseline model. The reported value is low, so explanatory performance is limited.
        - **MAPE:** expresses errors as percentages and can be unstable when actual values are zero or very small. Interpret it alongside RMSE and MAE.
        """
    )
    st.warning(
        "The Random Forest was selected based on comparative model evaluation, but its low R² means "
        "the GUI should be treated as a research demonstration. Do not describe the model as highly accurate "
        "or operationally validated."
    )
    st.caption(
        "Before submission, verify that these figures exactly match the final Chapter 4 results and that "
        "the saved model/preprocessor are the same artifacts used to obtain them."
    )

# ------------------------------------------------------------
# INTENDED USE
# ------------------------------------------------------------
with tab_intended:
    st.markdown('<div class="section-title">What this tool can and cannot support</div>', unsafe_allow_html=True)
    use_table = pd.DataFrame({
        "Use": [
            "Demonstrate an individual crash-level model prediction",
            "Compare model estimates for two hypothetical scenarios",
            "Summarise recorded festive-season fatalities by district/road",
            "Support academic discussion of predictive modelling",
            "Forecast total fatalities for a specific road or police station",
            "Predict whether a crash will occur",
            "Allocate officers, ambulances or enforcement resources",
            "Replace professional judgement or official RTMC reporting",
        ],
        "Status": [
            "Within prototype scope",
            "Exploratory only; not causal evidence",
            "Descriptive only; depends on available data quality",
            "Within research scope",
            "Not established by this model",
            "Not supported by this model",
            "Not validated; not recommended",
            "Not supported",
        ],
    })
    st.dataframe(use_table, width="stretch", hide_index=True)
    st.write(
        "The prototype demonstrates an interface for applying the selected model to an individual crash scenario. "
        "It is not a deployed real-time system and has not been established as suitable for operational decision-making."
    )
