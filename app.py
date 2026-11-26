import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------- Page config ----------
st.set_page_config(
    page_title="Used Laptop Price Estimator",
    page_icon="💻",
    layout="centered",
)

# ---------- Load model (cached so it's only loaded once) ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "used_laptop_estimator.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_gpu_options():
    """Extract unique GPU values from the dataset at startup."""
    data_path = BASE_DIR / "dataset" / "final_ebay_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        return sorted(df["GPU"].dropna().astype(str).unique().tolist())
    # Fallback: minimal set if dataset is unavailable
    return [
        "Intel HD Graphics",
        "Intel UHD Graphics 620",
        "NVIDIA Quadro P3200",
        "AMD Radeon Graphics",
    ]


model = load_model()
gpu_options = load_gpu_options()

# ---------- Static option lists (must match training exactly) ----------
BRANDS = [
    "ASUS", "Acer", "Dell", "Fujitsu", "GPD", "Gateway",
    "HP", "Lenovo", "Microsoft", "Panasonic", "Samsung",
]

PROCESSORS = [
    "AMD 3015Ce", "AMD A4-9120C", "AMD A6-7310 APU", "AMD E-Series",
    "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 7 Pro",
    "AMD Ryzen Pro 3", "Intel Celeron", "Intel Core Duo", "Intel Core M3",
    "Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Pentium",
    "Microsoft SQ1", "Microsoft SQ2",
]

RESOLUTION_SIZES = ["Small", "Medium", "Large"]

OS_OPTIONS = [
    "Chrome OS", "Linux", "Not Included", "Pop OS", "Windows 10",
    "Windows 10 Home", "Windows 10 Pro", "Windows 10 Pro 64",
    "Windows 11", "Windows 11 Home", "Windows 11 Pro", "Windows 7",
    "Windows 7 Pro", "Windows XP",
]

STORAGE_TYPES = ["HDD", "M.2", "NVMe", "SSD", "SSD + HDD", "SSD + eMMC", "eMMC"]

CONDITIONS = ["Certified", "Excellent", "Good", "Open box", "Used", "Very Good"]

RAM_SIZES = [2, 4, 8, 16, 32, 64]

# ---------- Header ----------
st.title("💻 Used Laptop Price Estimator")
st.write(
    "Enter the specs of a used laptop to get an estimated resale price in USD."
)

# ---------- Form ----------
with st.form("estimator_form"):
    st.subheader("Laptop Specifications")

    col1, col2 = st.columns(2)

    with col1:
        screen_size = st.number_input(
            "Screen Size (in)",
            min_value=7.0,
            max_value=17.3,
            value=14.0,
            step=0.1,
            format="%.1f",
        )
        ram_size = st.selectbox(
            "Ram Size (GB)",
            options=RAM_SIZES,
            index=2,  # 8 GB default
        )
        ssd_capacity = st.number_input(
            "SSD Capacity (GB)",
            min_value=0,
            max_value=2000,
            value=256,
            step=32,
            help="0 = no SSD",
        )
        processor_speed = st.number_input(
            "Processor Speed (GHz)",
            min_value=1.0,
            max_value=5.1,
            value=2.4,
            step=0.1,
            format="%.1f",
        )
        hdd_capacity = st.number_input(
            "Hard Drive Capacity (GB)",
            min_value=0,
            max_value=4000,
            value=0,
            step=100,
            help="0 = no HDD",
        )
        brand = st.selectbox("Brand", options=BRANDS, index=2)  # Dell default

    with col2:
        processor = st.selectbox(
            "Processor", options=PROCESSORS, index=12
        )  # Intel Core i5 default
        gpu = st.selectbox(
            "GPU",
            options=gpu_options,
            index=0,
            help="Type to search/filter the list.",
        )
        resolution_size = st.selectbox(
            "Resolution Size", options=RESOLUTION_SIZES, index=1
        )  # Medium
        os_choice = st.selectbox(
            "OS", options=OS_OPTIONS, index=4
        )  # Windows 10
        storage_type = st.selectbox(
            "Storage Type", options=STORAGE_TYPES, index=3
        )  # SSD
        condition = st.selectbox(
            "Condition", options=CONDITIONS, index=4
        )  # Used

    submitted = st.form_submit_button("Estimate Price", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    # Basic validation
    errors = []
    if screen_size <= 0:
        errors.append("Screen size must be positive.")
    if processor_speed <= 0:
        errors.append("Processor speed must be positive.")
    if not all([brand, processor, gpu, resolution_size, os_choice,
                storage_type, condition]):
        errors.append("All dropdown fields must have a selection.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        input_df = pd.DataFrame([{
            "Screen Size (in)": float(screen_size),
            "Ram Size (GB)": float(ram_size),
            "SSD Capacity (GB)": float(ssd_capacity),
            "Processor Speed (GHz)": float(processor_speed),
            "Hard Drive Capacity (GB)": float(hdd_capacity),
            "Brand": brand,
            "Processor": processor,
            "GPU": gpu,
            "Resolution Size": resolution_size,
            "OS": os_choice,
            "Storage Type": storage_type,
            "Condition": condition,
        }])

        try:
            prediction = model.predict(input_df)[0]
            prediction = max(prediction, 0.0)  # clamp negative predictions to $0

            st.success("Estimate generated.")
            st.markdown(
                f"<h2 style='text-align:center; color:#2e7d32;'>"
                f"${prediction:,.2f}</h2>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Estimate only — based on a small historical dataset of eBay "
                "listings. Typical error is roughly ±$90–100 and may be less "
                "accurate for unusual configurations."
            )

            with st.expander("Model performance details"):
                st.markdown(
                    "- Trained on only **255** historical eBay listings.\n"
                    "- Held-out test metrics: **MAE ≈ $94**, **RMSE ≈ $157**, "
                    "**R² ≈ 0.73** (explains ~73% of price variation)."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Built with Streamlit · Model: scikit-learn Random Forest pipeline · "
    "For informational purposes only."
)