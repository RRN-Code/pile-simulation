import streamlit as st
import numpy as np
from simulation import PreHomogenizationPileSimulation
from utils import get_deposit_pattern_truck
from plotting import plot_final_pile, plot_grade_matrix

st.set_page_config(
    page_title="Pile Simulation",
    page_icon="assets/icon.png",
    layout="wide"
)

# -------------------------------------------
# Simple Authentication
# -------------------------------------------
def login():
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        with st.form("login_form"):
            st.subheader("🔒 Login Required")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                # Read credentials from secrets.toml
                stored_username = st.secrets["auth"]["username"]
                stored_password = st.secrets["auth"]["password"]

                if username == stored_username and password == stored_password:
                    st.session_state.logged_in = True
                    st.success("Login successful! ✅")
                    st.rerun()
                else:
                    st.error("Invalid username or password ❌")



# ----------------------------------------------------
# Streamlit App
# ----------------------------------------------------
def main():
    st.title("Pre-Homo Pile Simulation")
    st.sidebar.title("Simulation Parameters")

    # --- Sidebar inputs ---
    st.sidebar.markdown("### Basic Parameters")
    pile_length = st.sidebar.slider("Pile Length (m)", min_value=100.0, max_value=300.0, value=100.0, step=10.0)
    stacker_velocity = st.sidebar.slider("Stacker Velocity (m/min)", min_value=10.0, max_value=20.0, value=10.0, step=10.0)
    production_rate = st.sidebar.slider("Production Rate (tons/hour)", min_value=300.0, max_value=900.0, value=600.0, step=100.0)
    total_weight = st.sidebar.slider("Total Material (tons)", min_value=900.0, max_value=9000.0, value=900.0, step=900.0)
    truck_payload = st.sidebar.slider("Truck Payload (tons)", min_value=20.0, max_value=60.0, value=30.0, step=10.0)

    # --- Ore Source Grades and Variabilities ---
    st.sidebar.markdown("### Ore Source Zones")
    grade_option = st.sidebar.radio("Number of ore sources feeding the pile", ["2 Sources", "3 Sources"])

    if grade_option == "2 Sources":
        grade_a = st.sidebar.slider("Grade A (%)", min_value=1.0, max_value=50.0, value=30.0, step=1.0)
        var_a = st.sidebar.slider("Grade Variability A (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        grade_b = st.sidebar.slider("Grade B (%)", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
        var_b = st.sidebar.slider("Grade Variability B (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        ore_grades = {'A': grade_a, 'B': grade_b}
        variabilities = {'A': var_a, 'B': var_b}
        deposit_ratio_choice = st.sidebar.selectbox("Deposit Ratio", ["1:1", "2:1"])

    else:
        grade_a = st.sidebar.slider("Grade A (%)", min_value=1.0, max_value=50.0, value=30.0, step=1.0)
        var_a = st.sidebar.slider("Grade Variability A (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        grade_b = st.sidebar.slider("Grade B (%)", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
        var_b = st.sidebar.slider("Grade Variability B (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        grade_c = st.sidebar.slider("Grade C (%)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
        var_c = st.sidebar.slider("Grade Variability C (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

        ore_grades = {'A': grade_a, 'B': grade_b, 'C': grade_c}
        variabilities = {'A': var_a, 'B': var_b, 'C': var_c}
        deposit_ratio_choice = st.sidebar.selectbox("Deposit Ratio", ["1:1:1", "2:1:1"])

    # --- Run simulation ---
    deposit_pattern_truck = get_deposit_pattern_truck(ore_grades, deposit_ratio_choice)
    sim = PreHomogenizationPileSimulation(
        pile_length, stacker_velocity, production_rate,
        total_weight, truck_payload, ore_grades
    )

    for source in deposit_pattern_truck * int(sim.total_weight // sim.truck_payload):
        sim.add_material_section(sim.truck_payload, source)

    # --- Display plots and metrics ---
    fig_mean, fig_std, deposit_values, grade_matrix, column_means = plot_grade_matrix(sim, deposit_ratio_choice, variabilities)

    col_plot, col_metrics = st.columns(2)

    with col_plot:
        st.subheader("Stacking: Longitudinal Cross-Section")
        st.pyplot(plot_final_pile(sim, deposit_ratio_choice, variabilities))

    with col_metrics:
        st.subheader("Reclaiming: Segment Statistics")
        st.pyplot(fig_mean)
        st.pyplot(fig_std)

        mean_val = np.mean(column_means)
        std_val = np.std(column_means)

        st.subheader("Reclaiming: Overall Performance")
        space1, col1, space2, col2, space3 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            st.markdown(" ")
            st.metric("Mean Grade (%)", f"{mean_val:.2f}")
        with col2:
            st.markdown(" ")
            st.metric("Std Dev (%)", f"{std_val:.2f}")

    st.markdown("---")

    image1, image2 = st.columns(2)
    with image1:
        st.subheader("Chevron Stacking/Reclaiming Layout")
        st.image("assets/chevron.png", use_container_width=True)
        st.markdown(
            "Chevron stacking forms a central ridge. The stacker reverses direction to build layers upward, "
            "achieving vertical buildup with alternating passes."
        )

    with image2:
        st.subheader("Stacker - Sequential Layer Formation")
        st.image("assets/pre_homo_pile.gif", use_container_width=True)

    # st.markdown("---")

    # st.markdown("Grade Matrix (First 10 layers at the bottom)")
    # st.dataframe(grade_matrix[:10])
    # st.dataframe(grade_matrix[-10:])


if __name__ == "__main__":
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login()
    else:
        main()