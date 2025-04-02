import streamlit as st
from auth import login_required

st.set_page_config(
    page_title="Heilmeier",
    page_icon="assets/icon.png",
    layout="wide"
)

login_required()  # 🚨 Block access if not logged in

st.title("Heilmeier Catechism")
st.subheader("Improving Homogenization of Chevron Pile Using Machine Learning Techniques")

st.markdown("""
Ricardo Ramos Nunes  
*Mining and Geological Engineering Department*  
*University of Arizona* 

---

### What are you trying to do?

The purpose of this research is to develop a control system employing machine learning techniques to enhance the deposition process of chevron piles. The primary objective is to achieve a highly homogeneous pile that facilitates the reclaiming operation with a consistent ore grade distribution, thereby reducing operational costs.

---

### How is it done today and what are the limits of current practice?

Current deposition methodologies predominantly depend on static control strategies related to stacker operations. These techniques:

- Use fixed, non-adaptive deposition patterns that do not account for dynamic variations in material properties.
- Often result in uneven ore grade distribution and segregation within the pile.

As a result, current practices can lead to inconsistent pile quality and suboptimal mill operation.

---

### What is new in your approach, and why do you think it will be successful?

Our approach leverages machine learning techniques, such as reinforcement learning and predictive modeling, to dynamically control the deposition process in real time.

Key innovations include:

- **Adaptive Control**: An ML-based system will continuously optimize the movement of the stackers and the strategies of dumping trucks to minimize the variability of the grade of the ore.
- **Simulation-Driven Optimization**: A high-fidelity simulation environment will be used to train the ML models, allowing them to explore a wide range of scenarios.
- **Data-Driven Decision Making**: Integration of real-time sensor data to refine control strategies.

These innovations are expected to overcome the limitations of current practices by achieving a more homogeneous pile with improved structural and economic performance.

---

### Who cares? If you are successful, what difference will it make?

- **Mine Operators**: Improved ore recovery and reduced processing costs through a more uniform pile.
- **Industry Stakeholders**: Increased operational efficiency and competitiveness through advanced data-driven technology.

---

### What are the risks?

- **Model Generalizability**: The ML model may struggle to capture all the complexities of real-world deposition.
- **Integration Challenges**: Difficulties in interfacing the control system with existing hardware and operations.
- **Data Quality**: Limited or noisy data can affect the performance and reliability of ML algorithms.

---

### How much will it cost?

We anticipate minimal direct costs by leveraging existing simulation frameworks, open-source ML libraries, and historical data sets. Any additional costs would be associated with the integration of the system into the operational environment.

---

### How long will it take?

- **Phase 1**: Model development and simulation (3–4 months).
- **Phase 2**: System integration and pilot testing (2–3 months).
- **Phase 3**: Full-scale deployment and optimization (1–2 months).

**Total Project Time**: Approximately 6–9 months.

---

### What are the mid-term and final “exams” to check for success?

**Mid-term Exam:**

- Demonstrate that the ML model reduces ore grade variability in a simulated environment by a predefined margin.
- Validate the adaptive control system in various operational scenarios.

**Final Exam:**

- Successfully integrate the system into a pilot operation and validate improved pile homogeneity through field measurements.
- Show measurable improvements in ore recovery, cost efficiency, and pile stability.

---

### Summary

Improving the Homogenization of the Chevron Pile Using Machine Learning Techniques aims to revolutionize the deposition process by introducing an adaptive, data-driven control system. By integrating advanced ML algorithms with real-time operational data, the project seeks to create a more uniform and efficient chevron pile, ultimately leading to enhanced ore recovery, improved safety, and reduced production costs. The project builds on existing technologies and methodologies, offering a scalable solution to modern challenges in mining operations.
""")
