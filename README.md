Prudentia – Basel III/IV Quantitative Risk Engine

A next-generation industrial-grade risk engine bridging Quantitative Finance, Regulatory Compliance, and Modern Software Engineering.

Executive Summary

Prudentia automatizes the end-to-end credit‑risk lifecycle, delivering a fully compliant and explainable framework aligned with Basel III/IV and European Banking Authority (EBA) expectations.

It addresses three core supervisory and banking needs:

1. Credit Scoring (IRB PD Model)

Logistic Regression with Weight of Evidence (WoE) encoding

Monotonic & interpretable scorecard‑style modeling

SME/Large Corporate scaling

2. Regulatory Capital Calculation (Basel ASRF)

Computation of RWA, Expected Loss (EL), and Capital Requirement (K)

Fully compliant with A‑IRB formulas

Pure mathematical domain kernel (no external deps)

3. Stress Testing Engine

Macroeconomic scenario application (GDP, Unemployment, Inflation)

Probit‑Shift methodology per EBA guidelines

Portfolio solvency simulation under extreme but plausible shocks

✨ Interactive Dashboard (UI)

Prudentia includes a user-friendly interface powered by Streamlit, allowing Risk Managers and Supervisors to interact with the engine without writing code.

Key Capabilities:

Loan Configuration: Adjust PD, LGD, Maturity, and Exposure in real-time.

Scenario Selection: Switch between Baseline, Adverse, and Severely Adverse economic scenarios.

Instant Feedback: Visualize the impact on Capital Requirements and RWA immediately.

(Note: Screenshots below are illustrative of the running application)

1. Configuration & Scenario Selection

The user configures a Corporate loan and applies a "Severely Adverse" macroeconomic shock (Z=3.0).

2. Impact Assessment Results

The engine computes the new PD (via Probit Shift) and the resulting Capital squeeze.

Architectural Overview

Prudentia adopts a strict Hexagonal Architecture (Ports & Adapters) ensuring maintainability, testability, and regulatory auditability.

Domain (Core Math)
│
├── Application (Orchestration)
│
└── Infrastructure (APIs, ML, IO)


Why Hexagonal?

The Basel math never depends on ML or FastAPI.

Regulatory logic remains frozen, deterministic, validated.

Infrastructure can evolve freely (APIs, pipelines, cloud deployment).

System Diagram (Mermaid)

graph TD
    subgraph "External Actors"
        Client[Bank / User]
        Macro[Macroeconomic Data]
    end

    subgraph "Infrastructure Layer (Adapters)"
        API[FastAPI Interface]
        ML[Scikit-Learn Pipeline]
        UI[Streamlit Dashboard]
    end

    subgraph "Application Layer"
        Stressor[Stress Test Orchestrator]
    end

    subgraph "Domain Layer (Pure Business Logic)"
        Entity[Loan & Portfolio Entities]
        Basel[Basel III Formulas]
    end

    UI -->|HTTP Requests| API
    Client -->|1. POST Portfolio| API
    API -->|2. Request Assessment| Stressor
    Stressor -->|3. Transform Features| ML
    Stressor -->|4. Apply Macro Shocks| Macro
    Stressor -->|5. Compute Capital| Basel
    Basel -->|6. Update State| Entity


Quantitative Framework (Mathematics)

Prudentia implements the Asymptotic Single Risk Factor (ASRF) model underlying Basel A‑IRB.

1. Asset Correlation ($R$)

High‑quality borrowers are more correlated with systemic risk.

Corporate exposures:

$$R = 0.12 \cdot \frac{1 - e^{-50PD}}{1 - e^{-50}} + 0.24 \cdot \left[1 - \frac{1 - e^{-50PD}}{1 - e^{-50}}\right]$$

SME correlation adjustment also available per BCBS guidelines.

2. Maturity Adjustment ($b$, $MF$)

Smoothing factor:

$$b = (0.11852 - 0.05478\ln(PD))^2$$

Final maturity factor:

$$MF = \frac{1 + (M - 2.5)b}{1 - 1.5b}$$

3. Capital Requirement ($K$) – Vasicek ASRF

This formula defines the unexpected loss at confidence level 99.9%.

$$K = \Bigg[ LGD \cdot \Phi \left( \frac{\Phi^{-1}(PD) + \sqrt{R}\Phi^{-1}(0.999)}{\sqrt{1-R}} \right) - LGD \cdot PD \Bigg] \cdot MF$$

Where:

$\Phi$ = Normal CDF

$\Phi^{-1}$ = Normal inverse CDF

4. Stress Testing – Probit Shift

To simulate macroeconomic deterioration:

$$PD_{stressed} = \Phi\left( \Phi^{-1}(PD_{base}) + Sensitivity \cdot Z_{scenario} \right)$$

This approach ensures coherent deformation of the PD distribution under recessionary shocks.

Machine Learning Approach

While XGBoost or Neural Nets outperform in raw AUC, regulators require transparency.

Therefore Prudentia uses:

✔ Logistic Regression

✔ Weight of Evidence (WoE) binning

✔ Monotonic risk factors

Benefits:

Fully explainable decision boundary

Governance‑ready for ACPR/ECB internal model approval

Natural handling of missing values

Installation & Usage

Prerequisites

Python 3.10+

Poetry (recommended)

1. Clone repository

git clone [https://github.com/YOUR_USERNAME/prudentia-risk-engine.git](https://github.com/YOUR_USERNAME/prudentia-risk-engine.git)
cd prudentia-risk-engine


2. Install dependencies

poetry install


3. Train the model

python -m src.scripts.train_model
# → Output: Model saved to data/models/scorecard_model.pkl


4. Launch the Application

You need two terminals to run the full stack (Engine + UI).

Terminal 1: The Calculation Engine (API)

python -m uvicorn src.api.main:app --reload
# Server running at [http://127.0.0.1:8000](http://127.0.0.1:8000)


Terminal 2: The User Interface (Streamlit)

python -m streamlit run app.py
# UI opening at http://localhost:8501


5. Swagger Documentation

Alternatively, access the raw API docs:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)


Project Structure

src/
├── domain/               # CORE MATHEMATICAL KERNEL
│   ├── entities.py       # Loan / Portfolio Models
│   └── basel_formulas.py # ASRF & Basel Calculations
├── engine/               # ORCHESTRATION LOGIC
│   └── stressor.py       # Stress Testing Engine
├── processing/           # DATA ENGINEERING
│   └── woe_encoder.py    # WoE Transformer
├── api/                  # REST INTERFACE
│   └── main.py           # FastAPI Endpoints
└── scripts/              # OPERATIONS / PIPELINES
    └── train_model.py


Continuous Integration (CI)

GitHub Actions ensure reliability on every commit.

Pipeline includes:

Ruff for linting

MyPy for static type safety

Pytest for mathematical & functional validation

License

Distributed under the MIT License.

Prudentia – Because risk engines deserve elegance too.