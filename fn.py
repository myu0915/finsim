import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import fsolve
import altair as alt  # Import Altair for custom charting

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# ---------- Helper: compute schedule (Updated) ----------
def simulate_loan(
    principal: float,
    annual_rate: float,
    years: int,
    annual_inflation: float,
    expected_loss_rate: float,
    monthly_income0: float,
    yearly_income_growth: list,
):
    """
    Simulates a fixed-rate loan under inflation, including:
    - Borrower's net cash flow (income - payment)
    - Bank's Expected Loss Rate (ELR)

    yearly_income_growth: list of annual income growth rates (%).
    """

    # Convert % -> decimal
    r_annual = annual_rate / 100.0
    infl = annual_inflation / 100.0
    elr = expected_loss_rate / 100.0  # Expected Loss Rate (ELR)

    n_months = years * 12

    # --- Monthly Payment (Annuity) ---
    if r_annual == 0:
        payment = principal / n_months
        r_m = 0.0
    else:
        r_m = r_annual / 12.0
        denom = (1 + r_m) ** n_months - 1
        if denom == 0:
            st.error("Annuity calculation error: check rate and term.")
            return pd.DataFrame(), {}
        # Annuity Formula
        payment = principal * (r_m * (1 + r_m) ** n_months) / denom

    # Time series
    months = np.arange(1, n_months + 1)
    t_years = months / 12.0

    # --- Inflation Discounting ---
    discount_factor = 1 / ((1 + infl) ** t_years) if infl != -1 else np.ones_like(t_years)

    # Real Payments
    real_payment = payment * discount_factor
    cumulative_nominal_payment = payment * months
    cumulative_real_payment = np.cumsum(real_payment)

    # --- Annual Income Growth ---
    growth_rates = np.array(yearly_income_growth, dtype=float) / 100.0

    if growth_rates.size == 0:
        growth_rates = np.zeros(years)
    if growth_rates.size < years:
        # Fill missing years with the last specified value
        growth_rates = np.concatenate(
            [growth_rates, np.full(years - growth_rates.size, growth_rates[-1])]
        )

    # Cumulative income growth: 1.0 for Year 1, compounded thereafter
    income_growth_factors = np.concatenate(([1.0], 1 + growth_rates))
    income_factor_by_year = np.cumprod(income_growth_factors)[: years + 1]

    # Year index for each month (0 for Year 1, 1 for Year 2, etc.)
    year_index = (months - 1) // 12

    # Nominal monthly income
    income_nominal = monthly_income0 * income_factor_by_year[year_index]
    income_real = income_nominal * discount_factor

    # --- Borrower Cash Flow ---
    cash_flow_nominal = income_nominal - payment
    cash_flow_real = cash_flow_nominal * discount_factor

    # Payment Burden (Share of Income)
    payment_share_of_income = np.where(income_nominal > 0, payment / income_nominal, np.nan)

    # --- Key Metrics ---
    total_nominal = cumulative_nominal_payment[-1]
    total_real = cumulative_real_payment[-1]
    total_cash_flow_real = np.sum(cash_flow_real)

    # Bank's Real Profit and Borrower's Real Gain on the Contract
    bank_real_profit = total_real - principal
    borrower_real_gain = principal - total_real

    # Bank's Expected Loss Cost and Net Real Profit
    expected_loss_real_cost = principal * elr
    bank_net_real_profit = bank_real_profit - expected_loss_real_cost

    # --- Real Effective Rate (IRR using real payments) ---
    def real_rate_solver(r_real_array):
        r_real = float(r_real_array)
        r_real_m = r_real / 12.0
        if r_real_m <= -1.0:
            return principal

        # PV of real payments discounted by the real rate should equal the principal
        pv = np.sum(real_payment / (1 + r_real_m) ** months)
        return pv - principal

    if r_annual == 0:
        real_effective_rate = -infl
    else:
        if infl == -1:
            initial_guess = r_annual
        else:
            # Fisher Approximation for initial guess
            initial_guess = (1 + r_annual) / (1 + infl) - 1
        try:
            sol = fsolve(real_rate_solver, initial_guess, maxfev=1000)
            real_effective_rate = float(sol[0])
        except Exception:
            real_effective_rate = initial_guess

    # --- DataFrame ---
    df = pd.DataFrame(
        {
            "month": months,
            "years": t_years,
            "payment_nominal": payment,
            "payment_real": real_payment,
            "cum_payment_real": cumulative_real_payment,
            "income_nominal": income_nominal,
            "income_real": income_real,
            "cash_flow_nominal": cash_flow_nominal,
            "cash_flow_real": cash_flow_real,
            "payment_share_of_income": payment_share_of_income,
        }
    )

    # Path of profit/gain over time
    df["bank_profit_path"] = df["cum_payment_real"] - principal
    df["borrower_gain_path"] = principal - df["cum_payment_real"]

    # Burden in %
    if np.all(np.isnan(payment_share_of_income)):
        initial_burden = np.nan
        final_burden = np.nan
    else:
        initial_burden = float(payment_share_of_income[0] * 100)
        final_burden = float(payment_share_of_income[-1] * 100)

    metrics = {
        "payment_monthly": payment,
        "total_nominal": total_nominal,
        "total_real": total_real,
        "bank_real_profit": bank_real_profit,
        "borrower_real_gain": borrower_real_gain,
        "bank_net_real_profit": bank_net_real_profit,
        "total_cash_flow_real": total_cash_flow_real,
        "real_effective_rate": real_effective_rate * 100,
        "initial_burden": initial_burden,
        "final_burden": final_burden,
        "expected_loss_real_cost": expected_loss_real_cost,
    }

    return df, metrics


# --------- Streamlit UI ----------
st.title("Loan vs Inflation Simulator ⚖️")
st.write(
    """
This tool analyzes how **inflation**, **default risk**, and **income growth**
affect the positions of the **Bank** and the **Borrower** under a fixed-rate loan.
"""
)

# --- Sidebar: Loan Parameters ---
st.sidebar.header("Loan Parameters")
principal = st.sidebar.number_input("Principal Amount", value=15000.0, min_value=1000.0, step=500.0)
annual_rate = st.sidebar.number_input("Annual Interest Rate (%)", value=8.0, min_value=0.0, step=0.5)
years = st.sidebar.number_input("Loan Term (Years)", value=5, min_value=1, max_value=40, step=1)
years_int = int(years)

# --- Sidebar: Economic Factors ---
st.sidebar.header("Economic Factors")
annual_inflation = st.sidebar.number_input(
    "Actual Annual Inflation Rate (%)",
    value=3.0,
    min_value=-5.0,
    max_value=100.0,
    step=0.5,
    help="The actual inflation rate used to assess real value."
)

# Expected Loss Rate (ELR)
expected_loss_rate = st.sidebar.number_input(
    "Expected Loss Rate (ELR, % of Principal)",
    value=1.0,
    min_value=0.0,
    max_value=50.0,
    step=0.1,
    help="The percentage of the principal the bank expects to lose due to portfolio defaults. Affects NET Real Profit."
)

# --- Sidebar: Borrower Parameters ---
st.sidebar.header("Borrower Parameters")
monthly_income0 = st.sidebar.number_input(
    "Initial Monthly Income",
    value=2000.0,
    min_value=0.0,
    step=100.0,
)

max_years_for_ui = min(years_int, 10)  # Limiting UI fields for long terms
st.sidebar.markdown("**Annual Income Growth (%)**")

yearly_income_growth = []
for y in range(max_years_for_ui):
    g = st.sidebar.number_input(
        f"Growth for Year {y+1} (%)",
        value=0.0,
        min_value=-20.0,
        max_value=50.0,
        step=1.0,
        key=f"growth_year_{y+1}",
    )
    yearly_income_growth.append(g)

if years_int > max_years_for_ui and yearly_income_growth:
    st.sidebar.caption(
        f"Set for {max_years_for_ui} years. The last input ({yearly_income_growth[-1]:.1f}%) "
        f"is used for the remaining years."
    )
st.sidebar.caption(
    "**Note:** For an N-year loan, the income level in the N-th year is determined by the growth rates of the first N-1 years."
)

# --- Run Simulation ---
df, metrics = simulate_loan(
    principal=principal,
    annual_rate=annual_rate,
    years=years_int,
    annual_inflation=annual_inflation,
    expected_loss_rate=expected_loss_rate,
    monthly_income0=monthly_income0,
    yearly_income_growth=yearly_income_growth,
)

if df.empty or not metrics:
    st.stop()

st.divider()

# --------- Comparison Metrics (Real Terms) ---------
st.subheader("Summary Comparison (Nominal + Real Terms)")

col_bank, col_borrower = st.columns(2)

with col_bank:
    st.markdown("### 🏦 Bank's Position")
    st.metric(
        "Real Effective Rate (Actual)",
        f"{metrics['real_effective_rate']:.2f}%",
        help="What the bank truly earned/lost after accounting for inflation.",
    )
    st.metric(
        "Real Profit on Contract (Pre-ELR)",
        f"${metrics['bank_real_profit']:,.2f}",
        help="Positive = bank earned real value; Negative = inflation eroded principal/profit.",
    )
    st.metric(
        "Net Real Profit (Post-ELR)",
        f"${metrics['bank_net_real_profit']:,.2f}",
        help="Real Profit minus the expected loss due to defaults (ELR).",
    )
    st.caption(
        f"*Expected Loss Cost (ELR): ${metrics['expected_loss_real_cost']:,.2f}*"
    )

with col_borrower:
    st.markdown("### 👤 Borrower's Position")
    st.metric(
        "Monthly Payment (Nominal)",
        f"${metrics['payment_monthly']:,.2f}",
        help="Fixed amount paid by the borrower each month."
    )
    st.metric(
        "Total Paid (Nominal)",
        f"${metrics['total_nominal']:,.2f}",
        help="Sum of all payments over the loan term, ignoring inflation."
    )
    st.metric(
        "Real Gain on Contract",
        f"${metrics['borrower_real_gain']:,.2f}",
        help="Positive = Debt value was eroded by inflation (Borrower wins on the contract).",
    )
    st.metric(
        "Total Net Cash Flow (Real)",
        f"${metrics['total_cash_flow_real']:,.2f}",
        help="Total income minus loan payments, adjusted for inflation. Reflects real spending power.",
    )
    if not np.isnan(metrics["initial_burden"]):
        delta_burden = metrics["final_burden"] - metrics["initial_burden"]
        st.metric(
            "Payment Share of Income (End Term)",
            f"{metrics['final_burden']:.1f}%",
            delta=f"{delta_burden:.1f} pp (from {metrics['initial_burden']:.1f}%)",
            delta_color="inverse",
            help="Rising share means the borrower's quality of life is declining.",
        )
    else:
        st.metric(
            "Payment Share of Income",
            "N/A",
            help="Income is zero; burden cannot be calculated.",
        )

st.divider()

# --------- Charts (Using Altair for custom colors) ---------
st.subheader("Analysis Charts")

col_left, col_right = st.columns(2)

# 1. Contract Value: Cumulative Real Payments vs Principal (Bank Focus)
with col_left:
    st.markdown("#### 1. Contract Value: Cumulative Real Payments")
    st.markdown("##### Bank's Real Cumulative Return vs Principal")

    plot_real_df = df[["years", "cum_payment_real"]].copy()
    plot_real_df = plot_real_df.rename(
        columns={"years": "Years", "cum_payment_real": "Cumulative Real Payments"}
    )
    plot_real_df["Principal Amount (Breakeven)"] = principal

    # Prepare data for Altair
    source = plot_real_df.melt("Years", var_name="Metric", value_name="Value")

    base = alt.Chart(source).encode(x="Years:Q")

    # Colors: red for payments, green for principal
    color_scale = alt.Scale(
        domain=["Cumulative Real Payments", "Principal Amount (Breakeven)"],
        range=["red", "green"],
    )

    line_cum_payment = base.transform_filter(
        alt.datum.Metric == "Cumulative Real Payments"
    ).mark_line().encode(
        y=alt.Y("Value:Q", title="Value (Today's Money)"),
        color=alt.Color("Metric:N", scale=color_scale),
        tooltip=["Years", alt.Tooltip("Value:Q", title="Real Payments", format="$,.2f")],
    )

    line_principal = base.transform_filter(
        alt.datum.Metric == "Principal Amount (Breakeven)"
    ).mark_line().encode(
        y=alt.Y("Value:Q"),
        color=alt.Color("Metric:N", scale=color_scale),
        tooltip=["Years", alt.Tooltip("Value:Q", title="Principal", format="$,.2f")],
    )

    chart = (line_cum_payment + line_principal).properties(
        title="Cumulative Real Payments vs Principal"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "If the **Red Line** (Real Payments) ends **above** the **Green Line** (Principal) — the bank earned a real profit.\n"
        "If **below** — inflation eroded the return, and the borrower won on the contract."
    )

# 2. Quality of Life: Real Cash Flow & Burden (Borrower Focus)
with col_right:
    st.markdown("#### 2. Quality of Life: Real Cash Flow & Burden")
    st.markdown("##### Borrower's Real Cash Flow and Loan Burden")

    base = alt.Chart(df).encode(x=alt.X("years", title="Years"))

    # Layer 1: Real Net Cash Flow (Green Area, Left Axis)
    cash_flow_area = base.mark_area(opacity=0.5, color="green").encode(
        y=alt.Y(
            "cash_flow_real",
            title="Real Cash Flow (Today's $)",
            axis=alt.Axis(titleColor="green"),
        ),
        tooltip=[
            alt.Tooltip("years", title="Years", format=".1f"),
            alt.Tooltip("cash_flow_real", title="Real Cash Flow", format="$,.2f"),
        ],
    )

    # Layer 2: Payment Share of Income (Red Line, Right Axis)
    burden_line = base.mark_line(color="red").encode(
        y=alt.Y(
            "payment_share_of_income",
            title="Payment Share (%)",
            axis=alt.Axis(titleColor="red", format=".1%", orient="right"),
        ),
        tooltip=[
            alt.Tooltip("years", title="Years", format=".1f"),
            alt.Tooltip(
                "payment_share_of_income", title="Payment Share", format=".1%"
            ),
        ],
    )

    chart_combined = alt.layer(cash_flow_area, burden_line).resolve_scale(
        y="independent"
    ).properties(
        title="Real Cash Flow (Green) vs Payment Burden (Red)"
    ).interactive()

    st.altair_chart(chart_combined, use_container_width=True)

    st.caption(
        "**Real Cash Flow (Green Area):** Monthly income remaining after payment, adjusted for inflation. "
        "A downward trend means declining purchasing power.\n"
        "**Payment Share (Red Line):** Percentage of nominal income used for payment (rising = decreasing quality of life)."
    )

# 3. Real Gain/Loss: Bank vs Borrower (Full Width)
st.markdown("#### 3. Real Gain/Loss: Bank vs Borrower")
st.markdown("##### The Zero-Sum Game of Real Contract Value")

bank_borrower_df = df[["years", "bank_profit_path", "borrower_gain_path"]].copy()
bank_borrower_df = bank_borrower_df.rename(
    columns={
        "years": "Years",
        "bank_profit_path": "Bank's Real Cumulative Profit",
        "borrower_gain_path": "Borrower's Real Cumulative Gain",
    }
)

source_gain_loss = bank_borrower_df.melt("Years", var_name="Metric", value_name="Value")

color_scale_gain_loss = alt.Scale(
    domain=["Bank's Real Cumulative Profit", "Borrower's Real Cumulative Gain"],
    range=["red", "green"],
)

chart_gain_loss = (
    alt.Chart(source_gain_loss)
    .mark_line()
    .encode(
        x="Years:Q",
        y=alt.Y("Value:Q", title="Real Value Difference ($)"),
        color=alt.Color("Metric:N", scale=color_scale_gain_loss),
        tooltip=["Years", alt.Tooltip("Value:Q", format="$,.2f")],
    )
    .properties(title="Bank's Profit vs Borrower's Gain (Real Terms)")
    .interactive()
)

st.altair_chart(chart_gain_loss, use_container_width=True)

st.caption(
    "These two lines are mirror images of the same contract: "
    "Bank's Profit = - Borrower's Gain. "
    "The distance from the zero line shows the accumulated real win/loss over the loan term."
)
