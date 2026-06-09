import streamlit as st
import pandas as pd
import pickle as pk
import re
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title='Loan Approval Dashboard',
    page_icon='💼',
    layout='wide'
)

# Load the trained model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Validation helpers
def is_valid_aadhaar(aadhaar_number):
    return bool(re.match(r'^\d{4} \d{4} \d{4}$', aadhaar_number))

def is_valid_mobile(mobile_number):
    return bool(re.match(r'^\d{10}$', mobile_number))

# EMI calculation and plot helper
def create_emi_figure(principal, rate_of_interest, time_period):
    monthly_rate = rate_of_interest / (12 * 100)
    months = time_period * 12
    emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)

    remaining_principal = principal
    principal_payments = []
    interest_payments = []
    emi_values = []
    timeline = np.arange(1, months + 1)

    for _ in timeline:
        interest_payment = remaining_principal * monthly_rate
        principal_payment = emi - interest_payment
        remaining_principal -= principal_payment
        principal_payments.append(principal_payment)
        interest_payments.append(interest_payment)
        emi_values.append(emi)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(timeline, emi_values, label='EMI', color='#1f77b4')
    ax.plot(timeline, interest_payments, label='Interest', color='#ff7f0e')
    ax.plot(timeline, principal_payments, label='Principal', color='#2ca02c')
    ax.set_title('EMI Breakdown Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount (INR)')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()

    return emi, fig

# Interest rate comparison helper
def create_interest_rate_figure(loan_type):
    banks = list(interest_rates[loan_type].keys())
    rates = list(interest_rates[loan_type].values())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    palette = ['#2c7fb8', '#7fcdbb', '#fed976', '#f03b20', '#bd0026']
    bars = ax.bar(banks, rates, color=palette)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{rate:.2f}%',
                ha='center', va='bottom', fontsize=9)

    ax.set_title(f'Interest Rates for {loan_type}')
    ax.set_xlabel('Bank')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_xticklabels(banks, rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.tight_layout()

    max_rate = max(rates)
    min_rate = min(rates)
    max_bank = banks[rates.index(max_rate)]
    min_bank = banks[rates.index(min_rate)]

    return fig, min_bank, min_rate, max_bank, max_rate

# Model explanation helpers
def approval_probability(data):
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(data)[0][1])
    if hasattr(model, 'decision_function'):
        score = float(model.decision_function(data)[0])
        return 1 / (1 + np.exp(-score))
    return 0.5

def explanation_summary(prediction, probability, profile):
    summary = []
    if prediction == 1:
        summary.append('🎯 Approval likelihood is strong based on your current profile.')
    elif probability >= 0.50:
        summary.append('⚠️ Approval is possible, but there are a few areas to strengthen.')
    else:
        summary.append('🚫 Approval is not likely with the current profile.')

    summary.append(f'📊 Estimated approval probability: {probability*100:.1f}%')
    summary.append('🔎 Key factors influencing the prediction are shown below.')
    return summary

def recommendation_insights(profile):
    advice = []
    dti_ratio = profile['loan_amount'] / max(profile['annual_income'], 1)
    assets = profile.get('assets', profile.get('Assets', 0))

    if profile['cibil_score'] < 700:
        advice.append('📈 Improve your CIBIL score by at least 50 points to increase approval confidence.')
    else:
        advice.append('✅ Your CIBIL score is solid; continue with timely repayments and low credit utilization.')

    if dti_ratio > 0.45:
        advice.append('⚖️ Your debt-to-income ratio is high; consider reducing the loan amount or increasing income.')
    else:
        advice.append('💡 Your loan amount is reasonable relative to your income.')

    if assets < profile['loan_amount'] * 0.4:
        advice.append('🏦 Strengthen your asset position by adding savings, investments, or collateral.')
    else:
        advice.append('🔒 Your asset coverage gives good support for the requested loan.')

    if profile['self_employed']:
        advice.append('🧾 Self-employed applicants benefit from stable business documentation and a longer income history.')
    else:
        advice.append('📊 Salaried applicants generally score higher when employment is stable and documents are current.')

    if profile['loan_purpose'] == 'Home Purchase':
        advice.append('🏠 Home loans may qualify for lower rates if documentation and property valuation are complete.')
    elif profile['loan_purpose'] == 'Education':
        advice.append('🎓 Education loans can qualify for special support schemes or subsidized rates.')
    elif profile['loan_purpose'] == 'Business Expansion':
        advice.append('📈 Business loans perform better with a clear cash flow or revenue growth plan.')
    elif profile['loan_purpose'] == 'Vehicle Purchase':
        advice.append('🚗 Vehicle loans are more attractive when the borrower has a stable income and low existing liabilities.')
    elif profile['loan_purpose'] == 'Personal Need':
        advice.append('💼 Personal loans need a strong repayment plan and a clear borrowing purpose.')

    return advice

# Interest rates dictionary
interest_rates = {
    'Home Loan': {
        'Kotak Mahindra Bank': 8.65,
        'SBI': 8.90,
        'Union Bank of India': 8.60,
        'Bank of Baroda': 8.60,
        'Bank of India': 8.65,
    },
    'Vehicle Loan': {
        'Bank of Baroda': 8.40,
        'Canara Bank': 9.15,
        'Axis Bank': 8.50,
        'Federal Bank': 10.75,
        'SBI': 8.50,
    },
    'Gold Loan': {
        'Axis Bank': 13.50,
        'HDFC Bank': 11.00,
        'Canara Bank': 7.35,
        'ICICI Bank': 11.00,
        'SBI': 7.00,
    },
    'Personal Loan': {
        'HDFC Bank': 10.5,
        'ICICI Bank': 10.75,
        'SBI': 9.60,
        'Yes Bank': 10.00,
        'Axis Bank': 10.25,
    },
}

# Page state
if 'page' not in st.session_state:
    st.session_state.page = 'Introduction'

# Sidebar navigation and status
steps = ['Introduction', 'Personal Details', 'Financial Details']
st.sidebar.title('Loan Dashboard')
selected_step = st.sidebar.radio('Navigation', steps, index=steps.index(st.session_state.page))
st.session_state.page = selected_step

# Introduction page

def show_introduction():
    st.markdown(
        "<div style='padding: 1.5rem; background: linear-gradient(135deg, #111827 0%, #0f172a 100%); border-radius: 18px;'>"
        "<h1 style='color: #f8fafc; margin-bottom: 0.2rem;'>💼 Loan Approval Navigator</h1>"
        "<h3 style='color: #cbd5e1; margin-top: 0;'>Transform your loan application into a confident financial decision.</h3>"
        "</div>", unsafe_allow_html=True
    )

    st.write('')
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('### Intelligent loan guidance with a professional edge')
        st.write('- **Personalized approval insight** tuned to your profile. 📈')
        st.write('- **EMI planning and cost visibility** with clear comparisons. 💡')
        st.write('- **Improvement suggestions** to strengthen your application. 🧭')
        st.write('- **Clean, simple workflow** from introduction to decision. 🛡️')
    with c2:
        st.markdown('### Quick Snapshot')
        st.metric('Loan clarity', 'Strong', 'Focused outcomes')
        st.metric('Action items', 'Ready', 'Personalized advice')
        st.metric('Review flow', 'Smooth', '3-step process')

    st.markdown('---')
    st.markdown(
        "<div style='padding: 1rem; background: #111827; border-radius: 16px;'>"
        "<h4 style='color: #f8fafc;'>Start in three steps</h4>"
        "<ol style='color: #cbd5e1; line-height: 1.8;'>"
        "<li>Fill in your personal profile for a trusted identity foundation.</li>"
        "<li>Share your financial details and loan purpose to get a precise assessment.</li>"
        "<li>Review approval probability, EMI plan, and targeted next steps.</li>"
        "</ol>"
        "</div>", unsafe_allow_html=True
    )

    st.success('Step 1 of 3 — Landing page: begin your loan assessment journey')

    if st.button('Begin Loan Assessment', key='intro_begin'):
        st.session_state.page = 'Personal Details'

# Personal details page

def show_personal_details():
    st.markdown(
        "<div style='padding: 1rem; background: #111827; border-radius: 18px;'>"
        "<h2 style='color: #e2e8f0; margin-bottom: 0.2rem;'>👤 Applicant Information</h2>"
        "<p style='color: #94a3b8; margin-top: 0;'>Complete your identity profile with clean details to strengthen your loan application.</p>"
        "</div>", unsafe_allow_html=True
    )
    st.write('')
    st.warning('Step 2 of 3 — Personal Information')
    st.markdown('Enter your contact and identity details clearly to build a professional loan profile.')

    with st.form(key='personal_form'):
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input('First Name')
            last_name = st.text_input('Last Name')
            age = st.number_input('Age', min_value=18, max_value=100, value=25)
        with col2:
            aadhaar_number = st.text_input('Aadhaar Number (xxxx xxxx xxxx)')
            mobile_number = st.text_input('Mobile Number (10 digits)', max_chars=10)
            address = st.text_area('Address')

        submit = st.form_submit_button('Save Personal Details')

    if submit:
        if not first_name or not last_name or not address:
            st.error('Please complete all personal fields.')
        elif not is_valid_aadhaar(aadhaar_number):
            st.error('Aadhaar number must be in the format xxxx xxxx xxxx.')
        elif not is_valid_mobile(mobile_number):
            st.error('Mobile number must be 10 digits.')
        else:
            st.session_state.personal_details = {
                'first_name': first_name,
                'last_name': last_name,
                'age': age,
                'address': address,
                'aadhaar_number': aadhaar_number,
                'mobile_number': mobile_number,
            }
            st.success('Personal details saved successfully. Proceeding to Financial Details ➜')
            st.session_state.page = 'Financial Details'

# Financial details page

def show_financial_details():
    st.markdown(
        "<div style='padding: 1rem; background: #111827; border-radius: 18px;'>"
        "<h2 style='color: #e2e8f0; margin-bottom: 0.2rem;'>💰 Financial Profile</h2>"
        "<p style='color: #94a3b8; margin-top: 0;'>Complete your loan profile with accurate financial inputs and a clear borrowing plan.</p>"
        "</div>", unsafe_allow_html=True
    )

    st.write('')
    st.warning('Step 3 of 3 — Financial Details')
    st.markdown('Enter your income, loan amount, assets and purpose to unlock approval insights and actionable advice.')

    with st.form(key='financial_form'):
        col1, col2 = st.columns(2)

        with col1:
            no_of_dep = st.number_input('Number of Dependents', min_value=0, max_value=5, value=0)
            education = st.selectbox('Education Status', ['Graduated', 'Not Graduated'])
            self_employed = st.selectbox('Employment Status', ['Salaried', 'Self-Employed'])
            loan_purpose = st.selectbox(
                'Loan Purpose',
                ['Home Purchase', 'Vehicle Purchase', 'Education', 'Business Expansion', 'Personal Need']
            )
            loan_type = st.selectbox('Loan Type', ['Home Loan', 'Vehicle Loan', 'Gold Loan', 'Personal Loan'])
            bank = st.selectbox('Preferred Bank', list(interest_rates[loan_type].keys()))

        with col2:
            annual_income = st.number_input('Annual Income (INR)', min_value=0, value=500000, format='%d')
            loan_amount = st.number_input('Loan Amount (INR)', min_value=50000, value=2000000, format='%d')
            assets = st.number_input('Assets (INR)', min_value=0, value=500000, format='%d')
            loan_duration = st.number_input('Loan Duration (years)', min_value=1, max_value=30, value=10)
            cibil_score = st.number_input('CIBIL Score', min_value=300, max_value=900, value=700)

        submit = st.form_submit_button('Assess Loan Eligibility')

    if submit:
        if annual_income <= 0 or loan_amount <= 0 or loan_duration <= 0 or cibil_score <= 0 or assets < 0:
            st.error('Please enter valid financial values.')
            return

        profile = {
            'no_of_dependents': no_of_dep,
            'education': 0 if education == 'Graduated' else 1,
            'self_employed': 1 if self_employed == 'Self-Employed' else 0,
            'annual_income': annual_income,
            'loan_amount': loan_amount,
            'loan_term': loan_duration,
            'cibil_score': cibil_score,
            'Assets': assets,
            'assets': assets,
            'loan_purpose': loan_purpose,
            'loan_type': loan_type,
            'bank': bank,
            'interest_rate': interest_rates[loan_type][bank],
        }

        input_data = pd.DataFrame([
            [
                profile['no_of_dependents'],
                profile['education'],
                profile['self_employed'],
                profile['annual_income'],
                profile['loan_amount'],
                profile['loan_term'],
                profile['cibil_score'],
                profile['Assets'],
            ]
        ], columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])

        scaled_data = scaler.transform(input_data)
        prediction = int(model.predict(scaled_data)[0])
        probability = approval_probability(scaled_data)
        summary = explanation_summary(prediction, probability, profile)
        advice = recommendation_insights(profile)

        st.write('---')
        st.markdown("<div style='padding: 1rem; background: #111827; border-radius: 16px;'>"
                    "<h3 style='color: #e2e8f0; margin-bottom: 0.2rem;'>🔍 Approval Insight Summary</h3>"
                    "<p style='color: #94a3b8; margin-top: 0;'>A clear, professional summary of how your application looks to lenders.</p>"
                    "</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            for line in summary:
                st.write(f'- {line}')

            st.write('**Profile factors:**')
            st.write(f'- Loan purpose: {loan_purpose}')
            st.write(f'- Loan type: {loan_type}')
            st.write(f'- Preferred bank: {bank}')
            st.write(f'- Annual income: ₹{annual_income:,}')
            st.write(f'- Requested loan amount: ₹{loan_amount:,}')
            st.write(f'- Loan duration: {loan_duration} years')
            st.write(f'- CIBIL score: {cibil_score}')
            st.write(f'- Asset coverage: ₹{assets:,}')

        with col2:
            if prediction == 1:
                st.success('✅ Loan Status: Approved')
            else:
                st.error('❌ Loan Status: Rejected')

            st.metric('Approval Probability', f'{probability*100:.1f}%')
            st.metric('Interest Rate Selected', f'{profile["interest_rate"]:.2f}%')
            dti_ratio = profile['loan_amount'] / max(profile['annual_income'], 1)
            st.metric('Debt-to-Income Ratio', f'{dti_ratio:.2f}')

        st.write('---')
        st.markdown("<div style='padding: 1rem; background: #111827; border-radius: 16px;'>"
                    "<h3 style='color: #e2e8f0; margin-bottom: 0.2rem;'>🛠️ Actionable Recommendations</h3>"
                    "<p style='color: #94a3b8; margin-top: 0;'>Focused advice to improve your loan profile next.</p>"
                    "</div>", unsafe_allow_html=True)
        for item in advice:
            st.write(f'- {item}')

        st.write('---')
        st.subheader('EMI and Rate Comparison')
        emi_value, emi_fig = create_emi_figure(loan_amount, profile['interest_rate'], loan_duration)
        rate_fig, min_bank, min_rate, max_bank, max_rate = create_interest_rate_figure(loan_type)

        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            st.pyplot(emi_fig)
            st.write(f'Estimated monthly EMI: **₹{emi_value:,.2f}**')
        with graph_col2:
            st.pyplot(rate_fig)
            st.info(f'Lowest rate: {min_bank} ({min_rate}%) • Highest rate: {max_bank} ({max_rate}%)')

# Main application route
if st.session_state.page == 'Introduction':
    show_introduction()
elif st.session_state.page == 'Personal Details':
    show_personal_details()
elif st.session_state.page == 'Financial Details':
    show_financial_details()
