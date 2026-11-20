import os
import pickle
import warnings

import pandas as pd
from flask import Flask, Response, jsonify, request

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Decision.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# Label encodings (alphabetical, as produced by sklearn LabelEncoder).
# If your training notebook used a different mapping, edit it here.
GENDER = {"female": 0, "male": 1}
EDUCATION = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
INTENT = {
    "DEBTCONSOLIDATION": 0,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3,
    "PERSONAL": 4,
    "VENTURE": 5,
}
DEFAULTS = {"No": 0, "Yes": 1}


def pick(mapping, value, field):
    if value not in mapping:
        raise ValueError(f"Invalid value for {field}")
    return mapping[value]


def num(data, key, lo, hi):
    try:
        v = float(data.get(key))
    except (TypeError, ValueError):
        raise ValueError(f"{key.replace('_', ' ').title()} must be a number")
    if not lo <= v <= hi:
        raise ValueError(f"{key.replace('_', ' ').title()} must be between {lo} and {hi}")
    return v


INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Decision Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --navy:#0B1F3A; --navy-2:#13315C; --teal:#0F9D8A; --teal-soft:#DDF3EF;
  --gold:#C8963E; --red:#B83A3A; --red-soft:#F8E3E3;
  --paper:#F3F6FA; --card:#FFFFFF; --ink:#12233D; --muted:#5B6B82; --line:#D9E1EC;
}
*{box-sizing:border-box;margin:0}
body{font-family:Manrope,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--paper);color:var(--ink);min-height:100vh;line-height:1.5}
.shell{display:grid;grid-template-columns:minmax(300px,380px) 1fr;min-height:100vh}

/* Left panel */
aside{background:linear-gradient(165deg,var(--navy) 0%,var(--navy-2) 100%);color:#fff;padding:48px 36px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;animation:enter .7s cubic-bezier(.2,.7,.2,1) both}
@keyframes enter{from{opacity:0;transform:translateX(-24px)}to{opacity:1;transform:none}}
.brand{font-weight:800;font-size:1.5rem;letter-spacing:-.02em}
.brand span{color:var(--gold)}
.tag{color:#B7C4D9;font-size:.95rem;margin-top:8px;max-width:28ch}
.gauge-wrap{margin:auto 0;text-align:center}
.gauge{position:relative;width:220px;height:220px;margin:0 auto}
.gauge svg{transform:rotate(-90deg)}
.gauge circle{fill:none;stroke-width:14}
.track{stroke:rgba(255,255,255,.12)}
.bar{stroke:var(--gold);stroke-linecap:round;stroke-dasharray:553;stroke-dashoffset:553;transition:stroke-dashoffset 1.3s cubic-bezier(.22,.8,.25,1),stroke .4s}
.bar.ok{stroke:#3DD6BE}.bar.no{stroke:#F08A8A}
.gauge-num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge-num b{font-size:3rem;font-weight:800;letter-spacing:-.03em}
.gauge-num small{color:#B7C4D9;font-size:.8rem}
.verdict{margin-top:28px;font-size:1.35rem;font-weight:700;min-height:2em}
.verdict-sub{color:#B7C4D9;font-size:.9rem;max-width:30ch;margin:6px auto 0}
.reveal{animation:pop .5s cubic-bezier(.2,.9,.3,1.2) both}
@keyframes pop{from{opacity:0;transform:scale(.92)}to{opacity:1;transform:none}}
.foot{color:#8DA0BC;font-size:.78rem}

/* Form */
main{padding:48px clamp(20px,5vw,72px);max-width:920px;width:100%}
h1{font-size:clamp(1.6rem,3vw,2.1rem);letter-spacing:-.02em;font-weight:800}
.lead{color:var(--muted);margin:6px 0 28px}
fieldset{border:1px solid var(--line);background:var(--card);border-radius:14px;padding:24px;margin-bottom:20px}
legend{font-weight:700;padding:0 10px;margin-left:-10px;color:var(--navy-2)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:18px}
label{display:block;font-size:.84rem;font-weight:600;color:var(--muted);margin-bottom:6px}
input,select{width:100%;padding:11px 13px;border:1px solid var(--line);border-radius:9px;font:inherit;color:var(--ink);background:#fff;transition:border-color .2s,box-shadow .2s}
input:hover,select:hover{border-color:#B6C3D6}
input:focus,select:focus{outline:none;border-color:var(--teal);box-shadow:0 0 0 3px rgba(15,157,138,.2)}
.hint{font-size:.75rem;color:var(--muted);margin-top:4px}
.ratio{display:flex;justify-content:space-between;align-items:center;background:var(--paper);border-radius:9px;padding:11px 13px;font-weight:700}
.ratio span{font-weight:500;color:var(--muted);font-size:.84rem}
button{width:100%;margin-top:6px;padding:15px;border:0;border-radius:11px;background:var(--teal);color:#fff;font:700 1rem Manrope,sans-serif;cursor:pointer;transition:background .2s,transform .15s,box-shadow .2s;box-shadow:0 6px 16px rgba(15,157,138,.28)}
button:hover{background:#0C8575;transform:translateY(-1px)}
button:active{transform:none}
button:focus-visible{outline:3px solid var(--gold);outline-offset:2px}
button[disabled]{opacity:.7;cursor:progress}
.err{display:none;background:var(--red-soft);color:var(--red);border-radius:9px;padding:11px 14px;margin-bottom:14px;font-size:.9rem;font-weight:600}

@media(max-width:860px){
  .shell{grid-template-columns:1fr}
  aside{position:relative;height:auto;padding:32px 24px}
  .gauge-wrap{margin:28px 0 8px}
  .gauge{width:180px;height:180px}.gauge svg{width:180px;height:180px}
}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="shell">
  <aside>
    <div>
      <div class="brand">Loan<span>Decision</span></div>
      <p class="tag">Check an application against the trained decision-tree model.</p>
    </div>
    <div class="gauge-wrap" id="result" aria-live="polite">
      <div class="gauge">
        <svg width="220" height="220" viewBox="0 0 200 200">
          <circle class="track" cx="100" cy="100" r="88"/>
          <circle class="bar" id="bar" cx="100" cy="100" r="88"/>
        </svg>
        <div class="gauge-num"><b id="pct">--</b><small>approval score</small></div>
      </div>
      <div class="verdict" id="verdict">Awaiting application</div>
      <p class="verdict-sub" id="sub">Fill in the form and select Check application.</p>
    </div>
    <p class="foot">Model output is a statistical estimate and not a lending decision.</p>
  </aside>

  <main>
    <h1>Applicant details</h1>
    <p class="lead">All fields are required. Loan-to-income is calculated for you.</p>
    <div class="err" id="err" role="alert"></div>

    <form id="form" novalidate>
      <fieldset>
        <legend>Applicant</legend>
        <div class="grid">
          <div><label for="person_age">Age</label><input id="person_age" name="person_age" type="number" min="18" max="100" value="28" required></div>
          <div><label for="person_gender">Gender</label>
            <select id="person_gender" name="person_gender"><option value="female">Female</option><option value="male">Male</option></select></div>
          <div><label for="person_education">Education</label>
            <select id="person_education" name="person_education"><option>High School</option><option>Associate</option><option selected>Bachelor</option><option>Master</option><option>Doctorate</option></select></div>
          <div><label for="person_home_ownership">Home ownership</label>
            <select id="person_home_ownership" name="person_home_ownership"><option value="RENT">Rent</option><option value="MORTGAGE">Mortgage</option><option value="OWN">Own</option><option value="OTHER">Other</option></select></div>
        </div>
      </fieldset>

      <fieldset>
        <legend>Income and credit</legend>
        <div class="grid">
          <div><label for="person_income">Annual income</label><input id="person_income" name="person_income" type="number" min="1" value="60000" required></div>
          <div><label for="person_emp_exp">Years of employment</label><input id="person_emp_exp" name="person_emp_exp" type="number" min="0" value="4" required></div>
          <div><label for="credit_score">Credit score</label><input id="credit_score" name="credit_score" type="number" min="300" max="850" value="650" required><div class="hint">Range 300 to 850</div></div>
          <div><label for="cb_person_cred_hist_length">Credit history (years)</label><input id="cb_person_cred_hist_length" name="cb_person_cred_hist_length" type="number" min="0" value="5" required></div>
          <div><label for="previous_loan_defaults_on_file">Previous loan defaults</label>
            <select id="previous_loan_defaults_on_file" name="previous_loan_defaults_on_file"><option>No</option><option>Yes</option></select></div>
        </div>
      </fieldset>

      <fieldset>
        <legend>Loan request</legend>
        <div class="grid">
          <div><label for="loan_amnt">Loan amount</label><input id="loan_amnt" name="loan_amnt" type="number" min="1" value="10000" required></div>
          <div><label for="loan_intent">Purpose</label>
            <select id="loan_intent" name="loan_intent"><option value="PERSONAL">Personal</option><option value="EDUCATION">Education</option><option value="MEDICAL">Medical</option><option value="VENTURE">Venture</option><option value="HOMEIMPROVEMENT">Home improvement</option><option value="DEBTCONSOLIDATION">Debt consolidation</option></select></div>
          <div><label for="loan_int_rate">Interest rate (%)</label><input id="loan_int_rate" name="loan_int_rate" type="number" step="0.01" min="0" max="40" value="11.5" required></div>
          <div><label>Loan-to-income</label><div class="ratio"><b id="ratio">16.7%</b><span>auto</span></div></div>
        </div>
      </fieldset>

      <button type="submit" id="go">Check application</button>
    </form>
  </main>
</div>

<script>
const $ = id => document.getElementById(id);
const form = $('form'), CIRC = 553;

function updateRatio(){
  const i = parseFloat($('person_income').value), l = parseFloat($('loan_amnt').value);
  $('ratio').textContent = (i > 0 && l >= 0) ? (l / i * 100).toFixed(1) + '%' : '--';
}
['person_income','loan_amnt'].forEach(id => $(id).addEventListener('input', updateRatio));
updateRatio();

function countUp(target){
  const el = $('pct'), start = performance.now(), dur = 1200;
  (function tick(now){
    const t = Math.min((now - start) / dur, 1), e = 1 - Math.pow(1 - t, 3);
    el.textContent = (target * e).toFixed(0) + '%';
    if (t < 1) requestAnimationFrame(tick);
  })(start);
}

form.addEventListener('submit', async ev => {
  ev.preventDefault();
  const err = $('err'), btn = $('go');
  err.style.display = 'none';
  const payload = Object.fromEntries(new FormData(form).entries());
  btn.disabled = true; btn.textContent = 'Checking...';
  try {
    const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Something went wrong');

    const bar = $('bar'), v = $('verdict'), sub = $('sub');
    bar.className = 'bar ' + (d.approved ? 'ok' : 'no');
    bar.style.strokeDashoffset = CIRC * (1 - d.probability / 100);
    countUp(d.probability);
    v.textContent = d.approved ? 'Likely approved' : 'Likely declined';
    sub.textContent = 'Loan amount is ' + d.loan_percent_income + '% of annual income.';
    [v, sub].forEach(n => { n.classList.remove('reveal'); void n.offsetWidth; n.classList.add('reveal'); });
    if (window.innerWidth <= 860) $('result').scrollIntoView({behavior:'smooth', block:'center'});
  } catch (e) {
    err.textContent = e.message; err.style.display = 'block';
    err.scrollIntoView({behavior:'smooth', block:'center'});
  } finally {
    btn.disabled = false; btn.textContent = 'Check application';
  }
});
</script>
</body>
</html>
'''


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        income = num(data, "person_income", 1, 10_000_000)
        loan_amnt = num(data, "loan_amnt", 1, 10_000_000)
        row = {
            "person_age": num(data, "person_age", 18, 100),
            "person_gender": pick(GENDER, data.get("person_gender"), "gender"),
            "person_education": pick(EDUCATION, data.get("person_education"), "education"),
            "person_income": income,
            "person_emp_exp": num(data, "person_emp_exp", 0, 80),
            "person_home_ownership": pick(HOME, data.get("person_home_ownership"), "home ownership"),
            "loan_amnt": loan_amnt,
            "loan_intent": pick(INTENT, data.get("loan_intent"), "loan purpose"),
            "loan_int_rate": num(data, "loan_int_rate", 0, 40),
            "loan_percent_income": round(loan_amnt / income, 4),
            "cb_person_cred_hist_length": num(data, "cb_person_cred_hist_length", 0, 60),
            "credit_score": num(data, "credit_score", 300, 850),
            "previous_loan_defaults_on_file": pick(
                DEFAULTS, data.get("previous_loan_defaults_on_file"), "previous defaults"
            ),
        }
    except ValueError as e:
        return jsonify(error=str(e)), 400

    X = pd.DataFrame([row], columns=FEATURES)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][list(model.classes_).index(1)])

    return jsonify(
        approved=bool(pred == 1),
        probability=round(proba * 100, 1),
        loan_percent_income=round(row["loan_percent_income"] * 100, 1),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
