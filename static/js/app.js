/**
 * DiaGuard AI - Interactive Client Logic & Assessment Engine
 * Author: Antigravity AI
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. DOM Elements
    const form = document.getElementById("prediction-form");
    const modelSelector = document.getElementById("model-selector");
    const riskGauge = document.getElementById("risk-gauge");
    const riskPercent = document.getElementById("risk-percent");
    const riskStatus = document.getElementById("risk-status");
    const riskCategoryText = document.getElementById("risk-category-text");
    const factorsList = document.getElementById("factors-list");
    const guidanceText = document.getElementById("guidance-text");
    const resultEngine = document.getElementById("result-engine");

    // Slider inputs & badges
    const inputs = {
        Glucose: { input: document.getElementById("glucose-input"), badge: document.getElementById("glucose-badge"), unit: " mg/dL" },
        BMI: { input: document.getElementById("bmi-input"), badge: document.getElementById("bmi-badge"), unit: " kg/m²" },
        Age: { input: document.getElementById("age-input"), badge: document.getElementById("age-badge"), unit: " yrs" },
        Insulin: { input: document.getElementById("insulin-input"), badge: document.getElementById("insulin-badge"), unit: " µU/mL" },
        BloodPressure: { input: document.getElementById("bp-input"), badge: document.getElementById("bp-badge"), unit: " mm Hg" },
        Pregnancies: { input: document.getElementById("pregnancies-input"), badge: document.getElementById("pregnancies-badge"), unit: "" },
        SkinThickness: { input: document.getElementById("skin-input"), badge: document.getElementById("skin-badge"), unit: " mm" },
        DiabetesPedigreeFunction: { input: document.getElementById("pedigree-input"), badge: document.getElementById("pedigree-badge"), unit: "" }
    };

    // Update Badges on Slider Change
    Object.keys(inputs).forEach(key => {
        const item = inputs[key];
        if (item.input && item.badge) {
            item.input.addEventListener("input", (e) => {
                item.badge.textContent = `${e.target.value}${item.unit}`;
            });
        }
    });

    // 2. Profile Presets
    const presets = {
        healthy: {
            Pregnancies: 1,
            Glucose: 92,
            BloodPressure: 68,
            SkinThickness: 20,
            Insulin: 72,
            BMI: 21.8,
            DiabetesPedigreeFunction: 0.22,
            Age: 26
        },
        borderline: {
            Pregnancies: 3,
            Glucose: 128,
            BloodPressure: 78,
            SkinThickness: 28,
            Insulin: 120,
            BMI: 28.6,
            DiabetesPedigreeFunction: 0.45,
            Age: 39
        },
        diabetic: {
            Pregnancies: 6,
            Glucose: 172,
            BloodPressure: 86,
            SkinThickness: 38,
            Insulin: 195,
            BMI: 37.4,
            DiabetesPedigreeFunction: 0.88,
            Age: 52
        }
    };

    const presetButtons = document.querySelectorAll(".btn-preset");
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const presetKey = btn.dataset.preset;
            if (presets[presetKey]) {
                const values = presets[presetKey];
                Object.keys(values).forEach(k => {
                    if (inputs[k] && inputs[k].input) {
                        inputs[k].input.value = values[k];
                        inputs[k].badge.textContent = `${values[k]}${inputs[k].unit}`;
                    }
                });
                runPrediction();
            }
        });
    });

    // 3. Prediction API Dispatcher
    async function runPrediction() {
        const payload = {
            model: modelSelector.value,
            Pregnancies: parseFloat(inputs.Pregnancies.input.value),
            Glucose: parseFloat(inputs.Glucose.input.value),
            BloodPressure: parseFloat(inputs.BloodPressure.input.value),
            SkinThickness: parseFloat(inputs.SkinThickness.input.value),
            Insulin: parseFloat(inputs.Insulin.input.value),
            BMI: parseFloat(inputs.BMI.input.value),
            DiabetesPedigreeFunction: parseFloat(inputs.DiabetesPedigreeFunction.input.value),
            Age: parseFloat(inputs.Age.input.value)
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success) {
                renderPredictionResults(data);
            } else {
                console.error("Prediction error:", data.error);
            }
        } catch (err) {
            console.error("Network or inference error:", err);
        }
    }

    function renderPredictionResults(data) {
        // Update model engine badge
        resultEngine.textContent = data.model_used;

        // Animate circular gauge
        const pct = data.probability_percent;
        riskPercent.textContent = `${pct.toFixed(1)}%`;
        riskGauge.style.setProperty("--risk-pct", `${pct}%`);
        riskGauge.style.setProperty("--gauge-color", data.risk_color);

        // Update risk status pill
        riskCategoryText.textContent = `${data.risk_category} (${data.diagnosis})`;
        riskStatus.style.color = data.risk_color;
        riskStatus.style.borderColor = `${data.risk_color}55`;
        riskStatus.style.background = `${data.risk_color}18`;

        // Update factors
        factorsList.innerHTML = "";
        data.contributing_factors.forEach(f => {
            const li = document.createElement("li");
            li.className = "factor-item";
            const badgeClass = f.impact.toLowerCase() === "high" ? "high" : (f.impact.toLowerCase() === "moderate" ? "moderate" : "positive");
            li.innerHTML = `
                <span class="factor-badge ${badgeClass}">${f.impact}</span>
                <div>
                    <strong>${f.factor}:</strong>
                    <span>${f.desc}</span>
                </div>
            `;
            factorsList.appendChild(li);
        });

        // Guidance text
        guidanceText.textContent = data.guidance;
    }

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        runPrediction();
    });

    modelSelector.addEventListener("change", () => {
        runPrediction();
    });

    // 4. EDA Tabs Switching
    const edaTabs = document.querySelectorAll(".eda-tab");
    const edaImg = document.getElementById("eda-active-img");
    const edaCaption = document.getElementById("eda-caption-text");

    edaTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            edaTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            edaImg.src = tab.dataset.img;
            edaCaption.textContent = tab.dataset.desc;
        });
    });

    // 5. Model Evaluation Tabs Switching
    const modelTabs = document.querySelectorAll(".model-tab");
    const modelImg = document.getElementById("model-active-img");

    modelTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            modelTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            modelImg.src = tab.dataset.img;
        });
    });

    // Run initial prediction on load
    runPrediction();
});
