const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#xray-file");
const fileName = document.querySelector("#file-name");
const result = document.querySelector("#result");
const submitButton = form.querySelector('button[type="submit"]');

async function showHealth() {
  const response = await fetch("/api/health");
  const health = await response.json();

  if (!health.model_loaded) {
    result.className = "result warning";
    result.textContent = `Model is not loaded yet. Train it first, then restart Uvicorn. ${health.error ?? ""}`;
  }
}

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0]?.name ?? "Choose an X-ray image";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const data = new FormData(form);
  result.className = "result processing";
  result.setAttribute("aria-busy", "true");
  result.textContent = "Processing X-ray...";
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: data,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail ?? "Prediction failed.");
    }

    const confidence = (payload.confidence * 100).toFixed(2);
    result.className = "result";
    result.innerHTML = `<strong>${payload.label}</strong><br />Confidence: ${confidence}%`;
  } catch (error) {
    result.className = "result warning";
    result.textContent = error.message;
  } finally {
    result.setAttribute("aria-busy", "false");
    submitButton.disabled = false;
  }
});

showHealth();
