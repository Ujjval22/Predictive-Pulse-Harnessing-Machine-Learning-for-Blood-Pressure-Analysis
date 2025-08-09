function sendPrediction() {
    // List of expected form fields
    const fields = [
        'Gender', 'Age', 'Patient', 'Severity', 'BreathShortness',
        'VisualChanges', 'NoseBleeding', 'Whendiagnosed',
        'Systolic', 'Diastolic', 'ControlledDiet'
    ];

    // Gather data and validate inputs
    let data = {};
    for (let field of fields) {
        let val = parseFloat(document.getElementById(field).value);
        if (isNaN(val)) {
            alert(`Please enter a valid number for ${field}`);
            return;
        }
        data[field] = val;
    }

    // Send data to backend prediction API
    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(async response => {
        const text = await response.text();
        try {
            const json = JSON.parse(text);
            if (json.error) {
                document.getElementById('prediction-result').textContent = "Error: " + json.error;
            } else {
                document.getElementById('prediction-result').textContent = json.prediction;
            }
        } catch(e) {
            document.getElementById('prediction-result').textContent = "Unexpected response: " + text;
        }
    })
    .catch(err => {
        document.getElementById('prediction-result').textContent = "An unexpected error occurred: " + err;
    });
}
