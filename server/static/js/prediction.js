document.addEventListener('DOMContentLoaded', function () {
    console.log("JavaScript is loaded!");

    let previousResultId = null; // Track the last known result ID

    async function loadResult() {
        try {
            console.log("Fetching latest prediction ID...");

            const response = await fetch('/get_latest_prediction_id');
            if (!response.ok) {
                console.error("Failed to fetch prediction ID:", response.status);
                throw new Error("Failed to get prediction ID");
            }

            const data = await response.json();
            const resultId = data.latest_id;

            // Only update if the ID has changed
            if (resultId !== previousResultId) {
                console.log("New result ID detected:", resultId);
                const fragment = await fetch('/result_fragment/' + resultId);
                if (!fragment.ok) {
                    console.error("Failed to fetch result fragment:", fragment.status);
                    throw new Error("Failed to load result fragment");
                }

                const html = await fragment.text();
                document.getElementById('result-container').innerHTML = html;

                // Update the cached result ID
                previousResultId = resultId;
            } else {
                console.log("No new prediction. Skipping update.");
            }

        } catch (err) {
            console.error("Error loading prediction:", err);
            document.getElementById('result-container').innerHTML = "<p style='color:red;'>Failed to load prediction.</p>";
        }
    }

    // Initial load
    loadResult();

    // Periodically check for updates (every 5 seconds)
    setInterval(loadResult, 5000);
});
