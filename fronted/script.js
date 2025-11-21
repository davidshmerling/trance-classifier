function analyze() {
    const url = document.getElementById("urlInput").value.trim();
    const loading = document.getElementById("loading");
    const resultBox = document.getElementById("result");

    if (!url) {
        alert("אנא הדבק קישור ליוטיוב.");
        return;
    }

    loading.classList.remove("hidden");
    resultBox.classList.add("hidden");

    fetch("http://localhost:5000/predict", {  // ← לעדכן לכתובת השרת האמיתית
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url })
    })
    .then(res => res.json())
    .then(data => {
        loading.classList.add("hidden");

        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById("vector").textContent =
            `[ ${data.vector.join(", ")} ]`;

        document.getElementById("final").textContent = data.final_genre;

        resultBox.classList.remove("hidden");
    })
    .catch(err => {
        loading.classList.add("hidden");
        alert("שגיאה בתקשורת עם השרת.");
        console.error(err);
    });
}
